# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in building balanced
# speech corpora, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Deterministic selection of the training corpus and its held-out sets.

Given the inventories and the recipe (seed included) the selection is a pure
function: the same inputs give byte-identical `selection.parquet`. Order is
never taken from the files; every tie is broken by a seeded hash of an id.

Splits:

* `unseen_test`: every utterance of whole speakers kept out of training,
  balanced by gender, per source;
* `voice_test`: per named voice, utterances whose text then appears nowhere
  in training (the voice is known, the sentence is not);
* `validation`: a fraction of the remaining training speakers' utterances,
  for checkpoint choice;
* `train`: the rest, filled to the recipe's hours per source with a per-speaker
  cap for auxiliary speakers and round-robin over (gender, region) groups, so
  no speaker or region dominates.

A text that appears in any evaluation set (the FLEURS sentences included) never
appears in `train` or `validation`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

from .config import Recipe, Workdir


def rank(seed: int, key: str) -> int:
    """A seeded, platform-independent ordering key."""
    return int.from_bytes(hashlib.blake2b(f"{seed}\x00{key}".encode(), digest_size=8).digest(), "big")


def norm_text(text: str) -> str:
    """The comparison form of a transcript: lowercase letters and digits only."""
    return re.sub(r"[^0-9a-zåäöéü]+", " ", text.lower()).strip()


@dataclass
class Utt:
    source: str
    id: str
    file: str
    row: int
    speaker: str
    gender: str
    age: str | None
    region: str | None
    text: str
    duration: float
    quality: str | None


def _eligible(u: Utt, sel: dict) -> str | None:
    """Why an utterance is not eligible, or None."""
    d = sel["duration_seconds"]
    if not d["min"] <= u.duration <= d["max"]:
        return "duration"
    if not norm_text(u.text):
        return "no_text"
    if u.source == "nst":
        words = norm_text(u.text).split()
        if len(words) < sel["nst"]["min_words"]:
            return "too_few_words"
        if sum(len(w) == 1 for w in words) * 2 >= len(words):
            return "spelled"  # NST's letter-by-letter items: a spelling task, not a sentence
        if u.gender not in ("female", "male"):
            return "gender_unspecified"
    if u.source == "rixvox":
        q = json.loads(u.quality)
        if q["is_silence"]:
            return "silence"
        if q["wer_whisper"] is None or q["wer_whisper"] > sel["rixvox"]["max_wer_whisper"]:
            return "wer"
        if q["lang_prob_sv"] is None or q["lang_prob_sv"] < sel["rixvox"]["min_lang_prob_sv"]:
            return "language"
        if u.gender not in ("female", "male"):
            return "gender_unspecified"
    return None


def _take_capped(utts: list[Utt], seed: int, cap_seconds: float) -> list[Utt]:
    out, total = [], 0.0
    for u in sorted(utts, key=lambda u: rank(seed, u.id)):
        if total + u.duration > cap_seconds:
            continue
        out.append(u)
        total += u.duration
    return out


def _round_robin(groups: dict[tuple, list[list[Utt]]], seed: int, budget: float) -> list[Utt]:
    """Whole speakers' capped utterance lists, one group at a time, until the budget is met."""
    queues = {g: sorted(lists, key=lambda l: rank(seed, l[0].speaker)) for g, lists in groups.items() if lists}
    order = sorted(queues, key=lambda g: rank(seed, repr(g)))
    taken, total = [], 0.0
    while total < budget and any(queues[g] for g in order):
        for g in order:
            if queues[g] and total < budget:
                lst = queues[g].pop(0)
                taken.extend(lst)
                total += sum(u.duration for u in lst)
    return taken


def select(recipe: Recipe, work: Workdir, eval_texts: set[str]) -> dict:
    seed, sel = recipe.seed, recipe["selection"]
    utts: list[Utt] = []
    for name in sel["share"]:
        t = pq.read_table(work.inventory / f"{name}.parquet").to_pylist()
        utts.extend(Utt(**r) for r in t)
    rejected: dict[str, int] = defaultdict(int)
    ok: list[Utt] = []
    for u in utts:
        why = _eligible(u, sel)
        if why:
            rejected[why] += 1
        else:
            ok.append(u)

    by_speaker: dict[str, list[Utt]] = defaultdict(list)
    for u in ok:
        by_speaker[u.speaker].append(u)
    speaker_meta = {s: (us[0].source, us[0].gender, us[0].region) for s, us in by_speaker.items()}

    # Named voices: the NST speaker with the most eligible audio in the voice's gender and region.
    voices, used = {}, set()
    for v in recipe["voices"]:
        cands = [s for s, (src, g, r) in speaker_meta.items() if src == "nst" and g == v["gender"] and r == v["region"] and s not in used]
        if not cands:
            raise RuntimeError(f"voice {v['name']}: no NST speaker is {v['gender']} from {v['region']}")
        best = max(cands, key=lambda s: (round(sum(u.duration for u in by_speaker[s]), 1), rank(seed, s)))
        used.add(best)
        voices[v["name"]] = {"speaker": best, "gender": v["gender"], "region": v["region"], "eligible_seconds": sum(u.duration for u in by_speaker[best])}
    voice_of = {meta["speaker"]: name for name, meta in voices.items()}

    # Whole held-out speakers, balanced by gender, never a voice.
    split: dict[str, str] = {}
    held = sel["held_out"]
    for src, n in held["unseen_speakers"].items():
        for gender in ("female", "male"):
            pool = sorted((s for s, (sr, g, _) in speaker_meta.items() if sr == src and g == gender and s not in voice_of), key=lambda s: rank(seed, "unseen" + s))
            for s in pool[: n // 2]:
                for u in by_speaker[s]:
                    split[u.id] = "unseen_test"

    # Per voice, test utterances, preferring texts few others read: every reading of a
    # test text is then banned from training, so a rare text costs the least audio.
    text_count: dict[str, int] = defaultdict(int)
    for u in ok:
        text_count[norm_text(u.text)] += 1
    for name, meta in voices.items():
        pool = sorted(by_speaker[meta["speaker"]], key=lambda u: (text_count[norm_text(u.text)], rank(seed, "voicetest" + u.id)))
        if len(pool) < 2 * held["voice_test_utterances"]:
            raise RuntimeError(f"voice {name}: {len(pool)} eligible utterances, too few to hold out {held['voice_test_utterances']}")
        for u in pool[: held["voice_test_utterances"]]:
            split[u.id] = "voice_test"
    banned = set(eval_texts) | {norm_text(u.text) for u in ok if split.get(u.id) in ("voice_test", "unseen_test")}

    # Training candidates: no banned text, each text at most three times.
    seen_text: dict[str, int] = defaultdict(int)
    cand: dict[str, list[Utt]] = defaultdict(list)
    for u in sorted(ok, key=lambda u: rank(seed, "text" + u.id)):
        if u.id in split:
            continue
        t = norm_text(u.text)
        if t in banned:
            rejected["eval_text"] += 1
            continue
        if seen_text[t] >= 3:
            rejected["repeated_text"] += 1
            continue
        seen_text[t] += 1
        cand[u.speaker].append(u)

    total_budget = sel["hours"] * 3600 * sel["oversample"]
    chosen: list[Utt] = []
    for src, share in sel["share"].items():
        budget = total_budget * share
        if src == "nst":
            for s in voice_of:
                chosen.extend(cand[s])
                budget -= sum(u.duration for u in cand[s])
        cap = sel[src]["cap_minutes_per_speaker"] * 60
        groups: dict[tuple, list[list[Utt]]] = defaultdict(list)
        for s, us in cand.items():
            src_s, g, r = speaker_meta[s]
            if src_s != src or s in voice_of:
                continue
            capped = _take_capped(us, seed, cap)
            if capped:
                groups[(g, r if src == "nst" else None)].append(capped)
        chosen.extend(_round_robin(groups, seed, budget))

    for u in chosen:
        is_val = (rank(seed, "val" + u.id) % 10_000) < held["validation_fraction"] * 10_000
        split[u.id] = "validation" if is_val else "train"

    rows = []
    for u in sorted(ok, key=lambda u: u.id):
        if u.id in split:
            rows.append({**u.__dict__, "split": split[u.id], "voice": voice_of.get(u.speaker)})
    work.selection.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), work.selection / "selection.parquet")
    report = _report(rows, voices, dict(rejected))
    (work.selection / "voices.json").write_text(json.dumps(voices, indent=1, ensure_ascii=False))
    (work.selection / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def _report(rows: list[dict], voices: dict, rejected: dict) -> dict:
    hours: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    speakers: dict[str, set] = defaultdict(set)
    for r in rows:
        hours[r["split"]][r["source"]] += r["duration"] / 3600
        hours[r["split"]]["gender:" + r["gender"]] += r["duration"] / 3600
        if r["source"] == "nst":
            hours[r["split"]]["region:" + str(r["region"])] += r["duration"] / 3600
        speakers[r["split"]].add(r["speaker"])
    return {
        "hours": {k: {kk: round(vv, 3) for kk, vv in sorted(v.items())} for k, v in hours.items()},
        "speakers": {k: len(v) for k, v in speakers.items()},
        "utterances": {s: sum(1 for r in rows if r["split"] == s) for s in speakers},
        "voices": {k: {**v, "train_hours": round(sum(r["duration"] for r in rows if r["voice"] == k and r["split"] == "train") / 3600, 3)} for k, v in voices.items()},
        "rejected": rejected,
    }
