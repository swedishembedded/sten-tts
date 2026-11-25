# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The selection's contract on a synthetic five-source inventory: deterministic, disjoint
held-out sets, no evaluation text in training, shares, caps, languages and voices honoured."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from sten_tts import config, select

RECIPE = Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b-v2.yaml"
REGIONS = ["Mellansverige", "Stockholm med omnejd", "Göteborg med omnejd", "Västra sydsverige", "Norrland", "Dalarna med omnejd"]
SOURCES = ("nst", "rixvox", "librivox", "fleurs", "libritts_r")


def _recipe(hours: float = 1.0) -> config.Recipe:
    r = config.load(RECIPE)
    raw = copy.deepcopy(r.raw)
    raw["selection"]["hours"] = hours
    return config.Recipe(raw=raw, path=r.path)


def _row(source: str, sid: str, k: int, speaker: str, gender: str, region, text: str, duration: float, quality=None, language="swedish") -> dict:
    return {"source": source, "id": f"{source}:{sid}:{k}", "file": "f", "row": k, "speaker": f"{source}:{speaker}", "gender": gender, "age": None,
            "region": region, "language": language, "text": text, "duration": duration, "quality": quality}


def _inventory(root: Path) -> None:
    rows = []
    for si in range(60):
        region = REGIONS[si % len(REGIONS)]
        gender = "female" if (si // len(REGIONS)) % 2 == 0 else "male"
        for k in range(40 + si):  # speakers differ in how much they recorded
            rows.append(_row("nst", str(si), k, str(si), gender, region, f"mening nummer {k % 50} av talare {si % 7}", 4.0))
    rows.append(_row("nst", "spelled", 0, "0", "female", REGIONS[0], "K O N T R O L L", 4.0))
    for si in range(20):
        for k in range(30):
            q = json.dumps({"wer_whisper": 0.05 if k % 5 else 0.5, "lang_prob_sv": 0.99, "is_silence": False, "year": 2023})
            rows.append(_row("rixvox", str(si), k, str(si), "female" if si % 2 else "male", None, f"riksdagen debatterar fråga {k} och {si}", 12.0, q))
    for si in range(3):  # LibriVox is short: its share is left to the others
        for k in range(10):
            rows.append(_row("librivox", str(si), k, f"bok{si}", "male" if si else "female", None, f"i boken {si} står mening {k}", 10.0))
    for k in range(50):
        rows.append(_row("fleurs", str(k), 0, str(k), "female" if k % 2 else "male", None, f"fleurs mening {k // 2}", 8.0))
    for si in range(10):
        for k in range(40):
            rows.append(_row("libritts_r", str(si), k, str(si), "unspecified", None, f"an english sentence {k} by {si}", 6.0, language="english"))
    (root / "inventory").mkdir(parents=True)
    t = pa.Table.from_pylist(rows)
    for src in SOURCES:
        pq.write_table(t.filter(pc.equal(t["source"], src)), root / "inventory" / f"{src}.parquet")


def _voices(work: config.Workdir, recipe: config.Recipe) -> dict:
    """What the voices stage would write: here, the first speaker of each slot."""
    inv = pq.read_table(work.inventory / "nst.parquet").to_pylist()
    out = {}
    for v in recipe["voices"]:
        s = next(r["speaker"] for r in inv if r["gender"] == v["gender"] and r["region"] == v["region"] and r["speaker"] not in [o["speaker"] for o in out.values()])
        out[v["name"]] = {"speaker": s, "gender": v["gender"], "region": v["region"], "eligible_seconds": 1.0, "score": 0.0}
    work.selection.mkdir(parents=True, exist_ok=True)
    (work.selection / "voices.json").write_text(json.dumps(out))
    return out


@pytest.fixture
def work(tmp_path: Path) -> config.Workdir:
    _inventory(tmp_path)
    w = config.Workdir(tmp_path)
    _voices(w, _recipe())
    return w


def _rows(work: config.Workdir) -> list[dict]:
    return pq.read_table(work.selection / "selection.parquet").to_pylist()


def test_selection_is_deterministic(work: config.Workdir) -> None:
    select.select(_recipe(), work, {"mening nummer 1 av talare 1"})
    first = (work.selection / "selection.parquet").read_bytes()
    select.select(_recipe(), work, {"mening nummer 1 av talare 1"})
    assert (work.selection / "selection.parquet").read_bytes() == first


def test_held_out_sets_are_disjoint_from_training(work: config.Workdir) -> None:
    banned = select.norm_text("mening nummer 3 av talare 2")
    select.select(_recipe(), work, {banned})
    rows = _rows(work)
    train = [r for r in rows if r["split"] in ("train", "validation")]
    unseen = [r for r in rows if r["split"] == "unseen_test"]
    assert {r["source"] for r in unseen} == {"nst", "rixvox", "librivox", "libritts_r"}, "whole speakers held out per source"
    assert not {r["speaker"] for r in unseen} & {r["speaker"] for r in train}
    train_texts = {select.norm_text(r["text"]) for r in train}
    assert banned not in train_texts, "an evaluation text never trains"
    voice_test = [r for r in rows if r["split"] == "voice_test"]
    assert voice_test and not {select.norm_text(r["text"]) for r in voice_test} & train_texts
    assert "nst:spelled:0" not in {r["id"] for r in rows}, "letter-by-letter items are not sentences"


def test_held_out_speakers_are_never_the_large_ones(work: config.Workdir) -> None:
    select.select(_recipe(), work, set())
    rows = _rows(work)
    voices = {v["speaker"] for v in json.loads((work.selection / "voices.json").read_text()).values()}
    secs: dict[str, float] = defaultdict(float)
    gender = {}
    for r in pq.read_table(work.inventory / "nst.parquet").to_pylist():
        if r["speaker"] not in voices and r["id"] != "nst:spelled:0":
            secs[r["speaker"]] += r["duration"]
            gender[r["speaker"]] = r["gender"]
    held = {r["speaker"] for r in rows if r["split"] == "unseen_test" and r["source"] == "nst"}
    assert len(held) == 12 and {gender[s] for s in held} == {"female", "male"}
    for s in held:
        same = sorted(v for k, v in secs.items() if gender[k] == gender[s])
        assert secs[s] <= same[(len(same) - 1) // 2], f"{s} has more than the median audio of its gender"


def test_voices_come_from_the_voices_stage_and_english_is_never_a_voice(work: config.Workdir) -> None:
    rec = _recipe()
    chosen = _voices(work, rec)
    select.select(rec, work, set())
    rows = _rows(work)
    for name, v in chosen.items():
        mine = [r for r in rows if r["speaker"] == v["speaker"]]
        assert mine and all(r["voice"] == name for r in mine)
    english = [r for r in rows if r["source"] == "libritts_r"]
    assert english and all(r["language"] == "english" and r["voice"] is None for r in english)
    assert {r["language"] for r in rows if r["source"] != "libritts_r"} == {"swedish"}
    stale = {**chosen, "astrid": {**chosen["astrid"], "speaker": chosen["erik"]["speaker"]}}
    (work.selection / "voices.json").write_text(json.dumps(stale))
    with pytest.raises(RuntimeError, match="voices stage"):
        select.select(rec, work, set())


def test_shares_caps_and_a_short_source_left_to_the_others(work: config.Workdir) -> None:
    rec = _recipe(hours=1.0)
    report = select.select(rec, work, set())
    rows = [r for r in _rows(work) if r["split"] in ("train", "validation")]
    sel = rec["selection"]
    budget = sel["hours"] * sel["oversample"] * 3600
    by_source: dict[str, float] = defaultdict(float)
    for r in rows:
        by_source[r["source"]] += r["duration"]
    assert by_source["librivox"] < budget * sel["share"]["librivox"], "the synthetic LibriVox is too small for its share"
    assert report["shortfall_hours_moved"] > 0
    assert sum(by_source.values()) >= budget, "its shortfall is taken from the other sources"
    assert by_source["libritts_r"] >= budget * sel["share"]["libritts_r"], "English keeps its share"
    per_speaker: dict[str, float] = defaultdict(float)
    for r in rows:
        if r["voice"] is None:
            per_speaker[r["speaker"]] += r["duration"]
    for s, secs in per_speaker.items():
        assert secs <= sel[s.split(":")[0]]["cap_minutes_per_speaker"] * 60 + 1e-6, f"{s}: {secs} s over the cap"
    assert all(json.loads(r["quality"])["wer_whisper"] <= sel["rixvox"]["max_wer_whisper"] for r in rows if r["source"] == "rixvox")
    assert report["hours"]["train"]["language:english"] > 0 and report["speakers_per_source"]["train"]["fleurs"] > 0


def test_a_bad_recipe_is_refused_naming_the_key(tmp_path: Path) -> None:
    text = RECIPE.read_text()
    for bad, key in ((text.replace("seed: 20261125", "seed: 20261125\nseeed: 1"), "seeed"),
                     (text.replace("  libritts_r:\n    cap_minutes_per_speaker: 12.0", "  libritts_r:\n    cap_minutes: 12.0"), "cap_minutes")):
        p = tmp_path / "r.yaml"
        p.write_text(bad)
        with pytest.raises(config.RecipeError, match=key):
            config.load(p)
