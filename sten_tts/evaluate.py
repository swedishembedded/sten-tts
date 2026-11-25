# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluation of speech synthesizers for its
# clients. If your team needs expertise in measuring text-to-speech quality,
# you can procure our services by sending an email to info@swedishembedded.com.

"""Does the release speak Swedish, in its voices, better than the base?

Everything is measured by models the training never touched, all served by
the brain server:

* every voice reads the evaluation sentences with each seed (`design` with the
  voice as preset speaker and `swedish` as language);
* the base reads the same sentences, cloning each voice from one of its
  held-out recordings (x-vector), with the language left to `auto`;
* the recogniser transcribes every clip, and the real held-out recordings of the
  voices (the floor);
* the speaker encoder embeds every clip and every held-out recording, for
  similarity and nearest-voice identification;
* the upstream `qwen-tts` package loads the release and speaks two sentences.

Clips and per-clip results go to `evaluation/clips.parquet`, the summary and
every gate to `evaluation/report.json`, and the model card is rewritten with
the numbers.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

from . import metrics
from .brainclient import BrainPool
from .config import Recipe, Workdir
from .modelcard import model_card

ASR_RATE = 16000


def pcm16k(wav: bytes) -> bytes:
    """A WAV as the recogniser's input: mono f32 little-endian PCM at 16 kHz."""
    x, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if rate != ASR_RATE:
        x = soxr.resample(x, rate, ASR_RATE)
    return np.asarray(x, dtype="<f4").tobytes()


def failure(wav: bytes, text: str, cap_seconds: float) -> str | None:
    """Why a clip is a failure: silent, cut by the length cap, or far too long for its text (a loop)."""
    x, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    seconds = len(x) / rate
    if seconds < 0.3 or float(np.sqrt(np.mean(x**2))) < 1e-3:
        return "silent"
    if seconds >= cap_seconds - 0.1:
        return "length_cap"
    if seconds > 0.25 * len(metrics.normalise(text)) + 3.0:  # ~4 characters a second is slow speech
        return "too_long"
    return None


def _voice_refs(work: Workdir) -> dict[str, list[dict]]:
    sel = {r["id"]: r for r in pq.read_table(work.selection / "selection.parquet").to_pylist() if r["split"] == "voice_test"}
    refs = defaultdict(list)
    for f in sorted(work.audio.glob("voice_test-*.parquet")):
        for r in pq.read_table(f, columns=["id", "wav"]).to_pylist():
            refs[sel[r["id"]]["voice"]].append({"id": r["id"], "wav": r["wav"], "text": sel[r["id"]]["text"]})
    return dict(refs)


def run(recipe: Recipe, work: Workdir, pool: BrainPool, concurrency: int, upstream_python: Path | None) -> dict:
    ev = recipe["evaluate"]
    sten = "local/" + recipe["train"]["install_as"]
    base, judge = recipe["base"]["repo"], recipe["judge"].get("served_as", recipe["judge"]["repo"])
    for model, action in ((sten, "design"), (base, "clone"), (base, "encode"), (judge, "transcribe")):
        pool.require_model(model, action)
    sentences = json.loads((work.evaluation / "sentences.json").read_text())
    refs = _voice_refs(work)
    voices = [v["name"] for v in recipe["voices"]]
    cap = recipe["train"]["max_frames"] / 12.5
    lang = recipe["language"]["name"]

    def transcribe(wav: bytes) -> str:
        return pool.get().run(judge, "transcribe", {"sample_rate": ASR_RATE, "language": recipe["evaluate"]["judge_language"]}, blobs={"audio": pcm16k(wav)},
                                  meta={"audio": {"media": "audio", "sample_rate": ASR_RATE}}).text("text")

    def xvector(wav: bytes) -> list[float]:
        return list(pool.get().encode_audio(wav, model=base).xvector)

    # The voices' own held-out recordings: the recogniser's floor and each voice's centroid.
    held = [(v, r) for v in voices for r in refs[v]]
    with ThreadPoolExecutor(concurrency) as ex:
        held_text = list(ex.map(lambda vr: transcribe(vr[1]["wav"]), held))
        held_x = list(ex.map(lambda vr: xvector(vr[1]["wav"]), held))
    floor = float(np.mean([metrics.cer(r["text"], t) for (_, r), t in zip(held, held_text)]))
    centroid = {v: np.mean([x for (vv, _), x in zip(held, held_x) if vv == v], axis=0).tolist() for v in voices}
    # How close a voice's own real recordings sit to its centroid (each left out of the centroid it is
    # compared with): the speaker encoder's scale, against which a synthesized clip is judged.
    real_similarity = {}
    for v in voices:
        xs = [x for (vv, _), x in zip(held, held_x) if vv == v]
        real_similarity[v] = float(np.median([metrics.cosine(x, np.mean([y for j, y in enumerate(xs) if j != i], axis=0).tolist()) for i, x in enumerate(xs)]))

    jobs = [("sten", v, s, seed) for v in voices for s in sentences for seed in ev["seeds"]]
    jobs += [("base", v, s, ev["seeds"][0]) for v in voices for s in sentences]

    def one(job):
        system, voice, sent, seed = job
        b = pool.get()
        if system == "sten":
            out = b.run(sten, "design", {"text": sent["text"], "speaker": voice, "lang": lang, "seed": seed})
        else:
            ref = refs[voice][0]
            out = b.run(base, "clone", {"text": sent["text"], "lang": "auto", "seed": seed}, blobs={"ref": ref["wav"]})
        wav = out.blobs["audio"]
        heard = transcribe(wav)
        x = xvector(wav)
        sims = {v: metrics.cosine(x, centroid[v]) for v in voices}
        return {"system": system, "voice": voice, "sentence": sent["id"], "seed": seed, "text": sent["text"], "heard": heard,
                "cer": metrics.cer(sent["text"], heard), "similarity": sims[voice], "nearest": max(sims, key=sims.get),
                "failure": failure(wav, sent["text"], cap), "wav": wav}

    rows = []
    with ThreadPoolExecutor(concurrency) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            rows.append(r)
            if i % 200 == 0:
                print(f"evaluate: {i}/{len(jobs)} clips", flush=True)
    work.evaluation.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), work.evaluation / "clips.parquet")

    st = [r for r in rows if r["system"] == "sten"]
    bs = [r for r in rows if r["system"] == "base"]
    per_voice = {v: {"cer": float(np.mean([r["cer"] for r in st if r["voice"] == v])),
                     "similarity": float(np.mean([r["similarity"] for r in st if r["voice"] == v])),
                     "real_similarity": real_similarity[v],
                     "identity_accuracy": float(np.mean([r["nearest"] == v for r in st if r["voice"] == v]))} for v in voices}
    first = ev["seeds"][0]
    pair_s = {(r["voice"], r["sentence"]): r["cer"] for r in st if r["seed"] == first}
    keys = sorted(pair_s)
    paired = metrics.paired_bootstrap([pair_s[k] for k in keys], [next(r["cer"] for r in bs if (r["voice"], r["sentence"]) == k) for k in keys], seed=recipe.seed)
    upstream = check_upstream(work, upstream_python, voices[:2], sentences[:1], lang)
    if upstream["status"] == "ok":
        upstream["cer"] = {Path(c).stem: metrics.cer(sentences[0]["text"], transcribe((work.root / c).read_bytes())) for c in upstream["clips"]}
        if max(upstream["cer"].values()) > floor + 2 * ev["gates"]["cer_over_floor_max"]:
            upstream["status"] = "speaks, but not intelligibly"
    g = ev["gates"]
    cer_all = float(np.mean([r["cer"] for r in st]))
    fail = float(np.mean([r["failure"] is not None for r in st]))
    gates = {
        "cer_over_floor_max": {"value": round(cer_all - floor, 4), "threshold": g["cer_over_floor_max"], "pass": cer_all - floor <= g["cer_over_floor_max"]},
        "cer_better_than_base": {"value": [round(c, 4) for c in paired["ci95"]], "threshold": "95% interval of Sten - base below 0", "pass": paired["ci95"][1] < 0},
        "voice_similarity_gap_max": {"value": round(max(m["real_similarity"] - m["similarity"] for m in per_voice.values()), 4), "threshold": g["voice_similarity_gap_max"],
                                     "pass": all(m["real_similarity"] - m["similarity"] <= g["voice_similarity_gap_max"] for m in per_voice.values())},
        "voice_identity_accuracy_min": {"value": round(min(m["identity_accuracy"] for m in per_voice.values()), 4), "threshold": g["voice_identity_accuracy_min"],
                                        "pass": all(m["identity_accuracy"] >= g["voice_identity_accuracy_min"] for m in per_voice.values())},
        "failure_rate_max": {"value": round(fail, 4), "threshold": g["failure_rate_max"], "pass": fail <= g["failure_rate_max"]},
        "upstream_loads": {"value": upstream["status"], "threshold": "loads and speaks", "pass": upstream["status"] == "ok"},
    }
    report = {"gates": gates, "voices": per_voice, "base": {"cer": float(np.mean([r["cer"] for r in bs]))}, "floor": {"cer": floor},
              "paired_cer": paired, "failures": {k: sum(1 for r in st if r["failure"] == k) for k in ("silent", "length_cap", "too_long")},
              "upstream": upstream, "clips": len(rows)}
    (work.evaluation / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    train_report = json.loads((work.train / "report.json").read_text())
    (work.release / "README.md").write_text(model_card(recipe, work, train_report))
    return {k: report[k] for k in ("gates", "base", "floor")}


UPSTREAM_SCRIPT = """
import json, sys, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel
release, out, voices, text, lang = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4], sys.argv[5]
m = Qwen3TTSModel.from_pretrained(release, device_map="cpu", dtype=torch.float32)
assert lang in m.model.get_supported_languages(), m.model.get_supported_languages()
for v in voices:
    assert v in m.model.get_supported_speakers(), m.model.get_supported_speakers()
    wavs, sr = m.generate_custom_voice(text=text, speaker=v, language=lang, max_new_tokens=400)
    sf.write(f"{out}/{v}.wav", wavs[0], sr)
print("ok")
"""


def check_upstream(work: Workdir, python: Path | None, voices: list[str], sentences: list[dict], lang: str) -> dict:
    """Load the release with the upstream package in its own interpreter and let two voices speak."""
    if python is None:
        return {"status": "not run: no --upstream-python given"}
    out = work.evaluation / "upstream"
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([str(python), "-c", UPSTREAM_SCRIPT, str(work.release), str(out), json.dumps(voices), sentences[0]["text"], lang],
                       capture_output=True, text=True, timeout=3600)
    if p.returncode != 0 or not p.stdout.strip().endswith("ok"):
        return {"status": "failed", "stderr": p.stderr[-2000:]}
    return {"status": "ok", "clips": sorted(str(f.relative_to(work.root)) for f in out.glob("*.wav"))}


if __name__ == "__main__":
    sys.exit("run through `sten-tts evaluate`")
