# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The clip checks evaluation applies before any model is asked anything."""

from __future__ import annotations

import io
import json

import numpy as np
import soundfile as sf

from sten_tts import evaluate


def _wav(seconds: float, amp: float, rate: int = 24000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    buf = io.BytesIO()
    sf.write(buf, (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32), rate, format="WAV")
    return buf.getvalue()


def test_failures_are_named() -> None:
    text = "Hej och välkommen till Sverige."
    assert evaluate.failure(_wav(2.0, 0.3), text, cap_seconds=30) is None
    assert evaluate.failure(_wav(2.0, 0.0), text, cap_seconds=30) == "silent"
    assert evaluate.failure(_wav(30.0, 0.3), text, cap_seconds=30) == "length_cap"
    assert evaluate.failure(_wav(20.0, 0.3), text, cap_seconds=30) == "too_long"


def test_the_recogniser_input_is_16k_mono_f32() -> None:
    pcm = evaluate.pcm16k(_wav(1.0, 0.3))
    assert len(pcm) == 16000 * 4


def _teacher(tmp_path, sentences: dict, voices: list[str], seed: int, real: dict) -> "object":
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = tmp_path / "teacher"
    d.mkdir()
    (d / "sentences.json").write_text(json.dumps(sentences))
    rows = [{"system": "base", "voice": v, "set": n, "sentence": s["id"], "seed": seed, "text": s["text"], "heard": s["text"], "cer": 0.1,
             "similarity": 0.9, "nearest": v, "failure": None, "wav": b"RIFF"} for n, items in sentences.items() for v in voices for s in items]
    rows += [{**r, "system": "sten", "seed": k} for r in rows for k in (1, 2)]
    pq.write_table(pa.Table.from_pylist(rows), d / "clips.parquet")
    (d / "report.json").write_text(json.dumps({"floor": {"cer": 0.147}, "voices": {v: {"real_similarity": x} for v, x in real.items()}}))
    return d


def test_the_base_clips_and_floor_are_carried_only_when_the_computation_is_provably_the_same(tmp_path) -> None:
    sentences = {"fleurs": [{"id": "fleurs:0", "text": "Hej."}, {"id": "fleurs:1", "text": "Då."}], "numbers": [{"id": "numbers:0", "text": "Ett."}]}
    voices, real = ["axel", "signe"], {"axel": 0.71, "signe": 0.64}
    d = _teacher(tmp_path, sentences, voices, seed=1, real=real)
    mine, other = tmp_path / "sentences.json", tmp_path / "other.json"
    mine.write_text(json.dumps(sentences))
    other.write_text(json.dumps({**sentences, "numbers": [{"id": "numbers:0", "text": "Två."}]}))
    ok = evaluate.carry(d, mine, voices, seed=1, real_similarity=real)
    assert ok is not None and ok["floor"] == 0.147 and len(ok["rows"]) == 6 and all(r["system"] == "base" for r in ok["rows"])
    assert all(r["wav"] is None for r in ok["rows"]), "the audio stays in the full model's clips, never copied"
    assert evaluate.carry(d, other, voices, 1, real) is None, "other sentences"
    assert evaluate.carry(d, mine, voices, 2, real) is None, "another seed"
    assert evaluate.carry(d, mine, voices + ["maja"], 1, {**real, "maja": 0.5}) is None, "another voice"
    assert evaluate.carry(d, mine, voices, 1, {**real, "signe": 0.6401}) is None, "other reference recordings or encoder"
