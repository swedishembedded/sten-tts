# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Cutting long recordings at pauses and giving each piece its span of the known text."""

from __future__ import annotations

import numpy as np

from sten_tts import segment

CFG = {"min_seconds": 2.0, "max_seconds": 20.0, "min_pause_seconds": 0.25, "silence_db": 30.0}


def _speech(pattern: list[tuple[float, bool]], rate: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(0)
    parts = [(0.3 if loud else 0.0003) * rng.standard_normal(int(sec * rate)) for sec, loud in pattern]
    return np.concatenate(parts).astype(np.float32)


def test_pauses_are_found_and_segments_end_mid_pause_within_bounds() -> None:
    x = _speech([(5, True), (0.6, False), (4, True), (0.3, False), (12, True), (1.0, False), (3, True)])
    gaps = segment.pauses(x, 16000, CFG)
    assert [(round(a, 2), round(b, 2)) for a, b in gaps] == [(5.0, 5.6), (9.6, 9.9), (21.9, 22.9)]
    cuts = segment.cuts(len(x) / 16000, gaps, CFG)
    # From 0 the window 2..20 s holds the pauses at 5.3 and 9.75: the longer one wins.
    assert [(round(c.start, 2), round(c.end, 2), c.reject) for c in cuts] == [(0.0, 5.3, None), (5.3, 22.4, None), (22.4, 25.9, None)]
    assert all(CFG["min_seconds"] <= c.end - c.start <= CFG["max_seconds"] for c in cuts)
    assert cuts == segment.cuts(len(x) / 16000, gaps, CFG), "deterministic"


def test_a_stretch_without_a_pause_is_rejected_and_a_short_tail_too() -> None:
    cuts = segment.cuts(40.0, [(30.0, 30.5), (39.0, 39.4)], CFG)
    assert [(c.start, c.end, c.reject) for c in cuts] == [(0.0, 30.25, "no_pause"), (30.25, 39.2, None), (39.2, 40.0, "too_short")]


def test_spans_come_from_the_book_text_with_punctuation() -> None:
    text = "Hej, sa hon. Det var en gång en LibriVox-inspelning; slut!"
    segs = [
        {"id": "a", "start": 0.0, "end": 2.0, "reject": None, "heard": "hej sa hon"},
        {"id": "b", "start": 2.0, "end": 5.0, "reject": None, "heard": "det var en gång en librivox inspelning"},
        {"id": "c", "start": 5.0, "end": 6.0, "reject": None, "heard": "slutt"},
    ]
    out = segment.assign(text, segs, max_cer=0.10)
    assert [s["text"] for s in out] == ["Hej, sa hon.", "Det var en gång en LibriVox-inspelning;", "slut!"]
    assert [s["reject"] for s in out] == [None, None, "cer"], "'slutt' is one edit in four letters"


def test_a_word_neither_neighbour_recognised_rejects_both() -> None:
    text = "ett två tre fyra fem sex"
    segs = [
        {"id": "a", "start": 0.0, "end": 2.0, "reject": None, "heard": "ett två"},
        {"id": "b", "start": 2.0, "end": 4.0, "reject": None, "heard": "fyra fem sex"},
    ]
    assert [s["reject"] for s in segment.assign(text, segs, max_cer=0.5)] == ["edge_words", "edge_words"]
    # Not when the stretch between them was never recognised (a rejected cut lies there).
    segs = [segs[0], {"id": "x", "start": 2.0, "end": 3.0, "reject": "no_pause", "heard": None}, {**segs[1], "start": 3.0}]
    assert [s["reject"] for s in segment.assign(text, segs, max_cer=0.5)] == [None, "no_pause", None]


class _Out:
    def __init__(self, text: str) -> None:
        self._t = text

    def text(self, name: str) -> str:
        return self._t


class _Pool:
    def __init__(self) -> None:
        self.calls = 0

    def require_model(self, model: str, action: str) -> None:
        assert (model, action) == ("brain/qwen3asr", "transcribe")

    def get(self):
        return self

    def run(self, model, action, params, blobs, meta):
        assert params["language"] == "Swedish" and params["sample_rate"] == 16000
        self.calls += 1
        return _Out("ett två")


def test_the_stage_writes_kept_segments_and_resumes_without_recognising_again(tmp_path) -> None:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf

    from sten_tts import config

    x = _speech([(3, True), (0.5, False), (3, True)])
    buf = io.BytesIO()
    sf.write(buf, x, 16000, format="WAV", subtype="PCM_16")
    raw = tmp_path / "raw" / "librivox" / "data"
    raw.mkdir(parents=True)
    row = {"__key__": "bok_chunk_0000", "audio": {"bytes": buf.getvalue(), "path": None}, "text": "Ett, två. Ett två!", "speaker_id": "bok",
           "gender": "female", "duration_sec": len(x) / 16000}
    pq.write_table(pa.Table.from_pylist([row]), raw / "data.parquet", row_group_size=1)
    recipe = config.load(config.Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b-v2.yaml")
    work, pool = config.Workdir(tmp_path), _Pool()
    report = segment.run(recipe, work, pool, concurrency=2, jobs=1)
    assert report["outcome"]["kept"]["segments"] == 2 and pool.calls == 2
    rows = pq.read_table(work.segments / "segments.parquet").to_pylist()
    assert [r["text"] for r in rows] == ["Ett, två.", "Ett två!"]
    audio = pq.read_table(work.segments / rows[0]["file"]).to_pylist()
    assert audio[rows[1]["file_row"]]["id"] == rows[1]["id"]
    segment.run(recipe, work, pool, concurrency=2, jobs=1)
    assert pool.calls == 2, "recognitions come from the cache on a rerun"


def test_recognitions_before_a_failure_are_kept(tmp_path, monkeypatch) -> None:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest
    import soundfile as sf

    x = _speech([(3, True), (0.5, False), (3, True), (0.5, False), (3, True)])
    buf = io.BytesIO()
    sf.write(buf, x, 16000, format="WAV", subtype="PCM_16")
    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist([{"audio": {"bytes": buf.getvalue(), "path": None}}]), path)
    segs = [{"id": f"s{k}", "row": 0, "start": 3.25 * k, "end": 3.25 * (k + 1), "reject": None} for k in range(3)]
    calls = []

    def flaky(pcm: bytes) -> str:
        calls.append(len(pcm))
        if len(calls) == 3:
            raise RuntimeError("server went away")
        return "ett"

    with pytest.raises(RuntimeError):
        segment.recognise(str(path), segs, tmp_path, flaky, concurrency=1)
    assert len(segment._cached(tmp_path)) == 2
