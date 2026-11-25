# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The training cut: exactly the budget, voices first, deterministic."""

from __future__ import annotations

import pytest

from sten_tts import shards


def test_cut_keeps_voices_and_stays_within_budget() -> None:
    clips = [{"id": f"c{i}", "seconds": 10.0} for i in range(1000)]
    meta = {c["id"]: {"voice": "astrid" if i < 50 else None} for i, c in enumerate(clips)}
    out = shards.cut(clips, meta, hours=1.0, seed=7)
    assert sum(c["seconds"] for c in out) == 3600.0
    assert {f"c{i}" for i in range(50)} <= {c["id"] for c in out}, "every voice clip is kept"
    assert out == shards.cut(list(reversed(clips)), meta, hours=1.0, seed=7)


def test_too_little_audio_is_refused() -> None:
    clips = [{"id": "a", "seconds": 10.0}]
    with pytest.raises(RuntimeError, match="select more"):
        shards.cut(clips, {"a": {"voice": None}}, hours=1.0, seed=1)


class _Enc:
    def __init__(self, i: int) -> None:
        self.codes, self.xvector, self.text_ids = [i] * 16, [0.5] * 4, [1, 2]
        self.frames, self.codebooks, self.frame_rate, self.producers = 1, 16, 12.5, {"codec": "sha256:x"}


def test_an_interrupted_encode_resumes_from_its_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shards, "CHUNK", 3)
    clips = [{"id": f"c{i}"} for i in range(8)]
    calls = []

    def flaky(c):
        if len(calls) == 5:
            raise RuntimeError("server went away")
        calls.append(c["id"])
        return _Enc(int(c["id"][1:]))

    with pytest.raises(RuntimeError):
        shards._encode_missing(clips, tmp_path, "train", flaky, concurrency=1)
    first = set(calls)
    calls.clear()
    done = shards._encode_missing(clips, tmp_path, "train", lambda c: (calls.append(c["id"]), _Enc(int(c["id"][1:])))[1], concurrency=1)
    assert set(done) == {c["id"] for c in clips}
    assert len(first) >= 3 and not set(calls) & {f"c{i}" for i in range(3)}, "the saved chunk is not encoded again"
