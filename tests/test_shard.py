# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The shard writer produces the layout brain reads: magic, header, aligned columns."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sten_tts import shard


def _writer() -> shard.ShardWriter:
    w = shard.ShardWriter(codebooks=2, codebook_size=8, xvector_dim=3, frame_rate=12.5, languages=["swedish"], voices=["astrid"])
    w.push(id={"id": "a", "speaker": "s", "source": "t", "text": "hej"}, text_ids=[1, 2, 3], codes=[[0, 1], [2, 3]], xvector=[0.1, 0.2, 0.3], voice="astrid", language="swedish", complete=True)
    w.push(id={"id": "b", "speaker": "s", "source": "t", "text": "då"}, text_ids=[4], codes=[[7, 7]], xvector=[1, 1, 1], voice=None, language="swedish", complete=False)
    return w


def test_round_trip_matches_the_brain_layout(tmp_path) -> None:
    p = tmp_path / "x.bttshrd"
    _writer().write(p)
    s = shard.read(p)
    assert s["meta"]["records"] == 2 and s["meta"]["format"] == "brain-tts-shard-v1"
    assert s["text"].tolist() == [1, 2, 3, 4] and s["text_off"].tolist() == [0, 3, 4]
    assert s["codes"].tolist() == [0, 1, 2, 3, 7, 7] and s["codes_off"].tolist() == [0, 2, 3]
    assert s["voice"].tolist() == [0, shard.NO_VOICE] and s["complete"].tolist() == [1, 0]
    assert [json.loads(l)["id"] for l in bytes(s["ids"]).decode().splitlines()] == ["a", "b"]
    raw = p.read_bytes()
    header = json.loads(raw[16 : 16 + int.from_bytes(raw[8:16], "little")])
    assert all(c["offset"] % 8 == 0 for c in header["columns"].values())
    with pytest.raises(FileExistsError):
        _writer().write(p)


def test_records_outside_the_meta_are_refused() -> None:
    w = _writer()
    with pytest.raises(ValueError, match="codebook"):
        w.push(id={"id": "c"}, text_ids=[1], codes=[[8, 0]], xvector=[0, 0, 0], voice=None, language="swedish", complete=True)
    with pytest.raises(ValueError, match="x-vector"):
        w.push(id={"id": "c"}, text_ids=[1], codes=[[1, 0]], xvector=[0, np.nan, 0], voice=None, language="swedish", complete=True)
