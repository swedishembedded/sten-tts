# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Two languages end to end: English records in the shards, English a base language of the job."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sten_tts import config, shard, shards, train

RECIPE = Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b-v2.yaml"


class _Enc:
    codes, xvector, text_ids, frames, codebooks, frame_rate, producers = [1] * 16, [0.5] * 4, [1, 2], 1, 16, 12.5, {"codec": "x"}


class _Pool:
    def require_model(self, model, action) -> None:
        pass

    def get(self):
        return self

    def encode_audio_batch(self, wavs, texts, model):
        return [_Enc() for _ in wavs]


def test_english_records_and_the_job_name_english_as_a_base_language(tmp_path: Path) -> None:
    rec = config.load(RECIPE)
    raw = copy.deepcopy(rec.raw)
    raw["selection"]["hours"] = 30.0 / 3600
    rec = config.Recipe(raw=raw, path=rec.path)
    work = config.Workdir(tmp_path)
    meta = [("nst:a", "swedish", "astrid"), ("rix:b", "swedish", None), ("lt:c", "english", None)]
    work.selection.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": i, "speaker": i, "source": i.split(":")[0], "text": "t", "language": lang, "voice": v} for i, lang, v in meta]),
                   work.selection / "selection.parquet")
    work.audio.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": i, "wav": b"w", "seconds": 10.0} for i, _, _ in meta]), work.audio / "train-0000.parquet")
    pq.write_table(pa.Table.from_pylist([{"id": "nst:a", "wav": b"w", "seconds": 1.0}]).slice(0, 0), work.audio / "validation-0000.parquet")
    pq.write_table(pa.Table.from_pylist([{"id": "nst:a", "wav": b"w", "seconds": 10.0}]), work.audio / "validation-0001.parquet")
    shards.build(rec, work, _Pool(), concurrency=1)
    s = shard.read(work.shards / "train.bttshrd")
    assert s["meta"]["languages"] == ["swedish", "english"]
    assert list(s["language"]) == [1, 0, 0], "records are in id order: lt:c (english), nst:a, rix:b"
    langs = json.loads(train.request(rec, work)["languages"])
    assert langs == {"swedish": {"new": True, "init_from": ["german", "english"]}, "english": {"new": False}}
