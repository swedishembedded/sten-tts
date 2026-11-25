# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The v2 sources' inventories: every row's language, speaker and duration, and a locator
the audio stage reads the same recording back from."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from sten_tts import audio, config, inventory


def _wav(seconds: float, rate: int, f: float = 220.0) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _fleurs(raw: Path) -> None:
    (raw / "data" / "sv_se" / "audio").mkdir(parents=True)
    for split, n in (("train", 3), ("dev", 1), ("test", 2)):
        lines = []
        with tarfile.open(raw / "data" / "sv_se" / "audio" / f"{split}.tar.gz", "w:gz") as tar:
            for k in range(n):
                data = _wav(1.0 + k, 16000)
                info = tarfile.TarInfo(f"{split}/{split}{k}.wav")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                lines.append(f"{100 + k}\t{split}{k}.wav\tMening {k} på svenska.\tmening {k} på svenska\tm e n\t{16000 * (1 + k)}\t{'FEMALE' if k % 2 else 'MALE'}")
        (raw / "data" / "sv_se" / f"{split}.tsv").write_text("\n".join(lines) + "\n")


def test_fleurs_trains_on_train_and_dev_only_and_reads_back(tmp_path: Path) -> None:
    work = config.Workdir(tmp_path)
    _fleurs(work.raw("fleurs"))
    rows = inventory.fleurs(work, jobs=1).to_pylist()
    assert sorted(r["id"] for r in rows) == ["fleurs:dev0", "fleurs:train0", "fleurs:train1", "fleurs:train2"], "the test split is evaluation only"
    r = next(r for r in rows if r["id"] == "fleurs:train1")
    assert (r["language"], r["gender"], r["text"], r["duration"]) == ("swedish", "female", "Mening 1 på svenska.", 2.0)
    data = audio._load_rows(inventory.audio_path(work, "fleurs", "fleurs", r["file"]), "fleurs", [r["row"]])[0]
    assert sf.info(io.BytesIO(data)).duration == 2.0


def test_libritts_r_is_english_at_24k(tmp_path: Path) -> None:
    work = config.Workdir(tmp_path)
    d = work.raw("libritts_r") / "data" / "train.clean.100"
    d.mkdir(parents=True)
    rows = [{"audio": {"bytes": _wav(1.5, 24000), "path": None}, "text_original": " Hello there. ", "text_normalized": "Hello there.",
             "speaker_id": "1088", "chapter_id": "7", "id": "1088_7_0", "path": "x"}]
    pq.write_table(pa.Table.from_pylist(rows), d / "train.clean.100-00000-of-00001.parquet")
    (r,) = inventory.libritts_r(work, jobs=1).to_pylist()
    assert (r["language"], r["speaker"], r["text"], r["duration"]) == ("english", "libritts_r:1088", "Hello there.", 1.5)
    data = audio._load_rows(inventory.audio_path(work, "libritts_r", "libritts-r", r["file"]), "libritts-r", [r["row"]])[0]
    assert sf.info(io.BytesIO(data)).samplerate == 24000


def test_librivox_utterances_are_the_kept_segments(tmp_path: Path) -> None:
    work = config.Workdir(tmp_path)
    work.segments.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": "librivox:b:0000000", "wav": _wav(3.0, 16000)}]), work.segments / "audio-0000.parquet")
    seg = [{"id": "librivox:b:0000000", "speaker": "librivox:b", "gender": "male", "duration": 3.0, "text": "Ja, sa han.", "cer": 0.02,
            "reject": None, "file": "audio-0000.parquet", "file_row": 0},
           {"id": "librivox:b:0003000", "speaker": "librivox:b", "gender": "male", "duration": 4.0, "text": None, "cer": None,
            "reject": "cer", "file": None, "file_row": None}]
    pq.write_table(pa.Table.from_pylist(seg), work.segments / "segments.parquet")
    (r,) = inventory.librivox(work, jobs=1).to_pylist()
    assert (r["id"], r["language"], r["text"], json.loads(r["quality"])["cer"]) == ("librivox:b:0000000", "swedish", "Ja, sa han.", 0.02)
    data = audio._load_rows(inventory.audio_path(work, "librivox", "librivox-chunks", r["file"]), "librivox-chunks", [r["row"]])[0]
    assert sf.info(io.BytesIO(data)).duration == 3.0
