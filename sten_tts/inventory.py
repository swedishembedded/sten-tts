# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in building speech
# datasets, you can procure our services by sending an email to
# info@swedishembedded.com.

"""One table of every candidate utterance, without its audio.

`inventory/<source>.parquet` has one row per utterance: a locator back into
the fetched file (`file`, `row`), the speaker and their metadata, the
transcript, the duration and the source's own quality fields. Selection reads
only these tables; audio is read again only for the utterances it picks.
"""

from __future__ import annotations

import json
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .config import Recipe, Workdir

COLUMNS = ["source", "id", "file", "row", "speaker", "gender", "age", "region", "text", "duration", "quality"]


def wav_duration(data: bytes) -> float:
    """The duration of a PCM WAV from its header alone."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, rate, block, n = 12, None, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        if cid == b"fmt ":
            _, _, rate, _, block = struct.unpack("<HHIIH", data[pos + 8 : pos + 22])
        elif cid == b"data":
            n = min(size, len(data) - pos - 8)
            break
        pos += 8 + size + (size & 1)
    if not rate or not block or n is None:
        raise ValueError("WAV without fmt or data chunk")
    return n / block / rate


def _nst_file(args: tuple[str, str, list[str], list[str]]) -> list[dict]:
    path, rel, genders, regions = args
    with open(path, "rb") as fh:
        table = ipc.open_stream(fh).read_all()
    meta = table.drop(["audio", "path"]).to_pylist()
    audio = table.column("audio")
    out = []
    for i, m in enumerate(meta):
        stem = Path(table.column("path")[i].as_py()).stem
        out.append({
            "source": "nst",
            "id": f"nst:{stem}",
            "file": rel,
            "row": i,
            # NST reuses a speaker id across recording sessions of different people
            # (age, gender and region differ); the session is the speaker.
            "speaker": f"nst:{stem.split('_')[0]}",
            "gender": genders[m["gender"]].lower(),
            "age": m["age"],
            "region": regions[m["region_of_youth"]],
            "text": m["text"].strip(),
            "duration": wav_duration(audio[i]["bytes"].as_py()),
            "quality": None,
        })
    return out


def nst(work: Workdir, jobs: int) -> pa.Table:
    raw = work.raw("nst")
    info = json.loads((raw / "train" / "dataset_info.json").read_text())["features"]
    genders, regions = info["gender"]["names"], info["region_of_youth"]["names"]
    files = sorted((raw / "train").glob("data-*.arrow"))
    if not files:
        raise RuntimeError(f"{raw}: no fetched NST files; run the fetch stage")
    rows: list[dict] = []
    with ProcessPoolExecutor(jobs) as pool:
        for part in pool.map(_nst_file, [(str(f), str(f.relative_to(raw)), genders, regions) for f in files]):
            rows.extend(part)
    return pa.Table.from_pylist(rows)


def _rixvox_file(args: tuple[str, str]) -> list[dict]:
    path, rel = args
    cols = ["speaker_id", "gender", "district", "year", "duration", "whisper_transcription", "wer_whisper", "lang_prob_sv", "is_silence", "speech_id", "start"]
    t = pq.read_table(path, columns=cols).to_pylist()
    out = []
    for i, m in enumerate(t):
        quality = json.dumps({"wer_whisper": m["wer_whisper"], "lang_prob_sv": m["lang_prob_sv"], "is_silence": bool(m["is_silence"]), "year": m["year"]})
        out.append({
            "source": "rixvox",
            "id": f"rixvox:{m['speech_id']}:{m['start']}",
            "file": rel,
            "row": i,
            "speaker": f"rixvox:{m['speaker_id']}",
            "gender": {"man": "male", "kvinna": "female", "woman": "female"}.get(str(m["gender"]).lower(), "unspecified"),
            "age": None,
            "region": m["district"] if isinstance(m["district"], str) else None,
            "text": (m["whisper_transcription"] or "").strip(),
            "duration": m["duration"] / 1000.0,
            "quality": quality,
        })
    return out


def rixvox(work: Workdir, jobs: int) -> pa.Table:
    raw = work.raw("rixvox")
    files = sorted((raw / "data").glob("rixvox_*.parquet"))
    if not files:
        raise RuntimeError(f"{raw}: no fetched RixVox files; run the fetch stage")
    rows: list[dict] = []
    with ProcessPoolExecutor(jobs) as pool:
        for part in pool.map(_rixvox_file, [(str(f), str(f.relative_to(raw))) for f in files]):
            rows.extend(part)
    return pa.Table.from_pylist(rows)


def build(recipe: Recipe, work: Workdir, jobs: int) -> None:
    work.inventory.mkdir(parents=True, exist_ok=True)
    for name, fn in (("nst", nst), ("rixvox", rixvox)):
        if name not in recipe["sources"]:
            continue
        table = fn(work, jobs)
        ids = table.column("id").to_pylist()
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"{name}: utterance ids are not unique")
        pq.write_table(table.select(COLUMNS), work.inventory / f"{name}.parquet")
        hours = sum(table.column("duration").to_pylist()) / 3600
        print(f"inventory {name}: {table.num_rows} utterances, {hours:.1f} h, {len(set(table.column('speaker').to_pylist()))} speakers", flush=True)
