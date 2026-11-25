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
transcript, its `language` (`swedish` or `english`), the duration and the
source's own quality fields. Selection reads only these tables; audio is read
again only for the utterances it picks.

Sources by format: NST (`arrow-stream`), RixVox (`parquet`), FLEURS (`fleurs`:
train and dev only; the test split is evaluation material), LibriTTS-R
(`libritts-r`, English) and the LibriVox chunks (`librivox-chunks`, whose
utterances are the kept segments of the `segment` stage). FLEURS does not
identify its readers, so each of its recordings is its own speaker; LibriTTS-R
here carries no gender, so its speakers are `unspecified`; a LibriVox
"speaker" is the reader of one audiobook.
"""

from __future__ import annotations

import csv
import json
import struct
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .config import Recipe, Workdir

COLUMNS = ["source", "id", "file", "row", "speaker", "gender", "age", "region", "language", "text", "duration", "quality"]
FLEURS_TRAINED_SPLITS = ("train", "dev")


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
            "language": "swedish",
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
            "language": "swedish",
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


def fleurs_members(tar_path: Path):
    """(index, name, bytes) of every WAV in a FLEURS audio archive, in archive order; `index` is the inventory's `row`."""
    with tarfile.open(tar_path, "r:gz") as tar:
        k = 0
        for member in tar:
            if member.isfile() and member.name.endswith(".wav"):
                yield k, Path(member.name).name, tar.extractfile(member).read()
                k += 1


def _fleurs_split(args: tuple[str, str, str]) -> list[dict]:
    root, split, rel = args
    meta = {}
    with (Path(root) / "data" / "sv_se" / f"{split}.tsv").open(encoding="utf-8") as fh:
        for rec in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            # id (the sentence's), file name, raw transcription, normalised transcription, characters, samples, gender
            meta[rec[1]] = {"sentence": rec[0], "text": rec[2].strip(), "gender": rec[6].lower()}
    out = []
    for k, name, data in fleurs_members(Path(root) / rel):
        m = meta.get(name)
        if m is None:
            continue  # a recording the transcript file leaves out
        stem = Path(name).stem
        out.append({"source": "fleurs", "id": f"fleurs:{stem}", "file": rel, "row": k, "speaker": f"fleurs:{stem}",
                    "gender": m["gender"] if m["gender"] in ("female", "male") else "unspecified", "age": None, "region": None,
                    "language": "swedish", "text": m["text"], "duration": wav_duration(data),
                    "quality": json.dumps({"split": split, "sentence": m["sentence"]})})
    return out


def fleurs(work: Workdir, jobs: int) -> pa.Table:
    raw = work.raw("fleurs")
    tasks = [(str(raw), split, f"data/sv_se/audio/{split}.tar.gz") for split in FLEURS_TRAINED_SPLITS]
    for _, _, rel in tasks:
        if not (raw / rel).exists():
            raise RuntimeError(f"{raw / rel}: not fetched; run the fetch stage")
    rows: list[dict] = []
    with ProcessPoolExecutor(min(jobs, len(tasks))) as pool:
        for part in pool.map(_fleurs_split, tasks):
            rows.extend(part)
    return pa.Table.from_pylist(rows)


def _libritts_file(args: tuple[str, str]) -> list[dict]:
    path, rel = args
    t = pq.read_table(path, columns=["audio", "text_original", "speaker_id", "chapter_id", "id"])
    audio = t.column("audio")
    out = []
    for i, m in enumerate(t.drop(["audio"]).to_pylist()):
        out.append({"source": "libritts_r", "id": f"libritts_r:{m['id']}", "file": rel, "row": i, "speaker": f"libritts_r:{m['speaker_id']}",
                    "gender": "unspecified", "age": None, "region": None, "language": "english", "text": (m["text_original"] or "").strip(),
                    "duration": wav_duration(audio[i]["bytes"].as_py()), "quality": json.dumps({"chapter": m["chapter_id"]})})
    return out


def libritts_r(work: Workdir, jobs: int) -> pa.Table:
    raw = work.raw("libritts_r")
    files = sorted(raw.glob("data/*/*.parquet"))
    if not files:
        raise RuntimeError(f"{raw}: no fetched LibriTTS-R files; run the fetch stage")
    rows: list[dict] = []
    with ProcessPoolExecutor(jobs) as pool:
        for part in pool.map(_libritts_file, [(str(f), str(f.relative_to(raw))) for f in files]):
            rows.extend(part)
    return pa.Table.from_pylist(rows)


def librivox(work: Workdir, jobs: int) -> pa.Table:
    """The kept segments of the `segment` stage; their audio is in `segments/`."""
    path = work.segments / "segments.parquet"
    if not path.exists():
        raise RuntimeError(f"{path}: not segmented yet; run the segment stage")
    rows = []
    for r in pq.read_table(path).to_pylist():
        if r["reject"] is None:
            rows.append({"source": "librivox", "id": r["id"], "file": r["file"], "row": r["file_row"], "speaker": r["speaker"],
                         "gender": r["gender"] if r["gender"] in ("female", "male") else "unspecified", "age": None, "region": None,
                         "language": "swedish", "text": r["text"], "duration": r["duration"], "quality": json.dumps({"cer": round(r["cer"], 4)})})
    return pa.Table.from_pylist(rows)


READERS = {"arrow-stream": nst, "parquet": rixvox, "fleurs": fleurs, "libritts-r": libritts_r, "librivox-chunks": librivox}


def audio_path(work: Workdir, source: str, fmt: str, file: str) -> Path:
    """Where an inventory row's audio file is: the fetched source, or the segments a stage cut."""
    return (work.segments if fmt == "librivox-chunks" else work.raw(source)) / file


def build(recipe: Recipe, work: Workdir, jobs: int, only: list[str] | None = None) -> None:
    work.inventory.mkdir(parents=True, exist_ok=True)
    for name, spec in recipe["sources"].items():
        if only and name not in only:
            continue
        table = READERS[spec["format"]](work, jobs)
        ids = table.column("id").to_pylist()
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"{name}: utterance ids are not unique")
        pq.write_table(table.select(COLUMNS), work.inventory / f"{name}.parquet")
        hours = sum(table.column("duration").to_pylist()) / 3600
        print(f"inventory {name}: {table.num_rows} utterances, {hours:.1f} h, {len(set(table.column('speaker').to_pylist()))} speakers", flush=True)
