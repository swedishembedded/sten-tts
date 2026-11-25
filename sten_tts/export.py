# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for publishing curated, licence-clean
# speech datasets for its clients. If your team needs expertise in building and
# releasing speech corpora, you can procure our services by sending an email to
# info@swedishembedded.com.

"""The v2 training collection as a Hugging Face dataset, under `export/`.

Every clip the audio stage kept, in the splits it trained or was held out in
(train as the shards cut it, validation, voice_test, unseen_test), with its
training transcript and metadata, and the cleaned, trimmed audio as FLAC at
the source's own rate inside parquet (`datasets`' `Audio` feature), a few
large shards per split: `export/data/<split>-NNNNN-of-MMMMM.parquet`. Beside
them: the dataset card (`README.md`), its terms (`TERMS.md`), the voices'
choice with their liveliness scores (`voices.json`), the selection, audio and
segment reports (`reports/`) and `manifest.json` (every file, its bytes and
rows), which `publish-dataset` checks the uploaded repository against.

Rows are written in id order; rerunning the stage gives the same files.
"""

from __future__ import annotations

import io
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from .audio import _parquet_rows
from .config import Recipe, Workdir
from .datasetcard import card, terms
from .shards import cut

SPLITS = ("train", "validation", "voice_test", "unseen_test")
COLUMNS = ["id", "audio", "text", "language", "speaker", "gender", "region", "source", "source_license", "source_revision", "split", "voice", "duration"]
_STR = {"dtype": "string", "_type": "Value"}
FEATURES = {"id": _STR, "audio": {"_type": "Audio"}, "text": _STR, "language": _STR, "speaker": _STR, "gender": _STR, "region": _STR,
            "source": _STR, "source_license": _STR, "source_revision": _STR, "split": _STR, "voice": _STR,
            "duration": {"dtype": "float64", "_type": "Value"}}
SCHEMA = pa.schema([("id", pa.string()), ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                    *[(c, pa.float64() if c == "duration" else pa.string()) for c in COLUMNS[2:]]],
                   metadata={"huggingface": json.dumps({"info": {"features": FEATURES}})})


def flac(wav: bytes) -> bytes:
    """A 16-bit WAV as lossless FLAC at the same rate."""
    x, rate = sf.read(io.BytesIO(wav), dtype="int16", always_2d=False)
    buf = io.BytesIO()
    sf.write(buf, x, rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _write_shard(args: tuple[str, list[dict]]) -> dict:
    path, rows = args
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_file[r["_file"]].append(r)
    audio = {}
    for f, items in by_file.items():
        for r, wav in zip(items, _parquet_rows(Path(f), "wav", [i["_row"] for i in items])):
            audio[r["id"]] = flac(wav)
    out = [{**{c: r[c] for c in COLUMNS if c != "audio"}, "audio": {"bytes": audio[r["id"]], "path": f"{r['id'].replace(':', '_')}.flac"}} for r in rows]
    tmp = Path(path).with_suffix(f".{os.getpid()}.partial")
    pq.write_table(pa.Table.from_pylist(out, schema=SCHEMA), tmp, row_group_size=500)
    tmp.rename(path)
    return {"file": Path(path).name, "rows": len(rows), "bytes": Path(path).stat().st_size, "hours": sum(r["duration"] for r in rows) / 3600}


def rows_for_export(recipe: Recipe, work: Workdir) -> dict[str, list[dict]]:
    """Per split, every exported clip's metadata and where its audio is, in id order."""
    meta = {r["id"]: r for r in pq.read_table(work.selection / "selection.parquet").to_pylist()}
    out = {}
    for split in SPLITS:
        clips = []
        for f in sorted(work.audio.glob(f"{split}-*.parquet")):
            for k, (cid, secs) in enumerate(zip(*pq.read_table(f, columns=["id", "seconds"]).to_pydict().values())):
                clips.append({"id": cid, "seconds": secs, "_file": str(f), "_row": k})
        if split == "train":  # what the shards trained on: the same cut
            kept = {c["id"] for c in cut(clips, meta, recipe["selection"]["hours"], recipe.seed)}
            clips = [c for c in clips if c["id"] in kept]
        rows = []
        for c in sorted(clips, key=lambda c: c["id"]):
            m = meta[c["id"]]
            src = recipe["sources"][m["source"]]
            rows.append({"id": c["id"], "text": m["text"], "language": m.get("language") or "swedish", "speaker": m["speaker"], "gender": m["gender"],
                         "region": m["region"], "source": m["source"], "source_license": src["license"], "source_revision": f"{src['repo']}@{src['revision']}",
                         "split": split, "voice": m["voice"], "duration": round(c["seconds"], 3), "_file": c["_file"], "_row": c["_row"]})
        out[split] = rows
    return out


def run(recipe: Recipe, work: Workdir, jobs: int) -> dict:
    spec = recipe["dataset"]
    export = work.root / "export"
    data = export / "data"
    shutil.rmtree(export, ignore_errors=True)
    data.mkdir(parents=True)
    per_split = rows_for_export(recipe, work)
    tasks = []
    for split, rows in per_split.items():
        n = max(1, -(-len(rows) // spec["rows_per_shard"]))
        for k in range(n):
            part = rows[k * spec["rows_per_shard"] : (k + 1) * spec["rows_per_shard"]]
            if part:
                tasks.append((str(data / f"{split}-{k:05d}-of-{n:05d}.parquet"), part))
    with ProcessPoolExecutor(jobs) as ex:
        files = list(ex.map(_write_shard, tasks))
    reports = export / "reports"
    reports.mkdir()
    for name, src in (("selection.json", work.selection / "report.json"), ("audio.json", work.audio / "report.json"), ("segment.json", work.segments / "report.json")):
        if src.exists():
            shutil.copyfile(src, reports / name)
    shutil.copyfile(work.selection / "voices.json", export / "voices.json")
    stats = _stats(per_split)
    manifest = {"repo": spec["repo"], "recipe": recipe.name, "splits": {s: {"rows": len(r), "hours": round(sum(x["duration"] for x in r) / 3600, 3)} for s, r in per_split.items()},
                "files": [{**f, "file": f"data/{f['file']}", "hours": round(f["hours"], 3)} for f in sorted(files, key=lambda f: f["file"])], "stats": stats}
    (export / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (export / "TERMS.md").write_text(terms(recipe))
    (export / "README.md").write_text(card(recipe, manifest, json.loads((work.selection / "voices.json").read_text()),
                                           json.loads((work.segments / "report.json").read_text()) if (work.segments / "report.json").exists() else None))
    return {"splits": manifest["splits"], "files": len(files), "gigabytes": round(sum(f["bytes"] for f in files) / 1e9, 2)}


def _stats(per_split: dict[str, list[dict]]) -> dict:
    """Hours per split by source, language, gender and region; speakers per split and source."""
    out = {}
    for split, rows in per_split.items():
        hours: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        speakers: dict[str, set] = defaultdict(set)
        for r in rows:
            h = r["duration"] / 3600
            hours["source"][r["source"]] += h
            hours["language"][r["language"]] += h
            hours["gender"][r["gender"]] += h
            if r["region"] is not None and r["source"] == "nst":
                hours["region"][r["region"]] += h
            speakers[r["source"]].add(r["speaker"])
        out[split] = {"hours": {k: {kk: round(vv, 2) for kk, vv in sorted(v.items())} for k, v in hours.items()},
                      "speakers": {k: len(v) for k, v in sorted(speakers.items())}}
    return out
