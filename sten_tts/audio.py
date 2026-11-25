# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in cleaning speech
# corpora, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Read the selected utterances' audio, measure it, trim it, keep the clean ones.

Each clip is decoded from its source file, mixed to mono, measured (clipping,
level, an energy-percentile SNR estimate), and trimmed to its speech with a
short margin. Clips that fail a recipe threshold are recorded with the reason
in `audio/rejected.parquet`; the rest go to `audio/<split>-NNNN.parquet`
(id, 16-bit PCM WAV at the source rate, measurements), a few large files.
Resampling to the model's rate is the model server's job.
"""

from __future__ import annotations

import bisect
import io
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import soundfile as sf

from .config import Recipe, Workdir
from .inventory import audio_path, fleurs_members

FRAME_SECONDS = 0.02


def measure_and_trim(x: np.ndarray, rate: int, cfg: dict) -> tuple[np.ndarray | None, dict]:
    """The trimmed clip (None when rejected) and its measurements, with `reject` set to why."""
    m: dict = {"rate": rate}
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    m["clipped_fraction"] = float(np.mean(np.abs(x) >= 0.999))
    hop = max(1, int(rate * FRAME_SECONDS))
    n = len(x) // hop
    if n < 10:
        return None, {**m, "reject": "too_short"}
    frames = x[: n * hop].reshape(n, hop)
    energy = np.mean(frames**2, axis=1) + 1e-12
    db = 10 * np.log10(energy)
    m["rms_dbfs"] = float(10 * np.log10(np.mean(x**2) + 1e-12))
    m["snr_db"] = float(np.percentile(db, 90) - np.percentile(db, 10))
    voiced = np.nonzero(db >= db.max() + cfg["silence_trim_db"])[0]
    margin = int(cfg["keep_margin_seconds"] / FRAME_SECONDS)
    lo, hi = max(0, voiced[0] - margin), min(n, voiced[-1] + 1 + margin)
    y = x[lo * hop : hi * hop]
    m["trimmed_seconds"] = float((len(x) - len(y)) / rate)
    m["seconds"] = float(len(y) / rate)
    if m["clipped_fraction"] > cfg["max_clipped_fraction"]:
        m["reject"] = "clipped"
    elif m["rms_dbfs"] < cfg["min_rms_dbfs"]:
        m["reject"] = "quiet"
    elif m["snr_db"] < cfg["min_snr_db"]:
        m["reject"] = "noisy"
    else:
        m["reject"] = None
    return (y if m["reject"] is None else None), m


def _wav16(y: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1.0, 1.0), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _load_rows(path: Path, fmt: str, rows: list[int]) -> list[bytes]:
    if fmt == "arrow-stream":
        with path.open("rb") as fh:
            audio = ipc.open_stream(fh).read_all().column("audio")
        return [audio[i]["bytes"].as_py() for i in rows]
    if fmt == "fleurs":
        wanted = set(rows)
        found = {k: data for k, _, data in fleurs_members(path) if k in wanted}
        return [found[i] for i in rows]
    column = "wav" if fmt == "librivox-chunks" else "audio"
    return [v if isinstance(v, bytes) else v["bytes"] for v in _parquet_rows(path, column, rows)]


def _parquet_rows(path: Path, column: str, rows: list[int]) -> list:
    """The values of `column` at `rows`, reading only the row groups that hold them (a RixVox
    shard is 2.9 GB; a worker never holds more than one row group of it)."""
    f = pq.ParquetFile(path)
    starts = [0]
    for g in range(f.num_row_groups):
        starts.append(starts[-1] + f.metadata.row_group(g).num_rows)
    by_group: dict[int, list[int]] = defaultdict(list)
    for i in rows:
        if not 0 <= i < starts[-1]:
            raise IndexError(f"{path}: row {i} of {starts[-1]}")
        by_group[bisect.bisect_right(starts, i) - 1].append(i)
    found = {}
    for g, wanted in sorted(by_group.items()):
        col = f.read_row_group(g, columns=[column]).column(column)
        for i in wanted:
            found[i] = col[i - starts[g]].as_py()
    return [found[i] for i in rows]


def _process_file(args: tuple) -> tuple[list[dict], list[dict]]:
    path, fmt, items, cfg = args
    kept, rejected = [], []
    for (uid, split), data in zip([(i["id"], i["split"]) for i in items], _load_rows(Path(path), fmt, [i["row"] for i in items])):
        x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        y, m = measure_and_trim(x, rate, cfg)
        if y is None:
            rejected.append({"id": uid, "split": split, "reason": m["reject"], "measures": json.dumps(m)})
        else:
            kept.append({"id": uid, "split": split, "wav": _wav16(y, rate), "seconds": m["seconds"], "measures": json.dumps(m)})
    return kept, rejected


def process(recipe: Recipe, work: Workdir, jobs: int, rows_per_file: int = 4000) -> dict:
    sel = pq.read_table(work.selection / "selection.parquet").to_pylist()
    by_file: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in sel:
        by_file[(r["source"], r["file"])].append(r)
    tasks = []
    for (src, f), items in sorted(by_file.items()):
        fmt = recipe["sources"][src]["format"]
        tasks.append((str(audio_path(work, src, fmt, f)), fmt, sorted(items, key=lambda i: i["row"]), recipe["audio"]))
    work.audio.mkdir(parents=True, exist_ok=True)
    for old in work.audio.glob("*.parquet"):
        old.unlink()
    buffers: dict[str, list[dict]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)
    rejected: list[dict] = []
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"clips": 0, "hours": 0.0})

    def flush(split: str, force: bool = False) -> None:
        while len(buffers[split]) >= rows_per_file or (force and buffers[split]):
            chunk, buffers[split] = buffers[split][:rows_per_file], buffers[split][rows_per_file:]
            pq.write_table(pa.Table.from_pylist(chunk), work.audio / f"{split}-{counters[split]:04d}.parquet")
            counters[split] += 1

    with ProcessPoolExecutor(jobs) as pool:
        for kept, rej in pool.map(_process_file, tasks):
            rejected.extend(rej)
            for k in kept:
                buffers[k["split"]].append(k)
                stats[k["split"]]["clips"] += 1
                stats[k["split"]]["hours"] += k["seconds"] / 3600
            for split in list(buffers):
                flush(split)
    for split in list(buffers):
        flush(split, force=True)
    pq.write_table(pa.Table.from_pylist(rejected, schema=pa.schema([("id", pa.string()), ("split", pa.string()), ("reason", pa.string()), ("measures", pa.string())])), work.audio / "rejected.parquet")
    reasons: dict[str, int] = defaultdict(int)
    for r in rejected:
        reasons[r["reason"]] += 1
    report = {"kept": {k: {"clips": v["clips"], "hours": round(v["hours"], 3)} for k, v in stats.items()}, "rejected": dict(reasons)}
    (work.audio / "report.json").write_text(json.dumps(report, indent=1))
    return report
