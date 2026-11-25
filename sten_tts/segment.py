# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for turning long-form recordings,
# such as audiobooks, into aligned speech training data for its clients. If
# your team needs expertise in building speech corpora, you can procure our
# services by sending an email to info@swedishembedded.com.

"""Cut long recordings with a known text (LibriVox 5-minute chunks) into checked utterances.

Each chunk is cut at its pauses into segments of `min_seconds`..`max_seconds`
(energy-based and deterministic: a pause is a run of frames far below the
chunk's speech level; within the allowed window a segment ends at the middle
of the longest pause). The recogniser (brain's Qwen3-ASR, told the language)
transcribes every segment; the recognised words of the whole chunk are aligned
in order to the chunk's reference text (a word-level edit-distance alignment
of the normalised strings), which gives every segment its span of the book
text. A segment is kept only when its recognised text matches that span
closely (CER at most `max_cer`); its transcript is the span as the book prints
it, punctuation included, never the recognition.

Rejections are recorded per segment: `no_pause` (no cut point in the window),
`too_short`, `unaligned` (no recognised word aligned), `cer` (recognition and
reference disagree), `edge_words` (reference words between two adjacent
segments that neither recognised: one of them holds speech its transcript
would miss).

Outputs, under the work directory: `segments/segments.parquet` (every
candidate segment, kept or not, with its times, reference, recognition, CER
and rejection reason), `segments/audio-NNNN.parquet` (id, 16-bit WAV of every
kept segment, a few large files) and `segments/report.json`. Recognitions are
saved in `segments/cache/` every few thousand segments, so an interrupted run
resumes.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import io
import itertools
import json
import os
import threading
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

from . import metrics
from .config import Recipe, Workdir

FRAME_SECONDS = 0.02
ASR_RATE = 16000
CHUNK = 2000  # recognitions per cache file
SEGMENTS_PER_FILE = 2000  # kept segments per audio file


@dataclass(frozen=True)
class Cut:
    start: float  # seconds into the chunk
    end: float
    reject: str | None  # None, "no_pause" or "too_short"


def pauses(x: np.ndarray, rate: int, cfg: dict) -> list[tuple[float, float]]:
    """(start, end) seconds of every pause: a run of at least `min_pause_seconds` of frames
    more than `silence_db` below the chunk's speech level (the 95th percentile of frame energy)."""
    hop = max(1, int(rate * FRAME_SECONDS))
    n = len(x) // hop
    if n == 0:
        return []
    db = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    quiet = db < np.percentile(db, 95) - cfg["silence_db"]
    out, i, min_frames = [], 0, int(round(cfg["min_pause_seconds"] / FRAME_SECONDS))
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= min_frames:
                out.append((i * FRAME_SECONDS, j * FRAME_SECONDS))
            i = j
        else:
            i += 1
    return out


def cuts(seconds: float, gaps: list[tuple[float, float]], cfg: dict) -> list[Cut]:
    """Consecutive segments covering the chunk: each ends at the middle of the longest pause
    whose middle lies `min_seconds`..`max_seconds` after the segment's start (the earliest of
    equally long ones). Without such a pause, the stretch up to the next pause is `no_pause`."""
    lo, hi = cfg["min_seconds"], cfg["max_seconds"]
    mids = [((a + b) / 2, b - a) for a, b in gaps]
    out, start = [], 0.0
    while seconds - start > 1e-9:
        window = [(m, d) for m, d in mids if start + lo <= m <= start + hi]
        if window:
            end = max(window, key=lambda md: (md[1], -md[0]))[0]
            out.append(Cut(start, end, None))
        elif seconds - start <= hi:
            out.append(Cut(start, seconds, None if seconds - start >= lo else "too_short"))
            break
        else:
            later = [m for m, _ in mids if m > start + hi]
            end = later[0] if later else seconds
            out.append(Cut(start, end, "no_pause"))
        start = end
    return out


def ref_tokens(text: str) -> list[tuple[str, str]]:
    """The reference as (printed, normalised) words; punctuation-only tokens join the word before."""
    out: list[list[str]] = []
    for tok in text.split():
        n = metrics.normalise(tok)
        if not n:
            if out:
                out[-1][0] += " " + tok
            continue
        for k, part in enumerate(n.split()):  # "LibriVox-inspelning" is two normalised words, one printed token
            out.append([tok if k == 0 else "", part])
    return [(p, n) for p, n in out]


def align(ref: list[str], hyp: list[str]) -> list[int | None]:
    """For every hypothesis word, the reference word it is aligned to (None for an insertion):
    a minimum edit-distance alignment over words, ties preferring match/substitution."""
    n, m = len(ref), len(hyp)
    # back[i][j]: 0 diagonal, 1 deletion of ref[i-1], 2 insertion of hyp[j-1]
    back = [bytearray(m + 1) for _ in range(n + 1)]
    prev = list(range(m + 1))
    for j in range(1, m + 1):
        back[0][j] = 2
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        row, r = back[i], ref[i - 1]
        row[0] = 1
        for j in range(1, m + 1):
            d = prev[j - 1] + (r != hyp[j - 1])
            de = prev[j] + 1
            ins = cur[j - 1] + 1
            if d <= de and d <= ins:
                cur[j] = d
            elif de <= ins:
                cur[j], row[j] = de, 1
            else:
                cur[j], row[j] = ins, 2
        prev = cur
    out: list[int | None] = [None] * m
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == 0 and i > 0 and j > 0:
            out[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif op == 1:
            i -= 1
        else:
            j -= 1
    return out


def assign(text: str, segments: list[dict], max_cer: float) -> list[dict]:
    """Give each recognised segment (`heard`, `start`, `end`; in time order) its reference span,
    CER and rejection reason. Segments already rejected (`reject` set) are passed through."""
    toks = ref_tokens(text)
    hyp, owner = [], []
    for k, s in enumerate(segments):
        if s["reject"] is None:
            words = metrics.normalise(s["heard"]).split()
            hyp += words
            owner += [k] * len(words)
    to_ref = align([n for _, n in toks], hyp)
    spans: dict[int, list[int]] = defaultdict(list)
    for k, r in zip(owner, to_ref):
        if r is not None:
            spans[k].append(r)
    out = []
    for k, s in enumerate(segments):
        s = {**s, "text": None, "cer": None, "ref_lo": None, "ref_hi": None}
        if s["reject"] is None:
            if not spans[k]:
                s["reject"] = "unaligned"
            else:
                lo, hi = min(spans[k]), max(spans[k])
                # A printed token that normalises to several words is taken whole.
                while lo > 0 and not toks[lo][0]:
                    lo -= 1
                while hi + 1 < len(toks) and not toks[hi + 1][0]:
                    hi += 1
                printed = " ".join(p for p, _ in toks[lo : hi + 1] if p).strip()
                s.update(text=printed, ref_lo=lo, ref_hi=hi, cer=metrics.cer(" ".join(n for _, n in toks[lo : hi + 1]), s["heard"]))
                if s["cer"] > max_cer:
                    s["reject"] = "cer"
        out.append(s)
    # Reference words that fall between two segments that touch in time were spoken in one of
    # them, but neither recognised them: both segments' transcripts could be missing a word.
    for a, b in zip(out, out[1:]):
        if a["ref_hi"] is not None and b["ref_lo"] is not None and abs(a["end"] - b["start"]) < 1e-9 and b["ref_lo"] > a["ref_hi"] + 1:
            for s in (a, b):
                if s["reject"] is None:
                    s["reject"] = "edge_words"
    return out


def _decode(data: bytes) -> tuple[np.ndarray, int]:
    x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return x.mean(axis=1), rate


def _wav16(y: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1.0, 1.0), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _seg_id(key: str, start: float) -> str:
    return f"librivox:{key}:{int(round(start * 1000)):07d}"


@functools.lru_cache(maxsize=4)
def _row_groups(path: str) -> list[int]:
    """The first row of every row group of a chunk file (the chunks are written one to a group)."""
    f = pq.ParquetFile(path)
    return list(itertools.accumulate((f.metadata.row_group(g).num_rows for g in range(f.num_row_groups)), initial=0))


def chunk_audio(path: str, i: int) -> tuple[np.ndarray, int]:
    """Chunk `i` of the file as mono float samples and their rate, reading only its row group."""
    starts = _row_groups(path)
    g = bisect.bisect_right(starts, i) - 1
    group = pq.ParquetFile(path).read_row_group(g, columns=["audio"]).column("audio")
    return _decode(group[i - starts[g]]["bytes"].as_py())


def _cut_chunk(args: tuple[str, int, dict]) -> list[dict]:
    path, i, cfg = args
    x, rate = chunk_audio(path, i)
    return [{"row": i, "start": c.start, "end": c.end, "reject": c.reject} for c in cuts(len(x) / rate, pauses(x, rate, cfg), cfg)]


def _finish_chunk(args: tuple[str, int, str, list[dict], float]) -> tuple[list[dict], list[dict]]:
    path, i, text, segs, max_cer = args
    done = assign(text, segs, max_cer)
    x, rate = chunk_audio(path, i)
    wavs = [{"id": s["id"], "wav": _wav16(x[int(s["start"] * rate) : int(s["end"] * rate)], rate)} for s in done if s["reject"] is None]
    return done, wavs


def _cached(cache: Path) -> dict[str, str]:
    out = {}
    for f in sorted(cache.glob("asr-*.parquet")):
        for r in pq.read_table(f).to_pylist():
            out[r["id"]] = r["heard"]
    return out


def _save(cache: Path, rows: list[dict]) -> None:
    name = f"asr-{hashlib.blake2b(''.join(r['id'] for r in rows).encode(), digest_size=8).hexdigest()}"
    tmp = cache / f".{name}.{os.getpid()}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    tmp.rename(cache / f"{name}.parquet")


def recognise(path: str, segs: list[dict], cache: Path, transcribe, concurrency: int) -> dict[str, str]:
    """Every unrejected segment's recognition, from the cache or the recogniser, saving every CHUNK."""
    done = _cached(cache)
    by_row: dict[int, list[dict]] = defaultdict(list)
    for s in segs:
        if s["reject"] is None and s["id"] not in done:
            by_row[s["row"]].append(s)
    total = sum(len(v) for v in by_row.values())
    print(f"segment: {len(done)} recognitions cached, {total} to recognise", flush=True)
    pending: list[dict] = []
    lock, slots = threading.Lock(), threading.Semaphore(concurrency * 4)

    def one(seg: dict, pcm: bytes) -> None:
        try:
            heard = transcribe(pcm)
            with lock:
                pending.append({"id": seg["id"], "heard": heard})
                done[seg["id"]] = heard
                if len(pending) >= CHUNK:
                    _save(cache, pending)
                    pending.clear()
                    print(f"segment: {len(done)} recognised", flush=True)
        finally:
            slots.release()

    futures = []
    try:
        with ThreadPoolExecutor(concurrency) as ex:
            for row in sorted(by_row):
                x, rate = chunk_audio(path, row)
                y = x if rate == ASR_RATE else soxr.resample(x, rate, ASR_RATE)
                for s in by_row[row]:
                    slots.acquire()
                    pcm = np.asarray(y[int(s["start"] * ASR_RATE) : int(s["end"] * ASR_RATE)], dtype="<f4").tobytes()
                    futures.append(ex.submit(one, s, pcm))
            for f in futures:
                f.result()  # a failed request fails the stage
    finally:
        with lock:
            if pending:  # what was recognised before a failure is kept for the rerun
                _save(cache, pending)
                pending.clear()
    return done


def run(recipe: Recipe, work: Workdir, pool, concurrency: int, jobs: int) -> dict:
    cfg = recipe["segment"]
    src = cfg["source"]
    path = work.raw(src) / recipe["sources"][src]["files"]
    if not path.exists():
        raise RuntimeError(f"{path}: not fetched; run the fetch stage")
    judge = recipe["judge"].get("served_as", recipe["judge"]["repo"])
    pool.require_model(judge, "transcribe")
    meta = pq.read_table(path, columns=["__key__", "speaker_id", "gender", "text", "duration_sec"]).to_pylist()
    out_dir = work.segments
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("forkserver")) as ex:
        per_chunk = list(ex.map(_cut_chunk, [(str(path), i, cfg) for i in range(len(meta))]))
    segs = []
    for i, cs in enumerate(per_chunk):
        m = meta[i]
        for c in cs:
            segs.append({**c, "id": _seg_id(m["__key__"], c["start"]), "chunk": m["__key__"], "speaker": f"{src}:{m['speaker_id']}",
                         "gender": m["gender"], "duration": c["end"] - c["start"], "heard": None})

    def transcribe(pcm: bytes) -> str:
        return pool.get().run(judge, "transcribe", {"sample_rate": ASR_RATE, "language": cfg["asr_language"]}, blobs={"audio": pcm},
                              meta={"audio": {"media": "audio", "sample_rate": ASR_RATE}}).text("text")

    heard = recognise(str(path), segs, cache, transcribe, concurrency)
    for s in segs:
        s["heard"] = heard.get(s["id"])

    by_row: dict[int, list[dict]] = defaultdict(list)
    for s in segs:
        by_row[s["row"]].append(s)
    for old in out_dir.glob("audio-*.parquet"):
        old.unlink()
    rows, buf, n_files = [], [], 0
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("forkserver")) as ex:
        tasks = [(str(path), i, meta[i]["text"], by_row[i], cfg["max_cer"]) for i in sorted(by_row)]
        for done, wavs in ex.map(_finish_chunk, tasks):
            rows.extend(done)
            buf.extend(wavs)
            while len(buf) >= SEGMENTS_PER_FILE:
                pq.write_table(pa.Table.from_pylist(buf[:SEGMENTS_PER_FILE]), out_dir / f"audio-{n_files:04d}.parquet")
                buf, n_files = buf[SEGMENTS_PER_FILE:], n_files + 1
    if buf:
        pq.write_table(pa.Table.from_pylist(buf), out_dir / f"audio-{n_files:04d}.parquet")
        n_files += 1
    # Where each kept segment's audio is, for the inventory.
    where = {}
    for k in range(n_files):
        for j, sid in enumerate(pq.read_table(out_dir / f"audio-{k:04d}.parquet", columns=["id"]).column("id").to_pylist()):
            where[sid] = (f"audio-{k:04d}.parquet", j)
    for r in rows:
        r["file"], r["file_row"] = where.get(r["id"], (None, None))
    pq.write_table(pa.Table.from_pylist(rows), out_dir / "segments.parquet")
    report = _report(rows, sum(m["duration_sec"] for m in meta))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def _report(rows: list[dict], source_seconds: float) -> dict:
    reasons: dict[str, dict[str, float]] = defaultdict(lambda: {"segments": 0, "hours": 0.0})
    for r in rows:
        k = reasons[r["reject"] or "kept"]
        k["segments"] += 1
        k["hours"] += r["duration"] / 3600
    kept = [r for r in rows if r["reject"] is None]
    return {
        "source_hours": round(source_seconds / 3600, 3),
        "outcome": {k: {"segments": v["segments"], "hours": round(v["hours"], 3)} for k, v in sorted(reasons.items())},
        "kept_cer_mean": round(float(np.mean([r["cer"] for r in kept])), 4) if kept else None,
        "kept_seconds": {"min": round(min((r["duration"] for r in kept), default=0), 2), "max": round(max((r["duration"] for r in kept), default=0), 2)},
    }
