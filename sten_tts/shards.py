# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in speech training
# data, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Encode the kept clips with brain and write the training shards.

Every clip goes to the base checkpoint's `encode` action on the brain server
with its transcript; brain returns its codec codes, its x-vector and the token
ids its own tokenizer makes of the transcript; results are kept in
`shards/cache/` every few thousand clips, so an interrupted run resumes. The
training split is cut at
exactly the recipe's hours: the named voices' clips first, then the rest in
the seed's order. Records are written in a fixed order, whatever order the
server answers in, so the shards are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .brainclient import BrainPool
from .config import Recipe, Workdir
from .select import rank
from .shard import ShardWriter

TRAINED_SPLITS = ("train", "validation")


def _clips(work: Workdir, split: str) -> list[dict]:
    rows = []
    for f in sorted(work.audio.glob(f"{split}-*.parquet")):
        rows.extend(pq.read_table(f, columns=["id", "wav", "seconds"]).to_pylist())
    return rows


def cut(clips: list[dict], meta: dict[str, dict], hours: float, seed: int) -> list[dict]:
    """The clips that make exactly `hours` (never more): voices first, then by the seed's order."""
    order = sorted(clips, key=lambda c: (meta[c["id"]]["voice"] is None, rank(seed, "cut" + c["id"])))
    out, total, budget = [], 0.0, hours * 3600
    for c in order:
        if total + c["seconds"] > budget:
            continue
        out.append(c)
        total += c["seconds"]
    if total < budget * 0.999:
        raise RuntimeError(f"only {total / 3600:.2f} h of clean training audio, the recipe asks for {hours} h; select more (selection.oversample)")
    return sorted(out, key=lambda c: c["id"])


CHUNK = 2000
# Clips per encode request: the server encodes every clip of a request in one device batch.
CLIPS_PER_REQUEST = 32


def _cached(cache: Path, split: str) -> dict[str, dict]:
    out = {}
    for f in sorted(cache.glob(f"{split}-*.parquet")):
        for r in pq.read_table(f).to_pylist():
            out[r["id"]] = r
    return out


def _encode_missing(clips: list[dict], cache: Path, split: str, encode, concurrency: int) -> dict[str, dict]:
    """Encode every clip not cached yet, CLIPS_PER_REQUEST to a request, saving every CHUNK results so an interrupted run resumes."""
    done = _cached(cache, split)
    todo = [c for c in clips if c["id"] not in done]
    print(f"shards {split}: {len(done)} cached, {len(todo)} to encode", flush=True)
    with ThreadPoolExecutor(concurrency) as ex:
        for lo in range(0, len(todo), CHUNK):
            part = todo[lo : lo + CHUNK]
            groups = [part[i : i + CLIPS_PER_REQUEST] for i in range(0, len(part), CLIPS_PER_REQUEST)]
            rows = []
            for c, enc in zip(part, (e for batch in ex.map(encode, groups) for e in batch)):
                rows.append({"id": c["id"], "codes": np.asarray(enc.codes, "<u2").tobytes(), "xvector": np.asarray(enc.xvector, "<f4").tobytes(),
                             "text_ids": np.asarray(enc.text_ids, "<u4").tobytes(), "frames": enc.frames, "codebooks": enc.codebooks,
                             "frame_rate": enc.frame_rate, "producers": json.dumps(enc.producers, sort_keys=True)})
            # Named by their content, so runs sharing a cache never write the same file.
            name = f"{split}-{hashlib.blake2b(''.join(r['id'] for r in rows).encode(), digest_size=8).hexdigest()}"
            tmp = cache / f".{name}.{os.getpid()}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), tmp)
            tmp.rename(cache / f"{name}.parquet")
            done.update({r["id"]: r for r in rows})
            print(f"shards {split}: {len(done)}/{len(clips)} encoded", flush=True)
    return done


def build(recipe: Recipe, work: Workdir, pool: BrainPool, concurrency: int) -> dict:
    model = recipe["base"]["repo"]
    pool.require_model(model, "encode")
    meta = {r["id"]: r for r in pq.read_table(work.selection / "selection.parquet").to_pylist()}
    voices = [v["name"] for v in recipe["voices"]]
    # The new language first, then the base's own languages the corpus also trains (English).
    languages = [recipe["language"]["name"], *recipe["language"].get("base_languages", [])]
    cache = work.shards / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in TRAINED_SPLITS:
        out_path = work.shards / f"{split}.bttshrd"
        if out_path.exists():
            report[split] = json.loads((work.shards / f"{split}.json").read_text())
            print(f"shards: {out_path} exists, kept", flush=True)
            continue
        clips = _clips(work, split)
        if split == "train":
            clips = cut(clips, meta, recipe["selection"]["hours"], recipe.seed)

        def encode(group: list[dict]):
            return pool.get().encode_audio_batch([c["wav"] for c in group], [meta[c["id"]]["text"] for c in group], model=model)

        enc = _encode_missing(clips, cache, split, encode, concurrency)
        producers = {enc[c["id"]]["producers"] for c in clips}
        if len(producers) != 1:
            raise RuntimeError(f"{split}: clips were encoded by different models ({sorted(producers)}); delete {cache} and encode again")
        first = enc[clips[0]["id"]]
        writer = ShardWriter(codebooks=first["codebooks"], codebook_size=2048, xvector_dim=len(first["xvector"]) // 4, frame_rate=first["frame_rate"],
                             languages=languages, voices=voices,
                             producers={**json.loads(first["producers"]), "model": f"{model}@{recipe['base']['revision']}", "recipe": recipe.name})
        frames = 0
        for c in clips:
            e, m = enc[c["id"]], meta[c["id"]]
            writer.push(id={"id": c["id"], "speaker": m["speaker"], "source": m["source"], "text": m["text"]},
                        text_ids=np.frombuffer(e["text_ids"], "<u4"), codes=np.frombuffer(e["codes"], "<u2"), xvector=np.frombuffer(e["xvector"], "<f4"),
                        voice=m["voice"], language=m.get("language") or languages[0], complete=True)
            frames += e["frames"]
        writer.write(out_path)
        report[split] = {"records": len(writer), "frames": frames, "hours": round(sum(c["seconds"] for c in clips) / 3600, 3),
                         "hours_per_language": {lang: round(sum(c["seconds"] for c in clips if (meta[c["id"]].get("language") or languages[0]) == lang) / 3600, 3) for lang in languages},
                         "voices": {v: sum(1 for c in clips if meta[c["id"]]["voice"] == v) for v in voices}}
        (work.shards / f"{split}.json").write_text(json.dumps(report[split], indent=1))
    return report
