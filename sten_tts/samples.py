# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements custom speech models for its clients. If your
# team needs a Swedish or Nordic voice in its product, you can procure our
# services by sending an email to info@swedishembedded.com.

"""The release's listening samples: one evaluation clip per voice, unpicked.

Every sample is the clip the evaluation synthesised for a FLEURS sv_se test
sentence (never trained on) with the recipe's sample seed: the model's real
output on unseen text, not the best of several takes. `publish.samples` sets
the order of the voices and which evaluation sentence each reads (by its index
in the evaluation set, 0 unless named). The clips go to `release/samples/` and
the model card plays them.
"""

from __future__ import annotations

import json

import pyarrow.parquet as pq

from .config import Recipe, Workdir


def write(recipe: Recipe, work: Workdir) -> list[dict]:
    spec = recipe["publish"]["samples"]
    voices = [v["name"] for v in recipe["voices"]]
    order = spec["order"]
    if sorted(order) != sorted(voices):
        raise ValueError(f"publish.samples.order must name every voice once: {sorted(set(voices) ^ set(order))}")
    sentences = json.loads((work.evaluation / "sentences.json").read_text())["fleurs"]
    pick = {v: spec.get("sentence", {}).get(v, 0) for v in order}
    for v, i in pick.items():
        if not 0 <= i < len(sentences):
            raise ValueError(f"publish.samples.sentence.{v}: {i} is outside the {len(sentences)} evaluation sentences")
    wanted = {(v, sentences[i]["id"]) for v, i in pick.items()}
    rows = pq.read_table(work.evaluation / "clips.parquet", columns=["system", "set", "voice", "sentence", "seed", "text", "heard", "cer", "wav"],
                         filters=[("system", "=", "sten"), ("set", "=", "fleurs"), ("seed", "=", spec["seed"])]).to_pylist()
    found = {(r["voice"], r["sentence"]): r for r in rows if (r["voice"], r["sentence"]) in wanted}
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"evaluation/clips.parquet has no seed-{spec['seed']} clip for {missing}")
    out = work.release / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.wav"):
        old.unlink()
    region = {v["name"]: v["region"] for v in recipe["voices"]}
    gender = {v["name"]: v["gender"] for v in recipe["voices"]}
    listing = []
    for v in order:
        r = found[(v, sentences[pick[v]]["id"])]
        (out / f"{v}.wav").write_bytes(r["wav"])
        listing.append({"voice": v, "gender": gender[v], "region": region[v], "file": f"samples/{v}.wav", "sentence": r["sentence"],
                        "text": r["text"], "heard": r["heard"], "cer": round(float(r["cer"]), 4), "seed": spec["seed"]})
    (out / "samples.json").write_text(json.dumps(listing, ensure_ascii=False, indent=1))
    return listing
