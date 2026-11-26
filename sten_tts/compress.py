# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements compression of speech models to the hardware
# a product has, without audible loss, for its clients. If your team needs
# expertise in shrinking text-to-speech models, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Cut a compact variant out of the full model: brain's `speech_surgery`, as the recipe plans it.

The job runs on the brain server (`brain/train` action `speech_surgery`): it
calibrates the full model on records of the training shard, applies the plan's
edits in its own order, measures each edit's loss and KL divergence from the
full model's predictions on validation records, and publishes the narrower
checkpoint in the upstream layout, installed as the student the `train` stage
starts from. A recipe may cut an earlier rung instead (`compress.from`); the
shards are then read through the full model's tokens and every edit is
measured against the full model. The evaluation sentences go along as the vocabulary trim's probe,
so its report says how many of them would read differently; with
`compress.corpus: inventory` the texts of every candidate utterance of the full
run's inventory go along as its corpus, whose tokens the trim keeps too.

This stage keeps the job's report with the surgery manifest in
`compress/report.json` and the checkpoint in `compress/checkpoint/`, and
refuses a cut that is larger than the rung allows or that changes how any
training text is tokenised.
"""

from __future__ import annotations

import json
import struct
import time
from math import prod
from pathlib import Path

import pyarrow.parquet as pq

from .brainclient import BrainPool
from .config import Recipe, Workdir
from .train import collect, publish_by_digest


def request(recipe: Recipe, work: Workdir) -> tuple[dict, dict[str, bytes]]:
    """The job's params and input blobs, from the recipe and the evaluation sentences."""
    c = recipe["compress"]
    params = {"base": recipe["teacher"]["model"], "plan": json.dumps(c["plan"]), "records": c["records"],
              "held_out_records": c["held_out_records"], "install_as": c["install_as"]}
    if "from" in c:
        # A cut of an earlier rung: the shards are still the full model's tokens, and every edit is measured
        # against the full model, not against the rung it starts from.
        params.update({"base": c["from"]["model"], "shard_tokenizer": recipe["teacher"]["model"], "reference": recipe["teacher"]["model"]})
    sets = json.loads((work.evaluation / "sentences.json").read_text())
    probe = [s["text"].replace("\n", " ") for items in sets.values() for s in items]
    blobs = {"probe": _lines(probe)}
    if c.get("corpus") == "inventory":
        # Every candidate utterance of every source, not only the selected ones: names and words a user's text
        # holds beyond the training texts keep their tokens. The inventory never holds an evaluation sentence.
        texts = [t for f in sorted(work.inventory.glob("*.parquet")) for t in pq.read_table(f, columns=["text"])["text"].to_pylist() if t]
        if not texts:
            raise RuntimeError(f"{work.inventory}: compress.corpus is inventory, and there is no inventory text")
        blobs["corpus"] = _lines(texts)
    return params, blobs


def _lines(texts: list[str]) -> bytes:
    return ("\n".join(t.replace("\n", " ") for t in texts) + "\n").encode()


def parameters(safetensors: Path) -> int:
    """The parameters a safetensors file holds, read from its header."""
    with safetensors.open("rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
    return sum(prod(t["shape"]) for name, t in header.items() if name != "__metadata__")


def check(report: dict, checkpoint: Path, max_parameters: int) -> int:
    """The cut's parameter count, after refusing one over the rung or one that reads a training text differently."""
    changed = {k: v for k, v in report.get("vocabulary", {}).items() if k.endswith("_changed") and not k.startswith("probe") and v}
    if changed:
        raise RuntimeError(f"the trimmed vocabulary tokenises training texts differently: {changed}")
    n = parameters(checkpoint / "model.safetensors")
    if n > max_parameters:
        raise RuntimeError(f"the cut holds {n:,} parameters, more than the rung's {max_parameters:,}")
    return n


def run(recipe: Recipe, work: Workdir, pool: BrainPool, artifacts: Path, allow_cpu: bool = False) -> dict:
    pool = BrainPool(pool.transport, pool.buses[:1], pool.factory)
    pool.require_model("brain/train", "speech_surgery")
    brain = pool.first()
    params, blobs = request(recipe, work)
    if allow_cpu:
        params["allow_cpu"] = True
    params["dataset_digest"] = publish_by_digest(work.shards / "train.bttshrd", artifacts)
    params["held_out_digest"] = publish_by_digest(work.shards / "validation.bttshrd", artifacts)
    started = time.time()
    out = brain.subscribe("brain/train", "speech_surgery", params, blobs=blobs, timeout=86400,
                          on_event=lambda e: print("compress:", json.dumps(e), flush=True)).outputs
    work.compress.mkdir(parents=True, exist_ok=True)
    report = {"request": params, "result": out, "seconds": round(time.time() - started)}
    (work.compress / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    collect(artifacts, out["checkpoint_digest"], work.compress / "checkpoint")
    report["parameters"] = check(out, work.compress / "checkpoint", recipe["compress"]["max_parameters"])
    (work.compress / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return {"parameters": report["parameters"], "baseline": out.get("baseline"), "edits": out.get("edits"), "vocabulary": out.get("vocabulary")}
