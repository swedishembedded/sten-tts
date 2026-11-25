# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in training
# text-to-speech models, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Run the fine-tune as a served job on the brain server, then collect the release.

The job is `brain/train` action `speech_sft`. The shards go to brain's artifact
store by digest (a D-Bus blob is capped below a full shard); brain
checks them and the request first (`validate_only`), then trains, exports the
checkpoint in the upstream layout, publishes its files to its artifact store
and installs it. This stage copies the published files into `release/`,
verifying every digest, and adds the model card.

Rerunning the stage with the same recipe resumes an interrupted job: the
request is identical and carries the same `resume_key`.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import blake3

from .brainclient import BrainPool
from .config import Recipe, Workdir
from .modelcard import model_card


def request(recipe: Recipe, work: Workdir) -> dict:
    """The job's params, all of them derived from the recipe and the shards."""
    t = recipe["train"]
    records = json.loads((work.shards / "train.json").read_text())["records"]
    steps = math.ceil(t["epochs"] * records / (t["batch_utterances"] * t["grad_accum"]))
    lang = recipe["language"]
    return {
        "base": recipe["base"]["repo"],
        "languages": json.dumps({lang["name"]: {"new": True, "init_from": lang["init_from"]}}),
        "voices": json.dumps({v["name"]: {"new": True} for v in recipe["voices"]}),
        "steps": steps,
        "lr": t["lr"],
        "weight_decay": t["weight_decay"],
        "warmup": max(1, round(t["warmup_fraction"] * steps)),
        "schedule": "cosine",
        "batch_utterances": t["batch_utterances"],
        "grad_accum": t["grad_accum"],
        "max_frames": t["max_frames"],
        "residual_weight": t["mtp_weight"],
        "streaming_fraction": t["streaming_fraction"],
        "new_row_lr_multiplier": t["lr_multipliers"]["new_rows"],
        "eval_every": t["eval_every"],
        "seed": recipe.seed,
        "cards": 1,
        "install_as": t["install_as"],
        "resume_key": t["resume_key"],
        "checkpoint_every": t["checkpoint_every"],
    }


def _verify(path: Path, digest: str) -> None:
    got = _blake3(path)
    if got != digest:
        raise RuntimeError(f"{path}: blake3 {got} is not the published {digest}")


def _blake3(path: Path) -> str:
    h = blake3.blake3()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_by_digest(path: Path, artifacts: Path) -> str:
    """Put `path` in the server's artifact store as `<blake3 hex>` (written under a temporary name, then renamed)."""
    digest = _blake3(path)
    dst = artifacts / digest
    if not dst.exists():
        tmp = artifacts / f".{digest}.{os.getpid()}.partial"
        shutil.copyfile(path, tmp)
        os.rename(tmp, dst)
    return digest


def collect(artifacts: Path, checkpoint_digest: str, release: Path) -> list[dict]:
    """Copy a published checkpoint (manifest + files, by digest) into `release`."""
    manifest_path = artifacts / checkpoint_digest
    _verify(manifest_path, checkpoint_digest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "brain-checkpoint-manifest-v1":
        raise RuntimeError(f"{manifest_path}: not a checkpoint manifest")
    tmp = release.with_name(release.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    for f in manifest["files"]:
        name = Path(f["name"])
        if name.is_absolute() or ".." in name.parts:
            raise RuntimeError(f"manifest names an unsafe path {f['name']!r}")
        src = artifacts / f["digest"]
        _verify(src, f["digest"])
        dst = tmp / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    shutil.rmtree(release, ignore_errors=True)
    tmp.rename(release)
    return manifest["files"]


def run(recipe: Recipe, work: Workdir, pool: BrainPool, artifacts: Path, allow_cpu: bool = False) -> dict:
    pool.require_model("brain/train", "speech_sft")
    brain = pool.get()
    params = request(recipe, work)
    if allow_cpu:
        params["allow_cpu"] = True
    # The shards go through the artifact store by digest: a D-Bus blob is capped
    # (256 MiB) below a 100-hour shard, and the server verifies each by its hash.
    params["dataset_digest"] = publish_by_digest(work.shards / "train.bttshrd", artifacts)
    params["held_out_digest"] = publish_by_digest(work.shards / "validation.bttshrd", artifacts)
    blobs: dict = {}
    plan = brain.run("brain/train", "speech_sft", {**params, "validate_only": True}, blobs=blobs, timeout=3600).outputs
    print("train: brain accepts the job:", json.dumps(plan), flush=True)

    work.train.mkdir(parents=True, exist_ok=True)
    events = (work.train / "events.jsonl").open("a")
    started = time.time()

    def metrics(event: dict) -> None:
        # A step frame's metrics (loss, talker_ce, code_predictor_ce, eos_accuracy, grad_norm, lr, frames_per_s, held-out losses).
        rec = {"t": round(time.time() - started, 1), **event}
        events.write(json.dumps(rec) + "\n")
        events.flush()
        shown = {k: event[k] for k in ("step", "steps", "loss", "talker_ce", "code_predictor_ce", "eos_accuracy", "frames_per_s", "eval_loss") if k in event}
        print("train:", json.dumps(shown), flush=True)

    out = brain.subscribe("brain/train", "speech_sft", params, blobs=blobs, on_event=metrics, timeout=30 * 86400).outputs
    events.close()
    files = collect(artifacts, out["checkpoint_digest"], work.release)
    report = {"request": params, "plan": plan, "result": out, "files": files, "seconds": round(time.time() - started)}
    (work.release / "README.md").write_text(model_card(recipe, work, report))
    (work.train / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report
