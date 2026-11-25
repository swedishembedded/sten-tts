# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in training
# text-to-speech models, you can procure our services by sending an email to
# info@swedishembedded.com.

"""`sten-tts <stage> --workdir DIR [--recipe FILE]`: one command per recipe stage.

Stages run in order and each reads only what earlier stages wrote into the
work directory, so any stage can be rerun on its own.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import config

DEFAULT_RECIPE = Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b.yaml"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sten-tts", description=__doc__)
    p.add_argument("stage", choices=["fetch", "inventory", "select", "audio", "shards", "train", "evaluate", "publish"])
    p.add_argument("--workdir", required=True, type=Path, help="where this run's data, shards, checkpoints and reports live")
    p.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    p.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1), help="worker processes for host-side stages")
    p.add_argument("--only", nargs="*", help="fetch: only these sources")
    p.add_argument("--brain-transport", choices=["dbus", "jsonl"], default="dbus", help="how brain-py reaches the brain server")
    p.add_argument("--brain-bus", choices=["session", "system"], default="session", help="the D-Bus bus the brain server is on")
    p.add_argument("--concurrency", type=int, default=8, help="requests in flight to the brain server")
    p.add_argument("--artifact-store", type=Path, help="train: the brain server's --artifact-store directory, where the checkpoint is published")
    p.add_argument("--allow-cpu", action="store_true", help="train: let the fine-tune run on the CPU backend (a smoke run; far too slow for the release)")
    p.add_argument("--upstream-python", type=Path, help="evaluate: an interpreter with torch and qwen-tts, to check the release loads upstream")
    p.add_argument("--hub-dir", type=Path, help="publish: a local mirror of the Hugging Face repository to copy the release into first")
    p.add_argument("--dry-run", action="store_true", help="publish: check everything, upload nothing")
    a = p.parse_args(argv)
    recipe = config.load(a.recipe)
    work = config.Workdir(a.workdir)
    work.root.mkdir(parents=True, exist_ok=True)

    if a.stage == "fetch":
        from . import fetch
        fetch.fetch(recipe, work, a.only, a.jobs)
    elif a.stage == "inventory":
        from . import inventory
        inventory.build(recipe, work, a.jobs)
    elif a.stage == "select":
        from . import evalset, select
        evalset.write(recipe, work)
        print(json.dumps(select.select(recipe, work, evalset.all_eval_texts(recipe, work)), indent=1, ensure_ascii=False))
    elif a.stage == "audio":
        from . import audio
        print(json.dumps(audio.process(recipe, work, a.jobs), indent=1))
    elif a.stage == "shards":
        from . import brainclient, shards
        print(json.dumps(shards.build(recipe, work, brainclient.connect(a.brain_transport, a.brain_bus), a.concurrency), indent=1))
    elif a.stage == "train":
        from . import brainclient, train
        if a.artifact_store is None:
            p.error("train needs --artifact-store (the brain server's artifact store directory)")
        print(json.dumps(train.run(recipe, work, brainclient.connect(a.brain_transport, a.brain_bus), a.artifact_store, a.allow_cpu), indent=1, ensure_ascii=False))
    elif a.stage == "evaluate":
        from . import brainclient, evaluate
        print(json.dumps(evaluate.run(recipe, work, brainclient.connect(a.brain_transport, a.brain_bus), a.concurrency, a.upstream_python), indent=1))
    elif a.stage == "publish":
        from . import publish
        publish.run(recipe, work, a.dry_run, a.hub_dir)


if __name__ == "__main__":
    main()
