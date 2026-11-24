# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in releasing
# text-to-speech models, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Publish the release directory to its Hugging Face model repository.

Refuses unless the evaluation report says every gate passed and the release
directory is complete. A repository is created private and checked to be
private BEFORE anything is uploaded (an upload to a missing repository would
create it public).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from .config import Recipe, Workdir

REQUIRED = ["config.json", "generation_config.json", "model.safetensors", "vocab.json", "merges.txt", "tokenizer_config.json",
            "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors", "README.md"]


def check_release(work: Workdir) -> dict:
    missing = [f for f in REQUIRED if not (work.release / f).exists()]
    if missing:
        raise RuntimeError(f"{work.release}: missing {missing}")
    report_path = work.evaluation / "report.json"
    if not report_path.exists():
        raise RuntimeError(f"{report_path}: no evaluation report; run the evaluate stage")
    report = json.loads(report_path.read_text())
    failed = [g for g, v in report["gates"].items() if not v["pass"]]
    if failed:
        raise RuntimeError(f"release gates failed: {failed}; nothing is published")
    return report


def run(recipe: Recipe, work: Workdir, dry_run: bool, hub_dir: Path | None = None) -> None:
    """Upload the release; with `hub_dir`, first mirror it there (the local copy of the repository) and upload that."""
    check_release(work)
    spec = recipe["publish"]
    if not spec["private"]:
        raise RuntimeError("publish.private must stay true for this release")
    if dry_run:
        print(f"publish: {work.release} would go to {spec['repo']} (private)")
        return
    api = HfApi()
    api.create_repo(spec["repo"], repo_type="model", private=True, exist_ok=True)
    if not api.model_info(spec["repo"]).private:
        raise RuntimeError(f"{spec['repo']} exists and is PUBLIC; refusing to upload")
    folder = work.release
    if hub_dir is not None:
        hub_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(work.release, hub_dir, dirs_exist_ok=True)
        folder = hub_dir
    api.upload_folder(repo_id=spec["repo"], repo_type="model", folder_path=str(folder), commit_message=f"{recipe.name}: trained release", ignore_patterns=[".git/*", ".cache/*"])
    if not api.model_info(spec["repo"]).private:
        raise RuntimeError(f"{spec['repo']} turned public during upload")
    print(f"publish: uploaded {work.release} to {spec['repo']} (private)")
