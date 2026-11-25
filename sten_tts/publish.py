# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in releasing
# text-to-speech models, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Publish the release directory to its Hugging Face model repository.

Refuses unless the evaluation report says every gate passed and the release
directory is complete. A missing repository is created private and checked to
be private BEFORE anything is uploaded (an upload to a missing repository would
create it public); only after a complete upload is it made public, and only if
`publish.private` is false.
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
    """Upload the release (to `publish.subfolder` when set); with `hub_dir`, first mirror it into the local copy of the repository and upload that."""
    check_release(work)
    spec = recipe["publish"]
    visibility = "private" if spec["private"] else "public"
    if dry_run:
        print(f"publish: {work.release} would go to {spec['repo']} ({visibility})")
        return
    api = HfApi()
    api.create_repo(spec["repo"], repo_type="model", private=True, exist_ok=True)
    if spec["private"] and not api.model_info(spec["repo"]).private:
        raise RuntimeError(f"{spec['repo']} exists and is PUBLIC but publish.private is true; refusing to upload")
    # A compact variant lives in its own subfolder of the same repository, beside the full model.
    sub = spec.get("subfolder")
    folder = work.release
    if hub_dir is not None:
        folder = hub_dir / sub if sub else hub_dir
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copytree(work.release, folder, dirs_exist_ok=True)
    api.upload_folder(repo_id=spec["repo"], repo_type="model", folder_path=str(folder), path_in_repo=sub or None, commit_message=f"{recipe.name}: trained release", ignore_patterns=[".git/*", ".cache/*"])
    if spec["private"] and not api.model_info(spec["repo"]).private:
        raise RuntimeError(f"{spec['repo']} turned public during upload")
    if not spec["private"]:
        api.update_repo_settings(spec["repo"], repo_type="model", private=False)
    print(f"publish: uploaded {work.release} to {spec['repo']} ({visibility})")
