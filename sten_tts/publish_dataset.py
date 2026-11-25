# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for publishing curated, licence-clean
# speech datasets for its clients. If your team needs expertise in releasing
# datasets safely, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Publish the exported dataset: private first, verified, and only then public.

1. The export is checked against its own manifest (every file, its bytes and rows).
2. With `--hub-dir`, the export is copied there first (a durable local copy) and uploaded from there.
3. The repository is created PRIVATE, and checked to be private, before anything is uploaded (an
   upload to a missing repository would create it public). A repository that is already public is
   refused: what it holds was never checked against this export.
4. Everything is uploaded.
5. The repository's files are compared with the manifest, file for file and byte for byte, and every
   parquet file's row count is read from the hub (its footer only).
6. Only then is it made public, and checked to be.

Each step is printed as it happens.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from .config import Recipe, Workdir

EXTRA = ("README.md", "TERMS.md", "voices.json", "manifest.json")


def check_export(export: Path) -> dict:
    manifest = json.loads((export / "manifest.json").read_text())
    for f in manifest["files"]:
        path = export / f["file"]
        if not path.exists() or path.stat().st_size != f["bytes"] or pq.ParquetFile(path).metadata.num_rows != f["rows"]:
            raise RuntimeError(f"{path}: not the file the manifest lists ({f['bytes']} bytes, {f['rows']} rows); run the export stage again")
    for name in EXTRA:
        if not (export / name).exists():
            raise RuntimeError(f"{export / name}: missing; run the export stage again")
    for split, s in manifest["splits"].items():
        if sum(f["rows"] for f in manifest["files"] if f["file"].startswith(f"data/{split}-")) != s["rows"]:
            raise RuntimeError(f"{split}: the files' rows do not add up to the manifest's {s['rows']}")
    return manifest


def _expected(folder: Path) -> dict[str, int]:
    """Every file to publish and its size."""
    return {str(p.relative_to(folder)): p.stat().st_size for p in sorted(folder.rglob("*")) if p.is_file() and not p.relative_to(folder).parts[0].startswith(".")}


def verify_remote(api, repo: str, expected: dict[str, int], manifest: dict, open_remote) -> None:
    """The repository holds exactly the expected files at their sizes, and each parquet file its rows."""
    remote = {e.path: e.size for e in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if getattr(e, "size", None) is not None}
    remote.pop(".gitattributes", None)
    missing = sorted(set(expected) - set(remote))
    extra = sorted(set(remote) - set(expected))
    wrong = sorted(p for p in set(expected) & set(remote) if remote[p] != expected[p])
    if missing or extra or wrong:
        raise RuntimeError(f"{repo}: does not match the export: missing {missing[:5]}, unexpected {extra[:5]}, other sizes {wrong[:5]}")
    for f in manifest["files"]:
        with open_remote(f"datasets/{repo}/{f['file']}") as fh:
            rows = pq.ParquetFile(fh).metadata.num_rows
        if rows != f["rows"]:
            raise RuntimeError(f"{repo}/{f['file']}: {rows} rows on the hub, {f['rows']} in the export")


def run(recipe: Recipe, work: Workdir, hub_dir: Path | None, api=None, open_remote=None) -> dict:
    repo = recipe["dataset"]["repo"]
    export = work.root / "export"
    manifest = check_export(export)
    print(f"publish-dataset: 1/6 the export matches its manifest: {len(manifest['files'])} parquet files, "
          + ", ".join(f"{s} {v['rows']} rows" for s, v in manifest["splits"].items()), flush=True)
    folder = export
    if hub_dir is not None:
        if hub_dir.exists():
            shutil.rmtree(hub_dir)
        shutil.copytree(export, hub_dir, copy_function=shutil.copyfile)  # no ownership or mode carried over
        check_export(hub_dir)
        folder = hub_dir
        print(f"publish-dataset: 2/6 copied to {hub_dir}", flush=True)
    else:
        print("publish-dataset: 2/6 no --hub-dir: uploading from the work directory", flush=True)
    if api is None:
        from huggingface_hub import HfApi, HfFileSystem
        api = HfApi()
        open_remote = HfFileSystem().open
    if api.repo_exists(repo, repo_type="dataset") and not api.dataset_info(repo).private:
        raise RuntimeError(f"{repo} exists and is PUBLIC; refusing to upload into a repository that was never checked against this export")
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.dataset_info(repo).private:
        raise RuntimeError(f"{repo} is not private after creation; refusing to upload")
    print(f"publish-dataset: 3/6 {repo} exists and is private", flush=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(folder), commit_message=f"{recipe.name}: the training collection",
                      ignore_patterns=[".*", ".*/**"], delete_patterns=["data/*", "reports/*"])
    if not api.dataset_info(repo).private:
        raise RuntimeError(f"{repo} turned public during the upload")
    print(f"publish-dataset: 4/6 uploaded {folder}", flush=True)
    expected = _expected(folder)
    verify_remote(api, repo, expected, manifest, open_remote)
    print(f"publish-dataset: 5/6 the hub holds the {len(expected)} files at their sizes and every parquet file its rows", flush=True)
    api.update_repo_settings(repo, private=False, repo_type="dataset")
    if api.dataset_info(repo).private:
        raise RuntimeError(f"{repo} is still private after making it public")
    print(f"publish-dataset: 6/6 {repo} is public: https://huggingface.co/datasets/{repo}", flush=True)
    return {"repo": repo, "files": len(expected), "bytes": sum(expected.values()), "splits": manifest["splits"]}
