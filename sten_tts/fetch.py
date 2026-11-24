# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in building speech
# datasets, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Fetch exactly the source files the recipe names, at their pinned revisions.

Only the files a source declares are downloaded (RixVox has 944 shards of
about 2.9 GB; the recipe takes a named range). Every file is recorded with its
size and blob id in `raw/<source>/fetched.json`, which later stages check.
"""

from __future__ import annotations

import fnmatch
import json

from huggingface_hub import HfApi, hf_hub_download

from .config import Recipe, Workdir


def source_files(spec: dict, listing: list[str]) -> list[str]:
    """The files of a source's listing the recipe selects, sorted."""
    files = spec["files"]
    if spec["format"] == "parquet" and isinstance(files, list):
        lo, hi = files
        wanted = {f"data/rixvox_{i:04d}.parquet" for i in range(lo, hi + 1)}
        chosen = sorted(f for f in listing if f in wanted)
        if len(chosen) != len(wanted):
            missing = sorted(wanted - set(chosen))
            raise RuntimeError(f"{spec['repo']}@{spec['revision']}: missing {missing[:5]}")
        return chosen
    chosen = sorted(f for f in listing if fnmatch.fnmatch(f, files))
    if not chosen:
        raise RuntimeError(f"{spec['repo']}@{spec['revision']}: nothing matches {files!r}")
    return chosen


def fetch(recipe: Recipe, work: Workdir, only: list[str] | None = None) -> None:
    api = HfApi()
    for name, spec in recipe["sources"].items():
        if only and name not in only:
            continue
        info = api.dataset_info(spec["repo"], revision=spec["revision"], files_metadata=True)
        meta = {s.rfilename: s for s in info.siblings}
        files = source_files(spec, list(meta))
        if spec["format"] == "arrow-stream":
            files += [f for f in ("train/dataset_info.json",) if f in meta]
        dest = work.raw(name)
        dest.mkdir(parents=True, exist_ok=True)
        record = []
        for f in files:
            path = hf_hub_download(spec["repo"], f, repo_type="dataset", revision=spec["revision"], local_dir=dest)
            size = (dest / f).stat().st_size
            if meta[f].size is not None and size != meta[f].size:
                raise RuntimeError(f"{name}: {f}: {size} bytes on disk, the hub lists {meta[f].size}")
            record.append({"file": f, "bytes": size, "blob_id": meta[f].blob_id})
            print(f"fetch {name}: {f} ({size / 1e9:.2f} GB)", flush=True)
            del path
        (dest / "fetched.json").write_text(json.dumps({"repo": spec["repo"], "revision": spec["revision"], "files": record}, indent=1))
