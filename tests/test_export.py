# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The exported dataset: HF-loadable FLAC parquet with every row's licence, and a publish that is
private until the hub's copy is verified against the export."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from sten_tts import config, export, publish_dataset

RECIPE = Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b-v2.yaml"


def _wav(seconds: float, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(np.arange(int(seconds * rate)) / 9)).astype(np.float32), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def work(tmp_path: Path):
    rec = config.load(RECIPE)
    raw = copy.deepcopy(rec.raw)
    raw["selection"]["hours"] = 20.0 / 3600
    raw["dataset"]["rows_per_shard"] = 2
    rec = config.Recipe(raw=raw, path=rec.path)
    w = config.Workdir(tmp_path)
    clips = [("nst:a", "train", "swedish", "astrid", 16000), ("libritts_r:b", "train", "english", None, 24000), ("rixvox:c", "train", "swedish", None, 16000),
             ("nst:d", "voice_test", "swedish", "astrid", 16000), ("fleurs:e", "validation", "swedish", None, 16000)]
    w.selection.mkdir(parents=True)
    sel = [{"id": i, "source": i.split(":")[0], "speaker": i, "gender": "female", "region": "Norrland" if i.startswith("nst") else None, "language": lang,
            "text": f"text of {i}", "split": split, "voice": v} for i, split, lang, v, _ in clips]
    pq.write_table(pa.Table.from_pylist(sel), w.selection / "selection.parquet")
    voices = {v["name"]: {"speaker": "nst:a", "gender": v["gender"], "region": v["region"], "score": 1.5,
                          "measures": {"f0_range": 8.0, "f0_sd": 3.0, "energy_sd": 5.0, "syllable_rate": 4.0}} for v in rec["voices"]}
    (w.selection / "voices.json").write_text(json.dumps(voices))
    (w.selection / "report.json").write_text("{}")
    w.audio.mkdir(parents=True)
    for split in ("train", "voice_test", "validation"):
        rows = [{"id": i, "wav": _wav(10.0 if s == "train" else 3.0, r), "seconds": 10.0 if s == "train" else 3.0} for i, s, _, _, r in clips if s == split]
        pq.write_table(pa.Table.from_pylist(rows), w.audio / f"{split}-0000.parquet")
    return rec, w


def test_the_export_is_loadable_audio_with_each_rows_licence(work) -> None:
    rec, w = work
    out = export.run(rec, w, jobs=1)
    assert out["splits"]["train"]["rows"] == 2, "train is the shards' cut: 20 s of the 30"
    assert out["splits"]["voice_test"]["rows"] == 1 and out["splits"]["unseen_test"]["rows"] == 0
    files = sorted((w.root / "export" / "data").glob("train-*.parquet"))
    assert [f.name for f in files] == ["train-00000-of-00001.parquet"]
    t = pq.read_table(files[0])
    assert json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["audio"] == {"_type": "Audio"}
    rows = t.to_pylist()
    for r in rows:
        x, rate = sf.read(io.BytesIO(r["audio"]["bytes"]))
        assert sf.info(io.BytesIO(r["audio"]["bytes"])).format == "FLAC" and len(x) == 10 * rate
        src = rec["sources"][r["source"]]
        assert r["source_license"] == src["license"] and r["source_revision"] == f"{src['repo']}@{src['revision']}"
    card = (w.root / "export" / "README.md").read_text()
    assert "license_link: TERMS.md" in card and "- split: voice_test" in card and "impersonat" in card
    assert "Contains information from the dataset RixVox-v2" in card and "Koizumi2023-hs" in card and "fleurs2022arxiv" in card
    assert "CC-BY-4.0" in (w.root / "export" / "TERMS.md").read_text()


class _Api:
    def __init__(self, folder_files: dict[str, int] | None = None) -> None:
        self.private, self.exists, self.calls, self.uploaded = None, False, [], {}

    def repo_exists(self, repo, repo_type):
        return self.exists

    def create_repo(self, repo, repo_type, private, exist_ok):
        self.calls.append("create")
        self.exists, self.private = True, private

    def dataset_info(self, repo):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, repo_id, repo_type, folder_path, commit_message, ignore_patterns, delete_patterns):
        assert self.private is True, "never upload into a public repository"
        self.calls.append("upload")
        root = Path(folder_path)
        self.uploaded = {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}

    def list_repo_tree(self, repo, repo_type, recursive):
        return [SimpleNamespace(path=k, size=p.stat().st_size) for k, p in self.uploaded.items()]

    def update_repo_settings(self, repo, private, repo_type):
        self.calls.append("public")
        self.private = private


def test_publish_is_private_until_verified_then_public(work, tmp_path) -> None:
    rec, w = work
    export.run(rec, w, jobs=1)
    api = _Api()
    out = publish_dataset.run(rec, w, tmp_path / "hub", api=api, open_remote=lambda p: open(api.uploaded[p.split("/", 3)[3]], "rb"))
    assert api.calls == ["create", "upload", "public"] and api.private is False
    assert out["splits"]["train"]["rows"] == 2 and (tmp_path / "hub" / "data").is_dir()


def test_a_hub_copy_that_differs_stays_private(work, tmp_path) -> None:
    rec, w = work
    export.run(rec, w, jobs=1)
    api = _Api()
    real = api.list_repo_tree
    api.list_repo_tree = lambda *a, **k: [e for e in real(*a, **k) if e.path != "TERMS.md"]
    with pytest.raises(RuntimeError, match="TERMS.md"):
        publish_dataset.run(rec, w, None, api=api, open_remote=lambda p: open(api.uploaded[p.split("/", 3)[3]], "rb"))
    assert api.private is True and "public" not in api.calls
    api2 = _Api()
    api2.exists, api2.private = True, False
    with pytest.raises(RuntimeError, match="PUBLIC"):
        publish_dataset.run(rec, w, None, api=api2, open_remote=None)
