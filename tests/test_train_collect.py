# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Collecting a published checkpoint: every digest checked, no path escapes the release."""

from __future__ import annotations

import json
from pathlib import Path

import blake3
import pytest

from sten_tts import train


def _publish(store: Path, files: dict[str, bytes]) -> str:
    store.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, data in files.items():
        d = blake3.blake3(data).hexdigest()
        (store / d).write_bytes(data)
        entries.append({"name": name, "digest": d, "bytes": len(data)})
    manifest = json.dumps({"format": "brain-checkpoint-manifest-v1", "files": entries}).encode()
    md = blake3.blake3(manifest).hexdigest()
    (store / md).write_bytes(manifest)
    return md


def test_a_published_checkpoint_is_collected_file_for_file(tmp_path: Path) -> None:
    store = tmp_path / "artifacts"
    md = _publish(store, {"config.json": b"{}", "speech_tokenizer/model.safetensors": b"weights"})
    train.collect(store, md, tmp_path / "release")
    assert (tmp_path / "release" / "speech_tokenizer" / "model.safetensors").read_bytes() == b"weights"


def test_a_tampered_file_or_an_escaping_name_is_refused(tmp_path: Path) -> None:
    store = tmp_path / "artifacts"
    md = _publish(store, {"config.json": b"{}"})
    d = blake3.blake3(b"{}").hexdigest()
    (store / d).write_bytes(b"{\"x\": 1}")
    with pytest.raises(RuntimeError, match="blake3"):
        train.collect(store, md, tmp_path / "release")
    md = _publish(tmp_path / "a2", {"../evil": b"x"})
    with pytest.raises(RuntimeError, match="unsafe"):
        train.collect(tmp_path / "a2", md, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_a_shard_is_published_by_its_digest(tmp_path: Path) -> None:
    src = tmp_path / "train.bttshrd"
    src.write_bytes(b"shard bytes")
    store = tmp_path / "artifacts"
    store.mkdir()
    d = train.publish_by_digest(src, store)
    assert d == blake3.blake3(b"shard bytes").hexdigest()
    assert (store / d).read_bytes() == b"shard bytes"
    assert train.publish_by_digest(src, store) == d and len(list(store.iterdir())) == 1
