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


def test_a_fleet_run_goes_through_splinter_and_becomes_the_release(tmp_path: Path) -> None:
    """`run_fleet` runs splinter's run-served with the derived params and the shards, records its step
    lines and turns its output directory into the release."""
    import sys
    from sten_tts import config
    recipe = config.load(Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b-v2.yaml")
    work = config.Workdir(tmp_path / "work")
    work.shards.mkdir(parents=True)
    (work.shards / "train.json").write_text(json.dumps({"records": 3520, "hours": 1.0, "voices": {}}))
    for f in ("train.bttshrd", "validation.bttshrd"):
        (work.shards / f).write_bytes(b"x")
    fake = tmp_path / "splinter.py"
    fake.write_text(
        "import json, sys, pathlib\n"
        "a = sys.argv[1:]\n"
        "assert a[:2] == ['jobs', 'run-served'] and a[a.index('--action') + 1] == 'speech_sft'\n"
        "p = json.load(open(a[a.index('--params') + 1]))\n"
        "out = pathlib.Path(a[a.index('--out') + 1]); out.mkdir(parents=True)\n"
        "(out / 'config.json').write_text('{}')\n"
        "(out / 'result.json').write_text(json.dumps({'steps': p['steps'], 'checkpoint': 'best'}))\n"
        "print(json.dumps({'type': 'step', 'step': 1, 'steps': p['steps'], 'loss': 3.0}))\n")
    calls = {}

    def card(recipe, work, report):
        calls["report"] = report
        return "card"

    import sten_tts.train as t
    orig = t.model_card
    t.model_card = card
    try:
        report = train.run_fleet(recipe, work, [sys.executable, str(fake)], ranks=11)
    finally:
        t.model_card = orig
    assert report["request"]["steps"] == 6 * 10, "3520 records over 11 ranks of 32 is 10 steps an epoch"
    assert (work.release / "config.json").exists() and not (work.release / "result.json").exists()
    assert (work.release / "README.md").read_text() == "card"
    assert json.loads((work.train / "events.jsonl").read_text().splitlines()[0])["step"] == 1
