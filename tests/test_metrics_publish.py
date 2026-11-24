# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Measures and the publish guard."""

from __future__ import annotations

import json

import pytest

from sten_tts import config, metrics, publish


def test_cer_normalises_case_and_punctuation() -> None:
    assert metrics.cer("Hej, världen!", "hej världen") == 0.0
    assert metrics.cer("abcd", "abed") == 0.25
    with pytest.raises(ValueError):
        metrics.cer("  !", "x")


def test_paired_bootstrap_sees_a_consistent_improvement() -> None:
    better = [0.02, 0.03, 0.01, 0.02] * 10
    worse = [0.10, 0.12, 0.08, 0.09] * 10
    r = metrics.paired_bootstrap(better, worse, seed=1, rounds=2000)
    assert r["mean_diff"] < 0 and r["ci95"][1] < 0 and r["p_a_below_b"] == 1.0


def test_publish_refuses_failed_gates_and_incomplete_releases(tmp_path) -> None:
    work = config.Workdir(tmp_path)
    with pytest.raises(RuntimeError, match="missing"):
        publish.check_release(work)
    for f in publish.REQUIRED:
        (work.release / f).parent.mkdir(parents=True, exist_ok=True)
        (work.release / f).write_text("x")
    work.evaluation.mkdir(parents=True)
    (work.evaluation / "report.json").write_text(json.dumps({"gates": {"cer_max": {"pass": True}, "voice_similarity_min": {"pass": False}}}))
    with pytest.raises(RuntimeError, match="voice_similarity_min"):
        publish.check_release(work)
