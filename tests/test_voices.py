# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Liveliness: the pitch tracker, the measures, and the choice of the liveliest speaker per slot."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from sten_tts import config, voices

RATE = 16000
RECIPE = Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b-v2.yaml"


def _tone(f0: np.ndarray) -> np.ndarray:
    """A voiced signal (three harmonics) following the F0 contour given per sample."""
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    return (0.3 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)).astype(np.float32)


def test_yin_tracks_a_steady_pitch() -> None:
    f0 = voices.yin(_tone(np.full(RATE, 180.0)), RATE)
    assert abs(np.nanmedian(f0) - 180.0) < 1.0
    assert np.mean(np.isnan(f0)) < 0.05


def test_a_moving_pitch_and_level_measure_livelier_than_a_flat_one() -> None:
    t = np.arange(3 * RATE) / RATE
    flat = voices.measures(_tone(np.full(len(t), 120.0)), RATE, "en mening med sju stavelser")
    contour = 120.0 * 2 ** (np.sin(2 * np.pi * 0.7 * t) * 6 / 12)  # +- 6 semitones
    lively = voices.measures(_tone(contour) * (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 2 * t))).astype(np.float32), RATE, "en mening med sju stavelser")
    assert flat["f0_range"] < 0.5 and 9.0 < lively["f0_range"] < 12.5
    assert lively["f0_sd"] > flat["f0_sd"] + 2 and lively["energy_sd"] > flat["energy_sd"] + 1.5
    assert abs(flat["syllable_rate"] - 8 / 3.0) < 0.2, "vowels of the text over the speech duration"


def test_the_liveliest_candidate_wins_each_slot_and_runners_up_are_kept() -> None:
    rec = config.load(RECIPE)
    raw = copy.deepcopy(rec.raw)
    raw["voices"] = [{"name": "astrid", "gender": "female", "region": "A"}, {"name": "erik", "gender": "male", "region": "A"}]
    rec = config.Recipe(raw=raw, path=rec.path)
    flat = {"f0_range": 3.0, "f0_sd": 1.0, "energy_sd": 5.0, "syllable_rate": 4.0}
    per = {"f1": flat, "f2": {**flat, "f0_range": 8.0, "f0_sd": 2.5}, "f3": {**flat, "energy_sd": 6.0},
           "m1": {**flat, "f0_range": 1.0}, "m2": flat}  # low male pitch range is judged among the men only
    gender = {s: "female" if s[0] == "f" else "male" for s in per}
    scored = voices.score(per, gender, rec["liveliness"]["weights"])
    chosen = voices.choose(rec, scored, {s: (gender[s], "A") for s in per}, {s: 1800.0 for s in per})
    assert chosen["astrid"]["speaker"] == "f2" and chosen["erik"]["speaker"] == "m2"
    assert [r["speaker"] for r in chosen["astrid"]["runners_up"]] == ["f3", "f1"]
    assert chosen["astrid"]["score"] > chosen["astrid"]["runners_up"][0]["score"]
