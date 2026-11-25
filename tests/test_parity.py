# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A compact variant passes parity only if it matches the full model on the very same clips, voice by voice."""

from __future__ import annotations

import pytest

from sten_tts import parity

SPEC = {"cer_max_delta": 0.0025, "similarity_max_drop": 0.01, "identity_max_drop": 0.02, "failure_max_delta": 0.005}
VOICES = ["axel", "signe"]


def clips(cer: float, similarity: float = 0.8, nearest: dict | None = None, failure_every: int = 0) -> list[dict]:
    rows = []
    for v in VOICES:
        for s in range(100):
            for seed in (1, 2):
                i = len(rows)
                rows.append({"system": "sten", "set": "fleurs", "voice": v, "sentence": f"fleurs:{s}", "seed": seed, "cer": cer,
                             "similarity": similarity, "nearest": (nearest or {}).get((v, s), v),
                             "failure": "silent" if failure_every and i % failure_every == 0 else None})
    # Rows that parity never reads: the base's clips and the committed sets.
    rows.append({"system": "base", "set": "fleurs", "voice": "axel", "sentence": "fleurs:0", "seed": 1, "cer": 0.9, "similarity": 0.1, "nearest": "signe", "failure": None})
    rows.append({"system": "sten", "set": "numbers", "voice": "axel", "sentence": "numbers:0", "seed": 1, "cer": 0.9, "similarity": 0.1, "nearest": "signe", "failure": None})
    return rows


def test_identical_clips_are_at_parity() -> None:
    out = parity.compare(clips(0.06), clips(0.06), VOICES, SPEC)
    assert all(g["pass"] for g in out["gates"].values()), out["gates"]
    assert out["cer_delta"] == pytest.approx(0.0)


def test_cer_a_hair_worse_passes_and_a_quarter_point_more_fails() -> None:
    assert parity.compare(clips(0.062), clips(0.06), VOICES, SPEC)["gates"]["cer_delta_max"]["pass"]
    assert not parity.compare(clips(0.063), clips(0.06), VOICES, SPEC)["gates"]["cer_delta_max"]["pass"]


def test_one_voice_losing_its_timbre_fails_even_if_the_mean_holds() -> None:
    student = clips(0.06)
    for r in student:
        if r["voice"] == "signe":
            r["similarity"] = 0.78
        elif r["voice"] == "axel":
            r["similarity"] = 0.83
    out = parity.compare(student, clips(0.06), VOICES, SPEC)
    assert not out["gates"]["similarity_drop_max"]["pass"]
    assert out["voices"]["signe"]["similarity_drop"] == pytest.approx(0.02)


def test_identity_and_failures_are_judged_against_the_teacher() -> None:
    confused = {("axel", s): "signe" for s in range(5)}  # 10 of 200 axel clips: accuracy 0.95
    assert not parity.compare(clips(0.06, nearest=confused), clips(0.06), VOICES, SPEC)["gates"]["identity_drop_max"]["pass"]
    assert not parity.compare(clips(0.06, failure_every=50), clips(0.06), VOICES, SPEC)["gates"]["failure_delta_max"]["pass"]


def test_clips_that_do_not_pair_are_an_error_not_a_pass() -> None:
    teacher = clips(0.06)
    with pytest.raises(ValueError, match="pair"):
        parity.compare(clips(0.06)[:-5], teacher, VOICES, SPEC)
