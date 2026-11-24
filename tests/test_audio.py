# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Trimming keeps the speech and a margin; clipped, quiet and noisy clips are rejected."""

from __future__ import annotations

import numpy as np

from sten_tts import audio

CFG = {"silence_trim_db": -40.0, "keep_margin_seconds": 0.15, "max_clipped_fraction": 0.001, "min_snr_db": 15.0, "min_rms_dbfs": -40.0}
RATE = 16000


def _tone(seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_silence_is_trimmed_to_the_margin() -> None:
    x = np.concatenate([np.zeros(RATE), _tone(1.0, 0.3), np.zeros(RATE)])
    y, m = audio.measure_and_trim(x, RATE, CFG)
    assert m["reject"] is None
    assert abs(m["seconds"] - 1.3) < 0.05, m


def test_bad_clips_are_rejected_with_the_reason() -> None:
    rng = np.random.default_rng(0)
    clipped = np.clip(_tone(1.0, 2.0), -1, 1)
    quiet = np.concatenate([np.zeros(RATE // 2), _tone(1.0, 0.003), np.zeros(RATE // 2)])
    noisy = _tone(2.0, 0.3) + rng.normal(0, 0.2, 2 * RATE).astype(np.float32)
    for x, why in ((clipped, "clipped"), (quiet, "quiet"), (noisy, "noisy")):
        y, m = audio.measure_and_trim(x, RATE, CFG)
        assert y is None and m["reject"] == why, (why, m)
