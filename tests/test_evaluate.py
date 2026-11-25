# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The clip checks evaluation applies before any model is asked anything."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from sten_tts import evaluate


def _wav(seconds: float, amp: float, rate: int = 24000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    buf = io.BytesIO()
    sf.write(buf, (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32), rate, format="WAV")
    return buf.getvalue()


def test_failures_are_named() -> None:
    text = "Hej och välkommen till Sverige."
    assert evaluate.failure(_wav(2.0, 0.3), text, cap_seconds=30) is None
    assert evaluate.failure(_wav(2.0, 0.0), text, cap_seconds=30) == "silent"
    assert evaluate.failure(_wav(30.0, 0.3), text, cap_seconds=30) == "length_cap"
    assert evaluate.failure(_wav(20.0, 0.3), text, cap_seconds=30) == "too_long"


def test_the_recogniser_input_is_16k_mono_f32() -> None:
    pcm = evaluate.pcm16k(_wav(1.0, 0.3))
    assert len(pcm) == 16000 * 4
