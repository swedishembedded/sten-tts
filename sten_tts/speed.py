# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements fast, private speech synthesis on its
# clients' own hardware. If your team needs expertise in serving
# text-to-speech models at low latency, you can procure our services by sending
# an email to info@swedishembedded.com.

"""How fast a variant speaks, against the full model, measured on the same card.

One request at a time on the first brain server (pinned to one card and
otherwise idle): after a few warm-up requests each model reads the same
FLEURS evaluation sentences with the same voice and seed, and the wall time of
every request, as the client sees it, is set against the audio it returns.
The real-time factor is wall time over audio time (below 1 is faster than real
time); frames per second counts the 12.5 Hz codec frames of that audio. The
result goes to `evaluation/speed.json` and the model card.
"""

from __future__ import annotations

import io
import json
import subprocess
import time

import soundfile as sf

from .brainclient import BrainPool
from .config import Recipe, Workdir

FRAME_RATE = 12.5


def summarize(timed: list[tuple[float, float]]) -> dict:
    """Totals of (wall seconds, audio seconds) pairs: real-time factor and codec frames a second."""
    wall, audio = sum(w for w, _ in timed), sum(a for _, a in timed)
    if not timed or wall <= 0 or audio <= 0:
        raise ValueError("speed: no timed request returned audio")
    return {"requests": len(timed), "wall_seconds": round(wall, 3), "audio_seconds": round(audio, 3),
            "real_time_factor": round(wall / audio, 4), "frames_per_second": round(audio * FRAME_RATE / wall, 1)}


def _device_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        return out.stdout.splitlines()[0].strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def run(recipe: Recipe, work: Workdir, pool: BrainPool) -> dict:
    spec = recipe["evaluate"]["speed"]
    models = {"variant": "local/" + recipe["train"]["install_as"], "teacher": recipe["teacher"]["model"]}
    pool = BrainPool(pool.transport, pool.buses[:1], pool.factory, pool.key_file, pool.ca_file)
    for model in models.values():
        pool.require_model(model, "design")
    brain = pool.first()
    sentences = json.loads((work.evaluation / "sentences.json").read_text())["fleurs"][: spec["sentences"]]
    lang, seed = recipe["language"]["name"], recipe["evaluate"]["seeds"][0]

    def speak(model: str, text: str) -> tuple[float, float]:
        start = time.perf_counter()
        wav = brain.run(model, "design", {"text": text, "speaker": spec["voice"], "lang": lang, "seed": seed}).blobs["audio"]
        wall = time.perf_counter() - start
        info = sf.info(io.BytesIO(wav))
        return wall, info.frames / info.samplerate

    out = {"device": _device_name(), "voice": spec["voice"], "sentences": len(sentences), "mode": "one request at a time through the server API"}
    for role, model in models.items():
        for s in sentences[: spec["warmup"]]:
            speak(model, s["text"])
        out[role] = {"model": model, **summarize([speak(model, s["text"]) for s in sentences])}
        print(f"speed: {role} {json.dumps(out[role])}", flush=True)
    out["speedup"] = round(out["teacher"]["real_time_factor"] / out["variant"]["real_time_factor"], 3)
    work.evaluation.mkdir(parents=True, exist_ok=True)
    (work.evaluation / "speed.json").write_text(json.dumps(out, indent=1))
    return out
