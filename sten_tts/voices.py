# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for choosing and building engaging
# synthetic voices from recorded speakers for its clients. If your team needs
# expertise in voice design for text-to-speech, you can procure our services
# by sending an email to info@swedishembedded.com.

"""Which NST speaker becomes each named voice: the liveliest one with enough audio.

For every voice slot (gender and region from the recipe) the candidates are
the NST recording sessions of that gender and region with at least
`min_minutes` of eligible audio (the selection's own eligibility rules). Each
candidate is measured on `utterances` of their own recordings, drawn by the
seed, and per utterance:

* `f0_range`: the spread of the voiced F0, 90th over 10th percentile, in semitones;
* `f0_sd`: the standard deviation of the voiced F0 in semitones around its median;
* `energy_sd`: the standard deviation, in dB, of the 10 ms frame level over the voiced frames;
* `syllable_rate`: syllables a second (vowel letters of the transcript, the Swedish syllable
  nuclei) over the speech duration (first to last frame within 40 dB of the loudest).

F0 comes from a YIN pitch tracker (40 ms frames, 10 ms hop, 60-400 Hz,
threshold 0.15); frames more than 6 semitones off their five-frame median
(octave errors) are dropped. A candidate's value of each measure is the median over their
utterances. The measures are standardised (z-scores) over all candidates of
the same gender, so that a slot is not judged against the other gender's
pitch, and the liveliness score is

    score = sum_k weight_k * z_k     (weights from the recipe; F0 range and variability first)

The highest score in the slot wins (a speaker is never two voices; ties by the
seed). `selection/voices.json` records every slot's chosen speaker, their
measures and score, and the runner-up candidates with theirs, so a person can
listen and check.
"""

from __future__ import annotations

import io
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from .audio import _load_rows
from .config import Recipe, Workdir
from .inventory import audio_path
from .metrics import normalise
from .select import Utt, _eligible, rank

MEASURES = ("f0_range", "f0_sd", "energy_sd", "syllable_rate")
FMIN, FMAX, YIN_THRESHOLD = 60.0, 400.0, 0.15
VOWELS = set("aeiouyåäöéü")


def yin(x: np.ndarray, rate: int, frame_seconds: float = 0.04, hop_seconds: float = 0.01) -> np.ndarray:
    """F0 in Hz per frame, NaN where unvoiced (de Cheveigné and Kawahara's YIN, steps 1-4 and parabolic refinement)."""
    w, hop = int(rate * frame_seconds), int(rate * hop_seconds)
    tau_min, tau_max = int(rate / FMAX), min(int(rate / FMIN), w - 1)
    if len(x) < w + tau_max:
        return np.zeros(0)
    n = 1 + (len(x) - w - tau_max) // hop
    idx = np.arange(w + tau_max)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    # Difference function by FFT: d(tau) = sum x_j^2 + sum x_{j+tau}^2 - 2 sum x_j x_{j+tau}, j over the window.
    size = 1 << int(np.ceil(np.log2(2 * (w + tau_max))))
    spec = np.fft.rfft(frames, size)
    ref = np.fft.rfft(frames[:, :w], size)
    corr = np.fft.irfft(spec * np.conj(ref), size)[:, : tau_max + 1]
    sq = np.cumsum(np.concatenate([np.zeros((n, 1)), frames**2], axis=1), axis=1)
    energy0 = sq[:, w]
    energy_tau = sq[:, w : w + tau_max + 1] - sq[:, : tau_max + 1]
    d = energy0[:, None] + energy_tau - 2 * corr
    d[:, 0] = 0
    cum = np.cumsum(d[:, 1:], axis=1) / np.arange(1, tau_max + 1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] / np.maximum(cum, 1e-12)
    f0 = np.full(n, np.nan)
    for i in range(n):
        row = cmnd[i]
        below = np.nonzero(row[tau_min : tau_max + 1] < YIN_THRESHOLD)[0]
        if not len(below):
            continue
        t = tau_min + below[0]
        while t + 1 <= tau_max and row[t + 1] < row[t]:
            t += 1  # down to the local minimum
        if 1 <= t < tau_max:
            a, b, c = row[t - 1], row[t], row[t + 1]
            denom = a - 2 * b + c
            t = t + 0.5 * (a - c) / denom if denom > 0 else t
        f0[i] = rate / t
    return f0


def measures(x: np.ndarray, rate: int, text: str) -> dict[str, float] | None:
    """The four measures of one recording, or None when it has too little voiced speech to judge."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    f0 = yin(x, rate)
    hop = int(rate * 0.01)
    n = min(len(f0), len(x) // hop)
    if n < 50:
        return None
    level = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    loud = np.nonzero(level >= level.max() - 40)[0]
    voiced = ~np.isnan(f0[:n]) & (level >= level.max() - 40)
    if voiced.sum() < 30:
        return None
    st = 12 * np.log2(f0[:n][voiced])
    # Octave errors of the tracker are isolated frames: drop those more than 6 semitones off the
    # median of their five-frame neighbourhood, so that they cannot pass for a wide pitch range.
    pad = np.pad(st, 2, mode="edge")
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(pad, 5), axis=1)
    keep = np.abs(st - smooth) <= 6
    st = st[keep]
    speech_seconds = (loud[-1] - loud[0] + 1) * 0.01
    syllables = sum(ch in VOWELS for ch in normalise(text))
    return {"f0_range": float(np.percentile(st, 90) - np.percentile(st, 10)), "f0_sd": float(np.std(st - np.median(st))),
            "energy_sd": float(np.std(level[:n][voiced][keep])), "syllable_rate": syllables / speech_seconds}


def _measure_file(args: tuple[str, str, list[dict]]) -> list[tuple[str, dict | None]]:
    path, fmt, items = args
    out = []
    for item, data in zip(items, _load_rows(Path(path), fmt, [i["row"] for i in items])):
        x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        out.append((item["speaker"], measures(x, rate, item["text"])))
    return out


def score(per_speaker: dict[str, dict[str, float]], gender_of: dict[str, str], weights: dict[str, float]) -> dict[str, dict]:
    """Every speaker's measures, their z-scores among the same gender, and the weighted score."""
    out = {}
    for gender in sorted(set(gender_of.values())):
        group = sorted(s for s in per_speaker if gender_of[s] == gender)
        stats = {}
        for k in MEASURES:
            v = np.array([per_speaker[s][k] for s in group])
            stats[k] = (float(v.mean()), float(v.std()) or 1.0)
        for s in group:
            z = {k: (per_speaker[s][k] - stats[k][0]) / stats[k][1] for k in MEASURES}
            out[s] = {"measures": {k: round(per_speaker[s][k], 3) for k in MEASURES}, "z": {k: round(z[k], 3) for k in MEASURES},
                      "score": round(sum(weights[k] * z[k] for k in MEASURES), 4)}
    return out


def choose(recipe: Recipe, scored: dict[str, dict], meta: dict[str, tuple[str, str]], seconds: dict[str, float], runners_up: int = 3) -> dict:
    """Each slot's highest-scoring candidate not already a voice, with its runners-up."""
    voices, used = {}, set()
    for v in recipe["voices"]:
        cands = sorted((s for s, (g, r) in meta.items() if g == v["gender"] and r == v["region"] and s in scored and s not in used),
                       key=lambda s: (-scored[s]["score"], rank(recipe.seed, s)))
        if not cands:
            raise RuntimeError(f"voice {v['name']}: no NST speaker is {v['gender']} from {v['region']} with {recipe['liveliness']['min_minutes']} min of audio")
        best = cands[0]
        used.add(best)
        voices[v["name"]] = {"speaker": best, "gender": v["gender"], "region": v["region"], "eligible_seconds": round(seconds[best], 1),
                             **scored[best], "candidates": len(cands),
                             "runners_up": [{"speaker": s, "eligible_seconds": round(seconds[s], 1), **scored[s]} for s in cands[1 : 1 + runners_up]]}
    return voices


def run(recipe: Recipe, work: Workdir, jobs: int) -> dict:
    cfg, sel = recipe["liveliness"], recipe["selection"]
    fmt = recipe["sources"]["nst"]["format"]
    by_speaker: dict[str, list[Utt]] = defaultdict(list)
    for r in pq.read_table(work.inventory / "nst.parquet").to_pylist():
        u = Utt(**r)
        if _eligible(u, sel) is None:
            by_speaker[u.speaker].append(u)
    seconds = {s: sum(u.duration for u in us) for s, us in by_speaker.items()}
    slots = {(v["gender"], v["region"]) for v in recipe["voices"]}
    meta = {s: (us[0].gender, us[0].region) for s, us in by_speaker.items()}
    cands = sorted(s for s in by_speaker if meta[s] in slots and seconds[s] >= cfg["min_minutes"] * 60)
    sample: dict[str, list[dict]] = defaultdict(list)
    for s in cands:
        for u in sorted(by_speaker[s], key=lambda u: rank(recipe.seed, "lively" + u.id))[: cfg["utterances"]]:
            sample[u.file].append({"speaker": s, "row": u.row, "text": u.text})
    tasks = [(str(audio_path(work, "nst", fmt, f)), fmt, sorted(items, key=lambda i: i["row"])) for f, items in sorted(sample.items())]
    values: dict[str, list[dict]] = defaultdict(list)
    with ProcessPoolExecutor(jobs) as ex:
        for part in ex.map(_measure_file, tasks):
            for s, m in part:
                if m is not None:
                    values[s].append(m)
    per_speaker = {s: {k: float(np.median([m[k] for m in ms])) for k in MEASURES} for s, ms in values.items() if len(ms) >= cfg["utterances"] // 2}
    scored = score(per_speaker, {s: meta[s][0] for s in per_speaker}, cfg["weights"])
    voices = choose(recipe, scored, meta, seconds)
    work.selection.mkdir(parents=True, exist_ok=True)
    (work.selection / "voices.json").write_text(json.dumps(voices, indent=1, ensure_ascii=False))
    return {name: {"speaker": v["speaker"], "score": v["score"], "minutes": round(v["eligible_seconds"] / 60, 1), "candidates": v["candidates"]} for name, v in voices.items()}
