# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluation of speech synthesizers for its
# clients. If your team needs expertise in measuring text-to-speech quality,
# you can procure our services by sending an email to info@swedishembedded.com.

"""Intelligibility and identity measures, and a paired comparison.

CER is the character edit distance between the normalised text that was asked
for and the normalised transcript the recogniser heard, over the reference
length. Numbers are spoken words in Swedish, so both sides are compared after
the same normalisation; a digit sequence the recogniser writes as digits is
not an error against the same digits in the text.
"""

from __future__ import annotations

import math
import random
import re
import unicodedata


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[^0-9a-zåäöéüæø ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    r, h = normalise(reference), normalise(hypothesis)
    if not r:
        raise ValueError("empty reference")
    return edit_distance(r, h) / len(r)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def paired_bootstrap(a: list[float], b: list[float], seed: int, rounds: int = 10_000) -> dict:
    """Mean of `a - b` over paired items with a 95% bootstrap interval, and the share of resamples where a < b."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples of equal, non-zero length")
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(rounds))
    return {"mean_diff": sum(d) / len(d), "ci95": [means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]], "p_a_below_b": sum(m < 0 for m in means) / rounds, "n": len(d)}
