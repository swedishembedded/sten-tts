# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements compression of speech models to the hardware
# a product has, without audible loss, for its clients. If your team needs
# expertise in shrinking text-to-speech models, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Is a compact variant as good as the full model it was carved from?

Both read the same FLEURS test sentences with the same voices and seeds, and
both are judged by the same recogniser and speaker encoder, so the comparison
is clip for clip. Parity holds only if the variant is no worse than the full
model by more than the recipe's margins: in mean CER, and, for every voice, in
similarity to the voice's real recordings, in nearest-voice identification, and
in the failure rate. A mean can hide one voice losing its timbre, so the
per-voice checks are separate gates.
"""

from __future__ import annotations

import numpy as np

from . import metrics

KEY = ("voice", "sentence", "seed")


def _clips(rows: list[dict]) -> dict[tuple, dict]:
    return {tuple(r[k] for k in KEY): r for r in rows if r["system"] == "sten" and r["set"] == "fleurs"}


def merge(gates: dict, par: dict, gate: bool) -> dict:
    """The release gates with the parity checks among them when parity is gated (`evaluate.parity.gate`); ungated,
    parity is reported (in the evaluation's `parity` and the card) and never blocks the release."""
    if gate:
        gates.update({f"parity_{k}": v for k, v in par["gates"].items()})
    return gates


def compare(student_rows: list[dict], teacher_rows: list[dict], voices: list[str], spec: dict, seed: int = 0) -> dict:
    """The parity report and its gates. `spec` holds cer_max_delta, similarity_max_drop, identity_max_drop and failure_max_delta."""
    student, teacher = _clips(student_rows), _clips(teacher_rows)
    if set(student) != set(teacher):
        raise ValueError(f"parity: the variant's and the full model's FLEURS clips do not pair "
                         f"({len(set(student) - set(teacher))} only in the variant, {len(set(teacher) - set(student))} only in the full model)")
    keys = sorted(student)

    def mean(rows: dict, field, voice: str | None = None) -> float:
        return float(np.mean([field(rows[k]) for k in keys if voice is None or k[0] == voice]))

    cer = lambda r: r["cer"]  # noqa: E731
    failed = lambda r: r["failure"] is not None  # noqa: E731
    per_voice = {}
    for v in voices:
        per_voice[v] = {
            "cer": mean(student, cer, v), "teacher_cer": mean(teacher, cer, v),
            "similarity_drop": mean(teacher, lambda r: r["similarity"], v) - mean(student, lambda r: r["similarity"], v),
            "identity_drop": mean(teacher, lambda r: r["nearest"] == v, v) - mean(student, lambda r: r["nearest"] == v, v),
        }
    cer_delta = mean(student, cer) - mean(teacher, cer)
    failure_delta = mean(student, failed) - mean(teacher, failed)
    paired = metrics.paired_bootstrap([student[k]["cer"] for k in keys], [teacher[k]["cer"] for k in keys], seed=seed)
    worst_sim = max(m["similarity_drop"] for m in per_voice.values())
    worst_id = max(m["identity_drop"] for m in per_voice.values())
    gates = {
        "cer_delta_max": {"value": round(cer_delta, 4), "threshold": spec["cer_max_delta"], "pass": cer_delta <= spec["cer_max_delta"] + 1e-12},
        "similarity_drop_max": {"value": round(worst_sim, 4), "threshold": spec["similarity_max_drop"], "pass": worst_sim <= spec["similarity_max_drop"] + 1e-12},
        "identity_drop_max": {"value": round(worst_id, 4), "threshold": spec["identity_max_drop"], "pass": worst_id <= spec["identity_max_drop"] + 1e-12},
        "failure_delta_max": {"value": round(failure_delta, 4), "threshold": spec["failure_max_delta"], "pass": failure_delta <= spec["failure_max_delta"] + 1e-12},
    }
    return {"gates": gates, "cer_delta": cer_delta, "cer_delta_ci95": paired["ci95"], "failure_delta": failure_delta,
            "voices": per_voice, "clips": len(keys)}
