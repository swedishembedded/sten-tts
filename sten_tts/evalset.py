# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluation of speech synthesizers for its
# clients. If your team needs expertise in measuring text-to-speech quality,
# you can procure our services by sending an email to info@swedishembedded.com.

"""The evaluation sentences: FLEURS sv_se test transcripts, chosen by the seed.

FLEURS is never trained on; its sentences are also banned from the training
selection by their normalised text.
"""

from __future__ import annotations

import csv
import json

from .config import Recipe, Workdir
from .select import norm_text, rank


def fleurs_sentences(recipe: Recipe, work: Workdir) -> list[dict]:
    path = work.raw("fleurs") / "data" / "sv_se" / "test.tsv"
    if not path.exists():
        raise RuntimeError(f"{path}: not fetched; run the fetch stage")
    seen, rows = set(), []
    with path.open(encoding="utf-8") as fh:
        for rec in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            # id, file name, raw transcription, normalised transcription, ...
            text = rec[2].strip()
            key = norm_text(text)
            if key and key not in seen:
                seen.add(key)
                rows.append({"id": f"fleurs:{rec[0]}", "text": text})
    rows.sort(key=lambda r: rank(recipe.seed, "fleurs" + r["id"]))
    return rows[: recipe["evaluate"]["fleurs_sentences"]]


def all_eval_texts(recipe: Recipe, work: Workdir) -> set[str]:
    """Every FLEURS sv_se sentence (all splits), normalised: none of them may be trained on."""
    out = set()
    for split in ("train", "dev", "test"):
        path = work.raw("fleurs") / "data" / "sv_se" / f"{split}.tsv"
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                out |= {norm_text(r[2]) for r in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) > 2}
    if not out:
        raise RuntimeError("no FLEURS sv_se transcripts fetched; run the fetch stage")
    return out


def write(recipe: Recipe, work: Workdir) -> None:
    work.evaluation.mkdir(parents=True, exist_ok=True)
    (work.evaluation / "sentences.json").write_text(json.dumps(fleurs_sentences(recipe, work), ensure_ascii=False, indent=1))
