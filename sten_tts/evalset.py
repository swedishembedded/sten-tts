# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements evaluation of speech synthesizers for its
# clients. If your team needs expertise in measuring text-to-speech quality,
# you can procure our services by sending an email to info@swedishembedded.com.

"""The evaluation sentences: FLEURS sv_se test transcripts chosen by the seed,
and the recipe's committed sets (`evaluate.sets`).

The committed sets test what v1 got wrong: Swedish sentences with English
words, names and brands in them (`code_switching`), and with numbers, dates,
times, units and abbreviations (`numbers`, which also give the spoken form).
Every item has `references`: the texts a recognised clip may be scored
against (the closest one counts), the written text first.

Nothing here is ever trained on: the FLEURS test sentences and every
committed text are banned from the training selection by their normalised text.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

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
                rows.append({"id": f"fleurs:{rec[0]}", "text": text, "references": [text]})
    rows.sort(key=lambda r: rank(recipe.seed, "fleurs" + r["id"]))
    return rows[: recipe["evaluate"]["fleurs_sentences"]]


def read_set(path: Path) -> list[dict]:
    """A committed set: `id<TAB>text[<TAB>spoken]` per line, `#` comments; ids unique."""
    items, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) not in (2, 3) or not all(c.strip() for c in cols):
            raise ValueError(f"{path}:{n}: expected id, text and optionally the spoken form, tab-separated")
        if cols[0] in seen:
            raise ValueError(f"{path}:{n}: id {cols[0]} appears twice")
        seen.add(cols[0])
        items.append({"id": cols[0], "text": cols[1].strip(), "references": [c.strip() for c in cols[1:]]})
    if not items:
        raise ValueError(f"{path}: no sentences")
    return items


def committed_sets(recipe: Recipe) -> dict[str, list[dict]]:
    """The recipe's own evaluation sets, their paths relative to the recipe file."""
    return {name: read_set(recipe.path.parent / rel) for name, rel in recipe["evaluate"].get("sets", {}).items()}


def all_eval_texts(recipe: Recipe, work: Workdir) -> set[str]:
    """Every evaluation text, normalised: the FLEURS sv_se test sentences (FLEURS train and dev are
    training material in v2; their sentences are disjoint from the test split's) and every
    reference of the committed sets. None of them may be trained on."""
    path = work.raw("fleurs") / "data" / "sv_se" / "test.tsv"
    if not path.exists():
        raise RuntimeError(f"{path}: not fetched; run the fetch stage")
    with path.open(encoding="utf-8") as fh:
        out = {norm_text(r[2]) for r in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) > 2}
    for items in committed_sets(recipe).values():
        out |= {norm_text(ref) for i in items for ref in i["references"]}
    return out


def write(recipe: Recipe, work: Workdir) -> None:
    work.evaluation.mkdir(parents=True, exist_ok=True)
    sets = {"fleurs": fleurs_sentences(recipe, work), **committed_sets(recipe)}
    (work.evaluation / "sentences.json").write_text(json.dumps(sets, ensure_ascii=False, indent=1))
