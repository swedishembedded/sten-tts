# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The full model's card plays every voice in every size on the same sentences: the evaluation's own takes, unpicked."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sten_tts import comparison, config

RECIPES = Path(__file__).resolve().parents[1] / "recipe"


def clips(path: Path, system_tag: str, voices: list[str], sentences: list[str]) -> None:
    rows = [{"system": system, "set": "fleurs" if s.startswith("fleurs") else "numbers", "voice": v, "sentence": s, "seed": seed,
             "text": f"text {s}", "heard": f"heard {s}", "cer": 0.0, "wav": f"{system_tag}-{system}-{v}-{s}-{seed}".encode()}
            for system in ("sten", "base") for v in voices for s in sentences for seed in (1, 2)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_every_voice_in_every_size_reads_the_same_sentences(tmp_path: Path) -> None:
    rec = config.load(RECIPES / "sten-tts-1.7b-v2.yaml")
    rec.raw["publish"]["family"] = ["sten-tts-950m.yaml"]
    rec.raw["publish"]["comparison"] = {"seed": 1, "sentences": ["fleurs:7", "nu01"]}
    variant = config.load(RECIPES / "sten-tts-950m.yaml")
    voices = [v["name"] for v in rec["voices"]]
    work = config.Workdir(tmp_path)
    clips(work.evaluation / "clips.parquet", "full", voices, ["fleurs:7", "fleurs:8", "nu01"])
    clips(work.root / "compact" / variant.name / "evaluation" / "clips.parquet", "small", voices, ["fleurs:7", "nu01"])

    listing = comparison.write(rec, work)
    assert [m["title"] for m in listing["models"]] == [rec.title, variant.title]
    assert [s["id"] for s in listing["sentences"]] == ["fleurs:7", "nu01"]
    voice = rec["publish"]["samples"]["order"][0]
    full = work.release / listing["clips"]["nu01"][voice][0]["file"]
    small = work.release / listing["clips"]["nu01"][voice][1]["file"]
    assert full.read_bytes() == f"full-sten-{voice}-nu01-1".encode(), "the full model's seed-1 take, never the base's"
    assert small.read_bytes() == f"small-sten-{voice}-nu01-1".encode()

    card = comparison.section(rec, work)
    assert card.count("<audio") == len(voices) * 2 * 2
    assert f'resolve/main/{listing["clips"]["nu01"][voice][1]["file"]}' in card
    assert card.index(voice.capitalize()) < card.index(rec["publish"]["samples"]["order"][1].capitalize()), "the samples' voice order"


def test_a_sentence_a_size_did_not_read_is_an_error(tmp_path: Path) -> None:
    rec = config.load(RECIPES / "sten-tts-1.7b-v2.yaml")
    rec.raw["publish"]["family"] = ["sten-tts-950m.yaml"]
    rec.raw["publish"]["comparison"] = {"seed": 1, "sentences": ["fleurs:7", "nu01"]}
    variant = config.load(RECIPES / "sten-tts-950m.yaml")
    voices = [v["name"] for v in rec["voices"]]
    work = config.Workdir(tmp_path)
    clips(work.evaluation / "clips.parquet", "full", voices, ["fleurs:7", "nu01"])
    clips(work.root / "compact" / variant.name / "evaluation" / "clips.parquet", "small", voices, ["fleurs:7"])
    with pytest.raises(RuntimeError, match="nu01"):
        comparison.write(rec, work)


def test_without_a_comparison_the_card_has_no_section(tmp_path: Path) -> None:
    rec = config.load(RECIPES / "sten-tts-1.7b-v2.yaml")
    rec.raw["publish"].pop("comparison", None)
    assert comparison.section(rec, config.Workdir(tmp_path)) == ""
    assert json.dumps(rec["publish"])  # the recipe stays valid without the key
