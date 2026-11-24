# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The selection's contract on a synthetic inventory: deterministic, disjoint
held-out sets, no evaluation text in training, budgets and caps honoured."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from sten_tts import config, select

RECIPE = Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b.yaml"
REGIONS = ["Mellansverige", "Stockholm med omnejd", "Göteborg med omnejd", "Västra sydsverige", "Norrland", "Dalarna med omnejd"]


def _recipe(hours: float = 1.0) -> config.Recipe:
    r = config.load(RECIPE)
    raw = copy.deepcopy(r.raw)
    raw["selection"]["hours"] = hours
    return config.Recipe(raw=raw, path=r.path)


def _inventory(root: Path) -> None:
    rows = []
    for si in range(60):
        region = REGIONS[si % len(REGIONS)]
        gender = "female" if (si // len(REGIONS)) % 2 == 0 else "male"
        for k in range(40 + si):  # speakers differ in how much they recorded
            rows.append({"source": "nst", "id": f"nst:{si}:{k}", "file": "f", "row": k, "speaker": f"nst:{si}", "gender": gender, "age": "30",
                         "region": region, "text": f"mening nummer {k % 50} av talare {si % 7}", "duration": 4.0, "quality": None})
    for si in range(20):
        for k in range(30):
            q = json.dumps({"wer_whisper": 0.05 if k % 5 else 0.5, "lang_prob_sv": 0.99, "is_silence": False, "year": 2023})
            rows.append({"source": "rixvox", "id": f"rix:{si}:{k}", "file": "g", "row": k, "speaker": f"rix:{si}", "gender": "female" if si % 2 else "male",
                         "age": None, "region": None, "text": f"riksdagen debatterar fråga {k} och {si}", "duration": 12.0, "quality": q})
    (root / "inventory").mkdir(parents=True)
    t = pa.Table.from_pylist(rows)
    for src in ("nst", "rixvox"):
        pq.write_table(t.filter(pc.equal(t["source"], src)), root / "inventory" / f"{src}.parquet")


@pytest.fixture
def work(tmp_path: Path) -> config.Workdir:
    _inventory(tmp_path)
    return config.Workdir(tmp_path)


def _rows(work: config.Workdir) -> list[dict]:
    return pq.read_table(work.selection / "selection.parquet").to_pylist()


def test_selection_is_deterministic(work: config.Workdir) -> None:
    select.select(_recipe(), work, {"mening nummer 1 av talare 1"})
    first = (work.selection / "selection.parquet").read_bytes()
    select.select(_recipe(), work, {"mening nummer 1 av talare 1"})
    assert (work.selection / "selection.parquet").read_bytes() == first


def test_held_out_sets_are_disjoint_from_training(work: config.Workdir) -> None:
    banned = select.norm_text("mening nummer 3 av talare 2")
    select.select(_recipe(), work, {banned})
    rows = _rows(work)
    train = [r for r in rows if r["split"] in ("train", "validation")]
    unseen_speakers = {r["speaker"] for r in rows if r["split"] == "unseen_test"}
    assert unseen_speakers, "whole speakers are held out"
    assert not unseen_speakers & {r["speaker"] for r in train}
    train_texts = {select.norm_text(r["text"]) for r in train}
    assert banned not in train_texts, "an evaluation text never trains"
    voice_test = [r for r in rows if r["split"] == "voice_test"]
    assert voice_test and not {select.norm_text(r["text"]) for r in voice_test} & train_texts


def test_voices_are_distinct_speakers_matching_their_spec(work: config.Workdir) -> None:
    rec = _recipe()
    raw = copy.deepcopy(rec.raw)
    raw["voices"] = [v for v in raw["voices"] if v["region"] in REGIONS]
    select.select(config.Recipe(raw=raw, path=rec.path), work, set())
    voices = json.loads((work.selection / "voices.json").read_text())
    assert len({v["speaker"] for v in voices.values()}) == len(voices) == len(raw["voices"])
    rows = _rows(work)
    for name, v in voices.items():
        mine = [r for r in rows if r["speaker"] == v["speaker"]]
        assert all(r["voice"] == name and r["gender"] == v["gender"] and r["region"] == v["region"] for r in mine)


def test_budget_shares_and_speaker_caps(work: config.Workdir) -> None:
    rec = _recipe(hours=1.0)
    select.select(rec, work, set())
    rows = [r for r in _rows(work) if r["split"] in ("train", "validation")]
    sel = rec["selection"]
    budget = sel["hours"] * sel["oversample"] * 3600
    rix = sum(r["duration"] for r in rows if r["source"] == "rixvox")
    assert rix <= budget * sel["share"]["rixvox"] + 20 * 60 + 1, "rixvox stays near its share (one speaker of slack)"
    per_speaker: dict[str, float] = {}
    for r in rows:
        if r["voice"] is None:
            per_speaker[r["speaker"]] = per_speaker.get(r["speaker"], 0) + r["duration"]
    for s, secs in per_speaker.items():
        cap = sel["nst" if s.startswith("nst") else "rixvox"]["cap_minutes_per_speaker"] * 60
        assert secs <= cap + 1e-6, f"{s}: {secs} s over the cap"
    assert all(json.loads(r["quality"])["wer_whisper"] <= sel["rixvox"]["max_wer_whisper"] for r in rows if r["source"] == "rixvox")


def test_a_bad_recipe_is_refused_naming_the_key(tmp_path: Path) -> None:
    text = RECIPE.read_text().replace("seed: 20261125", "seed: 20261125\nseeed: 1")
    p = tmp_path / "r.yaml"
    p.write_text(text)
    with pytest.raises(config.RecipeError, match="seeed"):
        config.load(p)
