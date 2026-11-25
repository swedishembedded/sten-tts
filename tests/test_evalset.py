# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The committed evaluation sets: loaded strictly, banned from training, scored per set, voice and region."""

from __future__ import annotations

from pathlib import Path

import pytest

from sten_tts import config, evalset, evaluate, select

RECIPE = Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b-v2.yaml"


def test_the_committed_sets_load_with_their_references() -> None:
    sets = evalset.committed_sets(config.load(RECIPE))
    assert sorted(sets) == ["code_switching", "numbers"]
    assert len(sets["code_switching"]) == 60 and len(sets["numbers"]) == 40
    for items in sets.values():
        assert len({i["id"] for i in items}) == len(items)
    nu = next(i for i in sets["numbers"] if i["id"] == "nu03")
    assert nu["text"] == "Biljetten kostar 149 kr." and nu["references"] == ["Biljetten kostar 149 kr.", "Biljetten kostar hundrafyrtionio kronor."]
    cs = sets["code_switching"][0]
    assert cs["references"] == [cs["text"]]


def test_a_malformed_set_is_refused(tmp_path: Path) -> None:
    (tmp_path / "bad.tsv").write_text("x1\tEn mening.\nx1\tSamma id igen.\n")
    with pytest.raises(ValueError, match="x1"):
        evalset.read_set(tmp_path / "bad.tsv")


def test_committed_sentences_never_train(tmp_path: Path) -> None:
    rec = config.load(RECIPE)
    work = config.Workdir(tmp_path)
    (work.raw("fleurs") / "data" / "sv_se").mkdir(parents=True)
    (work.raw("fleurs") / "data" / "sv_se" / "test.tsv").write_text("1\ta.wav\tEn testmening.\tx\n")
    banned = evalset.all_eval_texts(rec, work)
    assert select.norm_text("En testmening.") in banned
    assert select.norm_text("Det är 25 grader varmt i dag.") in banned and select.norm_text("Det är tjugofem grader varmt i dag.") in banned
    assert select.norm_text("Vi kör en daily standup klockan nio varje morgon.") in banned


def test_a_clip_is_scored_against_its_closest_reference_and_summarised_per_set_voice_and_region() -> None:
    assert evaluate.best_cer(["Det kostar 149 kr.", "Det kostar hundrafyrtionio kronor."], "det kostar hundrafyrtionio kronor") == 0.0
    rows = [{"system": "sten", "set": "numbers", "voice": "astrid", "cer": 0.1}, {"system": "sten", "set": "numbers", "voice": "erik", "cer": 0.3},
            {"system": "sten", "set": "numbers", "voice": "linnea", "cer": 0.2}, {"system": "base", "set": "numbers", "voice": "astrid", "cer": 0.5},
            {"system": "sten", "set": "fleurs", "voice": "astrid", "cer": 0.05}]
    region = {"astrid": "Mellansverige", "erik": "Mellansverige", "linnea": "Stockholm med omnejd"}
    out = evaluate.per_set(rows, region)
    assert out["numbers"]["cer"] == pytest.approx(0.2) and out["numbers"]["base_cer"] == pytest.approx(0.5)
    assert out["numbers"]["voices"] == pytest.approx({"astrid": 0.1, "erik": 0.3, "linnea": 0.2})
    assert out["numbers"]["regions"] == pytest.approx({"Mellansverige": 0.2, "Stockholm med omnejd": 0.2})
    assert out["fleurs"]["base_cer"] is None
