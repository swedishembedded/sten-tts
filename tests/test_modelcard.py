# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The model card is written from the run's reports: sources, languages, voices and every evaluation set."""

from __future__ import annotations

import json
from pathlib import Path

from sten_tts import config, modelcard

RECIPE = Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b-v2.yaml"


def test_the_card_names_every_source_and_reports_sets_per_voice_and_region(tmp_path: Path) -> None:
    rec = config.load(RECIPE)
    work = config.Workdir(tmp_path)
    for d in (work.selection, work.shards, work.evaluation):
        d.mkdir(parents=True)
    hours = {"nst": 180.0, "rixvox": 150.0, "librivox": 30.0, "fleurs": 8.0, "libritts_r": 42.0, "language:english": 42.0, "language:swedish": 368.0}
    (work.selection / "report.json").write_text(json.dumps({"hours": {"train": hours}, "speakers": {"train": 1500}}))
    voices = {v["name"]: {"speaker": f"nst:{v['name']}", "gender": v["gender"], "region": v["region"], "score": 1.25} for v in rec["voices"]}
    (work.selection / "voices.json").write_text(json.dumps(voices))
    (work.shards / "train.json").write_text(json.dumps({"hours": 420.0, "records": 150000, "voices": {"astrid": 300},
                                                        "hours_per_language": {"swedish": 378.0, "english": 42.0}}))
    names = [v["name"] for v in rec["voices"]]
    regions = sorted({v["region"] for v in rec["voices"]})
    sets = {s: {"cer": 0.05, "base_cer": 0.2, "voices": {n: 0.05 for n in names}, "regions": {r: 0.04 for r in regions}, "clips": 10}
            for s in ("fleurs", "code_switching", "numbers")}
    gates = {"failure_rate_max": {"value": 0.01, "threshold": 0.02, "pass": True}}
    per_voice = {n: {"cer": 0.05, "similarity": 0.9, "identity_accuracy": 1.0} for n in names}
    (work.evaluation / "report.json").write_text(json.dumps({"gates": gates, "voices": per_voice, "base": {"cer": 0.2}, "floor": {"cer": 0.03}, "sets": sets}))
    report = {"request": {"steps": 100}, "seconds": 3600, "result": {"initial_loss": 9.0, "final_loss": 4.0}}
    card = modelcard.model_card(rec, work, report)
    for src in ("NST", "RixVox", "LibriVox", "FLEURS", "LibriTTS-R"):
        assert src in card
    assert "- en" in card and "english" in card
    assert "| code_switching |" in card and "| numbers |" in card
    assert all(f"| {r} |" in card for r in regions), "per-region results"
    assert "1.25" in card, "the voices' liveliness scores"
    assert "LibriTTS-R (Koizumi et al., 2023)" in card, "every source's attribution"
    head = card.split("## Use")[0]
    assert "<!-- swedishembedded-offer:start -->" in head and "linkedin.com/in/martinschroder" in head, "the offer sits above the usage"
    assert "error rate is 0.050, against 0.200 for the base model; 1 of 1 release gates pass" in head, "the offer's proof is this run's evaluation"
    assert "Run: one NVIDIA GH200, 1.0 h wall clock" in card


def test_a_fleet_run_names_its_cards_and_the_released_checkpoint(tmp_path: Path) -> None:
    rec = config.load(RECIPE)
    work = config.Workdir(tmp_path)
    for d in (work.selection, work.shards):
        d.mkdir(parents=True)
    (work.selection / "report.json").write_text(json.dumps({"hours": {"train": {"nst": 1.0}}, "speakers": {"train": 10}}))
    (work.selection / "voices.json").write_text(json.dumps({v["name"]: {"gender": v["gender"], "region": v["region"]} for v in rec["voices"]}))
    (work.shards / "train.json").write_text(json.dumps({"hours": 420.0, "records": 194769, "voices": {}}))
    result = {"initial_loss": 3.7, "final_loss": 2.2, "checkpoint": "best", "best": {"step": 1100, "held_out_loss": 2.3595},
              "steps_completed": 1500, "stopped_early": True}
    card = modelcard.model_card(rec, work, {"request": {"steps": 3324}, "ranks": 11, "seconds": 1770, "result": result})
    assert "Run: 11 NVIDIA GH200 cards, 30 min wall clock" in card
    assert "step-1100 checkpoint, the lowest held-out loss (2.3595) of 1500 steps, stopped early" in card
    assert "194,769 utterances in 30 minutes on 11 GPUs" in card and "evaluation is still pending" in card
