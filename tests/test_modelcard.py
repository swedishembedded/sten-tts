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
