# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The model card is written from the run's reports: sources, languages, voices and every evaluation set."""

from __future__ import annotations

import json
from pathlib import Path

from sten_tts import config, modelcard

RECIPE = Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b-v2.yaml"


def _run_reports(work: config.Workdir, rec: config.Recipe, records: int = 194769) -> None:
    for d in (work.selection, work.shards):
        d.mkdir(parents=True, exist_ok=True)
    hours = {"nst": 180.0, "rixvox": 150.0, "librivox": 30.0, "fleurs": 8.0, "libritts_r": 42.0}
    (work.selection / "report.json").write_text(json.dumps({"hours": {"train": hours}, "speakers": {"train": 1500}}))
    voices = {v["name"]: {"speaker": f"nst:{v['name']}", "gender": v["gender"], "region": v["region"], "score": 1.25} for v in rec["voices"]}
    (work.selection / "voices.json").write_text(json.dumps(voices))
    (work.shards / "train.json").write_text(json.dumps({"hours": 420.0, "records": records, "voices": {"astrid": 300},
                                                        "hours_per_language": {"swedish": 378.4, "english": 41.6}}))


def test_the_card_serves_a_user_and_takes_every_result_from_the_report(tmp_path: Path) -> None:
    rec = config.load(RECIPE)
    work = config.Workdir(tmp_path)
    _run_reports(work, rec)
    work.evaluation.mkdir(parents=True)
    names = [v["name"] for v in rec["voices"]]
    regions = sorted({v["region"] for v in rec["voices"]})
    clips = {"fleurs": 200, "code_switching": 60, "numbers": 40}
    sets = {s: {"cer": cer, "base_cer": base, "voices": {n: cer for n in names} | {"maja": 0.321}, "regions": {r: 0.043 for r in regions},
                "clips": clips[s] * len(names) * len(rec["evaluate"]["seeds"])}
            for s, cer, base in (("fleurs", 0.051, 0.203), ("code_switching", 0.062, 0.071), ("numbers", 0.154, 0.187))}
    gates = {"failure_rate_max": {"value": 0.0, "threshold": 0.02, "pass": True}}
    per_voice = {n: {"cer": 0.051, "similarity": 0.991, "real_similarity": 0.993, "identity_accuracy": 0.97} for n in names}
    (work.evaluation / "report.json").write_text(json.dumps({
        "gates": gates, "voices": per_voice, "base": {"cer": 0.203}, "floor": {"cer": 0.133}, "sets": sets,
        "paired_cer": {"mean_diff": -0.152, "ci95": [-0.171, -0.134], "n": 2400}, "upstream": {"status": "ok"}}))
    report = {"request": {"steps": 100}, "seconds": 7200, "result": {"initial_loss": 9.0, "final_loss": 4.0}}
    card = modelcard.model_card(rec, work, report)

    meta = card.split("---")[1]
    for line in ("license: apache-2.0", "- sv", "- en", f"base_model: {rec['base']['repo']}", "pipeline_tag: text-to-speech",
                 f"datasets:\n- {rec['dataset']['repo']}"):
        assert line in meta, "valid Hub metadata"

    head, usage = card.split("## Usage")
    assert "<!-- swedishembedded-offer:start -->" in head and "linkedin.com/in/martinschroder" in head, "the offer sits above the usage"
    assert "error rate is 0.051, against 0.203 for the base model" in head, "the offer's proof is this run's evaluation"
    assert f"https://huggingface.co/datasets/{rec['dataset']['repo']}" in head, "the dataset is linked before the usage"
    assert "generate_custom_voice" in usage and "/v1/audio/speech" in usage and '"language": "swedish"' in usage
    assert "private" not in usage and "hf auth" not in card, "the repository is public"
    assert all(f"| `{n}` |" in usage for n in names) and "Central Sweden (Mellansverige)" in usage, "the voice table"

    assert "| FLEURS | 200 | **0.051** | 0.203 |" in card and "| Code-switching | 60 | **0.062** | 0.071 |" in card
    assert "| Numbers | 40 | **0.154** | 0.187 |" in card
    assert "is -0.152 (95% paired-bootstrap interval -0.171 to -0.134, n = 2,400" in card
    assert "| maja | Western southern Sweden | 0.321 | 0.321 | 0.321 |" in card, "per-voice results"
    assert all(f"| {modelcard.REGIONS[r]} | 0.043 | 0.043 | 0.043 |" in card for r in regions), "per-region results"
    assert "Identification accuracy is 0.97" in card and "0.991 for synthesized" in card and "0.993 for real" in card
    assert "recogniser scores 0.133, which is higher than on the synthesized speech" in card, "the floor is context, not a win"
    assert "Failure rate 0 " in card and "upstream `qwen-tts` package, unconverted" in card
    assert f"[{rec['judge']['repo']}]" in card and "told that the language is Swedish" in card

    for src in ("NST", "RixVox", "LibriVox", "FLEURS", "LibriTTS-R"):
        assert src in card
    assert "420 hours of speech (378 h Swedish, 42 h English), 194,769 utterances from 1,500 speakers" in card
    assert rec["base"]["revision"] in card
    assert "Trained with brain on one NVIDIA GH200 GPU in 2.0 hours." in card
    assert "derived from LibriTTS and LibriSpeech, CC-BY-4.0" in card and "ODC Attribution License" in card, "every source's attribution"
    for internal in ("gate", "liveliness", "Pending", "pending", "TODO", "!\n"):
        assert internal not in card, f"{internal!r} is not for the card's reader"


def test_a_fleet_run_names_its_cards_and_the_released_checkpoint(tmp_path: Path) -> None:
    rec = config.load(RECIPE)
    work = config.Workdir(tmp_path)
    _run_reports(work, rec)
    result = {"initial_loss": 3.7, "final_loss": 2.2, "checkpoint": "best", "best": {"step": 1100, "held_out_loss": 2.3595},
              "steps_completed": 1500, "stopped_early": True}
    card = modelcard.model_card(rec, work, {"request": {"steps": 3324}, "ranks": 11, "seconds": 1770, "result": result})
    assert "Trained with brain on 11 NVIDIA GH200 GPUs in 30 minutes." in card
    assert "Training stopped early; the release is the checkpoint with the lowest loss on held-out speech, step 1100 of 1500." in card
    assert "(194,769 utterances) in 30 minutes on 11 GPUs." in card
    assert "## Evaluation" not in card and "error rate" not in card, "an unevaluated release claims no results"
