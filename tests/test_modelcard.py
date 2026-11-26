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


VARIANT = RECIPE.parent / "sten-tts-950m.yaml"


def _header(path: Path, numel: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"w": {"dtype": "BF16", "shape": [numel], "data_offsets": [0, 0]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def _variant_reports(rec: config.Recipe, work: config.Workdir, cer_delta: float = 0.001, ci: tuple = (-0.002, 0.004)) -> dict:
    """Every report a variant card reads; parity off by `cer_delta` (CI `ci`). Answers the train report."""
    _run_reports(work, rec)
    names = [v["name"] for v in rec["voices"]]
    _header(work.release / "model.safetensors", 953_600_000)
    _header(work.release / "speech_tokenizer" / "model.safetensors", 170_600_000)
    work.compress.mkdir(parents=True)
    edits = [{"edit": {"trim_vocabulary": {"vocabulary": "text"}}, "params_before": 1_916_676_352, "params_after": 1_670_615_296, "loss": 2.6726, "kl": 0.0},
             {"edit": {"select_channels": {"dims": "talker.layers.*.mlp", "keep": 3072, "score": "fisher"}}, "params_before": 1_670_615_296, "params_after": 1_142_100_000, "loss": 2.91, "kl": 0.341},
             {"edit": {"remove_layers": {"stack": "talker.layers", "count": 6, "by": "kl"}}, "params_before": 1_142_100_000, "params_after": 953_600_000, "loss": 3.2, "kl": 0.62}]
    kept_layers = [i for i in range(28) if i not in (6, 7, 8, 9, 12, 13)]
    changes = [({"edit": "trim_vocabulary", "config": {"talker_config.text_vocab_size": [151936, 31789]}, "targets": ["text"]}, "corpus"),
               ({"edit": "select_channels", "config": {"talker_config.intermediate_size": [6144, 3072]}, "targets": [f"talker.layers.{i}.mlp" for i in range(28)]}, "fisher"),
               ({"edit": "remove_layers", "config": {"talker_config.num_hidden_layers": [28, 22]}, "kept": {"talker.layers": kept_layers}, "targets": ["talker.layers"]}, "ablation_kl")]
    (work.compress / "report.json").write_text(json.dumps({"parameters": 953_600_000, "request": {"records": 128, "held_out_records": 128}, "result": {
        "baseline": {"loss": 2.6726}, "edits": edits, "params_before": 1_916_676_352, "params_after": 953_600_000,
        "vocabulary": {"kept": 31789, "of": 151676}, "manifest": {"manifest": {"edits": [{"change": c, "method": m} for c, m in changes]}}}}))
    work.evaluation.mkdir(parents=True)
    sets = {s: {"cer": 0.065, "base_cer": 0.102, "voices": {n: 0.065 for n in names}, "regions": {v["region"]: 0.06 for v in rec["voices"]}, "clips": c * 24}
            for s, c in (("fleurs", 200), ("code_switching", 60), ("numbers", 40))}
    par_voices = {n: {"cer": 0.065, "teacher_cer": 0.064, "similarity_drop": 0.004, "identity_drop": 0.0} for n in names}
    gates = {"failure_rate_max": {"value": 0.0, "threshold": 0.02, "pass": True},
             "parity_cer_delta_max": {"value": 0.001, "threshold": 0.0025, "pass": True}}
    (work.evaluation / "report.json").write_text(json.dumps({
        "gates": gates, "voices": {n: {"cer": 0.065, "similarity": 0.70, "real_similarity": 0.71, "identity_accuracy": 1.0} for n in names},
        "base": {"cer": 0.102}, "floor": {"cer": 0.147}, "sets": sets, "paired_cer": {"mean_diff": -0.037, "ci95": [-0.049, -0.026], "n": 2400},
        "upstream": {"status": "ok"}, "parity": {"cer_delta": cer_delta, "cer_delta_ci95": list(ci), "failure_delta": 0.0, "voices": par_voices, "clips": 4800,
                                                "gates": {"cer_delta_max": gates["parity_cer_delta_max"]}}}))
    (work.evaluation / "speed.json").write_text(json.dumps({"device": "NVIDIA GH200 120GB", "voice": "astrid", "sentences": 20,
        "variant": {"real_time_factor": 0.21, "frames_per_second": 59.5}, "teacher": {"real_time_factor": 0.30, "frames_per_second": 41.7}, "speedup": 1.43}))
    report = {"request": {"steps": 6000, "lr": 1e-4}, "ranks": 4, "seconds": 7200,
              "result": {"checkpoint": "best", "best": {"step": 2400, "held_out_loss": 0.21}, "steps_completed": 3200, "stopped_early": True}}
    teacher = work.root / rec["evaluate"]["parity"]["teacher_evaluation"]
    teacher.mkdir()
    (teacher / "report.json").write_text(json.dumps({"sets": {"fleurs": {"cer": 0.064}, "code_switching": {"cer": 0.050}, "numbers": {"cer": 0.070}}}))
    return report


def test_a_variant_card_states_its_size_cut_parity_speed_and_how_to_load_it(tmp_path: Path) -> None:
    rec = config.load(VARIANT)
    work = config.Workdir(tmp_path)
    report = _variant_reports(rec, work)
    card = modelcard.model_card(rec, work, report)

    assert "It reads code-switching sentences less clearly than Sten-TTS-1.7B (CER 0.065 against 0.050)" in card, "a weaker set is said"
    assert "numbers sentences less clearly" not in card
    assert card.startswith("---\n") and "base_model: swedishembedded/sten-tts" in card.split("---")[1]
    assert "# Sten-TTS-950M" in card and "953.6 M parameters" in card and "1,916.7 M" in card
    assert "170.6 M" in card and "shared" in card and "counted separately" in card, "the codec"
    assert 'allow_patterns="sten-tts-950m/*"' in card and 'f"{path}/sten-tts-950m"' in card, "loading from the subfolder"
    for frag in ("Trim the text vocabulary to 31,789 of 151,676 tokens", "Keep 3,072 of 6,144 channels in every Talker MLP", "Remove 6 of 28 Talker layers (6, 7, 8, 9, 12 and 13), the ones it misses least alone, by KL divergence", "by Fisher score"):
        assert frag in card, frag
    assert "| astrid |" in card and "0.004" in card, "the per-voice parity table"
    assert "0.21" in card and "59.5" in card and "0.30" in card and "NVIDIA GH200" in card, "measured speed of both"
    assert "step 2400 of 3200" in card and "4 NVIDIA GH200 GPUs" in card
    for internal in ("gate", "Pending", "TODO", "teacher", "student"):
        assert internal not in card, f"{internal!r} is not for the card's reader"


def test_an_ungated_variant_that_misses_parity_says_so_plainly(tmp_path: Path) -> None:
    rec = config.load(VARIANT.parent / "sten-tts-475m.yaml")
    work = config.Workdir(tmp_path)
    report = _variant_reports(rec, work, cer_delta=0.0125, ci=(0.0103, 0.0150))
    r = json.loads((work.evaluation / "report.json").read_text())
    r["parity"]["gates"]["cer_delta_max"] = {"value": 0.0125, "threshold": 0.0025, "pass": False}
    r["gates"].pop("parity_cer_delta_max")  # ungated: not a release gate
    (work.evaluation / "report.json").write_text(json.dumps(r))
    card = modelcard.model_card(rec, work, report)
    head, parity_section = card.split("## Parity with Sten-TTS-1.7B")
    expected = "It is not at parity with Sten-TTS-1.7B: its FLEURS CER is 0.0125 higher (95% paired-bootstrap interval 0.0103 to 0.0150)"
    assert expected in head and expected in parity_section, "said in the description and in the parity section"
    assert "at parity with Sten-TTS-1.7B: no worse" not in card


def test_the_full_model_card_lists_its_compact_variants_from_their_evidence(tmp_path: Path) -> None:
    import yaml
    raw = yaml.safe_load(RECIPE.read_text())
    raw["publish"]["family"] = ["sten-tts-950m.yaml", "sten-tts-475m.yaml"]
    path = RECIPE.parent / ".family-test.yaml"
    path.write_text(yaml.safe_dump(raw))
    try:
        rec = config.load(path)
    finally:
        path.unlink()
    work = config.Workdir(tmp_path)
    _run_reports(work, rec)
    for name, params, cer, delta, ci, ok, rtf in (("sten-tts-950m", 952_628_480, 0.0635, -0.0001, [-0.0031, 0.0030], True, 0.123),
                                                  ("sten-tts-475m", 469_589_504, 0.0762, 0.0125, [0.0103, 0.0150], False, 0.110)):
        d = work.root / "compact" / name
        (d / "compress").mkdir(parents=True)
        (d / "evaluation").mkdir()
        (d / "compress" / "report.json").write_text(json.dumps({"parameters": params}))
        (d / "evaluation" / "report.json").write_text(json.dumps({"sets": {"fleurs": {"cer": cer}}, "parity": {
            "cer_delta": delta, "cer_delta_ci95": ci, "gates": {"cer_delta_max": {"pass": ok}, "similarity_drop_max": {"pass": True}}}}))
        (d / "evaluation" / "speed.json").write_text(json.dumps({"variant": {"real_time_factor": rtf}}))
    card = modelcard.model_card(rec, work, {"request": {"steps": 100}, "seconds": 7200, "result": {}})
    section = card.split("## Compact variants")[1].split("\n## ")[0]
    assert "| [Sten-TTS-950M](https://huggingface.co/swedishembedded/sten-tts/tree/main/sten-tts-950m) | 952.6 M | 0.064 | yes | -0.0001 (-0.0031 to +0.0030) | 0.12 |" in section
    assert "| [Sten-TTS-475M](https://huggingface.co/swedishembedded/sten-tts/tree/main/sten-tts-475m) | 469.6 M | 0.076 | no | +0.0125 (+0.0103 to +0.0150) | 0.11 |" in section
    (work.root / "compact" / "sten-tts-475m" / "evaluation" / "report.json").unlink()
    import pytest
    with pytest.raises(FileNotFoundError):
        modelcard.model_card(rec, work, {"request": {"steps": 100}, "seconds": 7200, "result": {}})
