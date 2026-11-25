# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A compact variant's recipe: it inherits the full recipe, is validated as strictly, and drives surgery and distillation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sten_tts import compress, config, train

RECIPES = Path(__file__).resolve().parents[1] / "recipe"
VARIANT = RECIPES / "sten-tts-950m.yaml"


def _variant(tmp_path: Path, edit) -> Path:
    raw = yaml.safe_load(VARIANT.read_text())
    edit(raw)
    path = tmp_path / "variant.yaml"
    raw["variant_of"] = str(RECIPES / "sten-tts-1.7b-v2.yaml")
    path.write_text(yaml.safe_dump(raw))
    return path


def test_a_variant_inherits_the_data_voices_and_gates_and_adds_its_own_stages() -> None:
    full, var = config.load(RECIPES / "sten-tts-1.7b-v2.yaml"), config.load(VARIANT)
    assert var.is_variant and not full.is_variant
    assert var["voices"] == full["voices"] and var["sources"] == full["sources"] and var["selection"] == full["selection"]
    assert var["evaluate"]["gates"] == full["evaluate"]["gates"] and var["evaluate"]["seeds"] == full["evaluate"]["seeds"]
    assert var["evaluate"]["parity"]["teacher_evaluation"] == "teacher"
    assert var["publish"]["repo"] == full["publish"]["repo"] and var["publish"]["samples"] == full["publish"]["samples"]
    assert var["publish"]["subfolder"] == "sten-tts-950m" and var.name == "sten-tts-950m"
    assert var["train"]["install_as"] != full["train"]["install_as"]
    assert [next(iter(e)) for e in var["compress"]["plan"]] == ["trim_vocabulary", "select_channels", "remove_layers"]


@pytest.mark.parametrize("edit, message", [
    (lambda r: r.update(colour="red"), "unknown keys"),
    (lambda r: r.pop("teacher"), "missing keys"),
    (lambda r: r["compress"].update(plan=[]), "plan"),
    (lambda r: r["compress"].update(max_parameters=0), "max_parameters"),
    (lambda r: r["train"]["distill"].update(alpha=1), "unknown keys"),
    (lambda r: r["train"].pop("distill"), "missing keys"),
    (lambda r: r["evaluate"].pop("parity"), "parity"),
    (lambda r: r["publish"].pop("subfolder"), "subfolder"),
    (lambda r: r.update(voices=[]), "unknown keys"),  # the data and voices are the full recipe's alone
])
def test_a_malformed_variant_is_refused_by_name(tmp_path: Path, edit, message: str) -> None:
    with pytest.raises(config.RecipeError, match=message):
        config.load(_variant(tmp_path, edit))


def _shards(work: config.Workdir, records: int = 1000) -> None:
    work.shards.mkdir(parents=True)
    (work.shards / "train.json").write_text(json.dumps({"records": records}))


def test_distillation_trains_the_cut_checkpoint_against_the_full_model_with_no_new_rows(tmp_path: Path) -> None:
    rec = config.load(VARIANT)
    work = config.Workdir(tmp_path)
    _shards(work, records=1000)
    p = train.request(rec, work, ranks=4)
    assert p["base"] == "local/" + rec["compress"]["install_as"] and p["teacher"] == rec["teacher"]["model"]
    assert json.loads(p["languages"]) == {"swedish": {"new": False}, "english": {"new": False}}
    assert json.loads(p["voices"]) == {v["name"]: {"new": False} for v in rec["voices"]}
    assert "new_row_lr_multiplier" not in p
    d = rec["train"]["distill"]
    assert {k: p[k] for k in d} == d and "hidden_kd_layers" not in p
    assert p["steps"] == -(-1000 // (rec["train"]["batch_utterances"] * 4)) * rec["train"]["epochs"], "whole epochs over 4 cards"
    assert p["install_as"] == rec["train"]["install_as"] and p["keep_best"] and p["patience"] == 4


def test_the_full_recipe_still_adds_its_language_and_voices(tmp_path: Path) -> None:
    rec = config.load(RECIPES / "sten-tts-1.7b-v2.yaml")
    work = config.Workdir(tmp_path)
    _shards(work)
    p = train.request(rec, work, ranks=1)
    assert p["base"] == rec["base"]["repo"] and "teacher" not in p
    assert json.loads(p["languages"])["swedish"]["new"] is True and p["new_row_lr_multiplier"] == 5.0


def test_the_surgery_request_carries_the_plan_and_probes_with_the_evaluation_sentences(tmp_path: Path) -> None:
    rec = config.load(VARIANT)
    work = config.Workdir(tmp_path)
    work.evaluation.mkdir(parents=True)
    (work.evaluation / "sentences.json").write_text(json.dumps({"fleurs": [{"id": "f:0", "text": "Hej  då."}], "numbers": [{"id": "n:0", "text": "Klockan 9."}]}))
    work.inventory.mkdir(parents=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"text": ["Hej.", None, "Ett två."]}), work.inventory / "nst.parquet")
    pq.write_table(pa.table({"text": ["Good day."]}), work.inventory / "libritts_r.parquet")
    params, blobs = compress.request(rec, work)
    assert sorted(blobs["corpus"].decode().splitlines()) == ["Ett två.", "Good day.", "Hej."], "every source's candidate texts"
    assert params["base"] == rec["teacher"]["model"] and json.loads(params["plan"]) == rec["compress"]["plan"]
    assert params["records"] == 128 and params["held_out_records"] == 128 and params["install_as"] == rec["compress"]["install_as"]
    assert blobs["probe"].decode().splitlines() == ["Hej  då.", "Klockan 9."]


def _safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    header = json.dumps({n: {"dtype": "BF16", "shape": s, "data_offsets": [0, 0]} for n, s in shapes.items()} | {"__metadata__": {}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def test_the_cut_is_refused_if_it_is_larger_than_the_rung_or_changes_how_a_training_text_reads(tmp_path: Path) -> None:
    ck = tmp_path / "checkpoint"
    ck.mkdir()
    _safetensors(ck / "model.safetensors", {"a": [1000, 900], "b": [50]})
    assert compress.parameters(ck / "model.safetensors") == 900_050
    ok = {"vocabulary": {"corpus_changed": 0, "probe_changed": 3}}
    compress.check(ok, ck, max_parameters=900_050)
    with pytest.raises(RuntimeError, match="900,050 parameters"):
        compress.check(ok, ck, max_parameters=900_000)
    with pytest.raises(RuntimeError, match="corpus_changed"):
        compress.check({"vocabulary": {"corpus_changed": 2}}, ck, max_parameters=10**9)


def test_speed_is_wall_time_over_audio_time_and_codec_frames_a_second() -> None:
    from sten_tts import speed
    s = speed.summarize([(1.0, 4.0), (1.0, 4.0)])
    assert s["real_time_factor"] == 0.25 and s["frames_per_second"] == 50.0 and s["requests"] == 2
    with pytest.raises(ValueError):
        speed.summarize([])
