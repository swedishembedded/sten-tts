# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The release's samples are evaluation clips of unseen sentences, unpicked, in the recipe's order, and the card plays them."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sten_tts import config, modelcard, samples

RECIPE = Path(__file__).resolve().parents[1] / "recipe" / "sten-tts-1.7b-v2.yaml"


def test_samples_come_from_the_evaluation_seed_in_order_and_play_on_the_card(tmp_path: Path) -> None:
    rec = config.load(RECIPE)
    rec.raw["publish"]["samples"]["sentence"] = {"signe": 1, "arvid": 1}  # the per-voice override, exercised here
    work = config.Workdir(tmp_path)
    for d in (work.selection, work.shards, work.evaluation):
        d.mkdir(parents=True)
    sentences = [{"id": f"fleurs:{i}", "text": f"Mening {i}.", "references": [f"Mening {i}."]} for i in range(3)]
    (work.evaluation / "sentences.json").write_text(json.dumps({"fleurs": sentences}))
    rows = [{"system": system, "set": "fleurs", "voice": v["name"], "sentence": s["id"], "seed": seed, "text": s["text"],
             "heard": s["text"], "cer": 0.0 if seed == 1 else 0.5, "wav": f"{system}-{v['name']}-{s['id']}-{seed}".encode()}
            for system in ("sten", "base") for v in rec["voices"] for s in sentences for seed in (1, 2)]
    pq.write_table(pa.Table.from_pylist(rows), work.evaluation / "clips.parquet")
    listing = samples.write(rec, work)
    assert [i["voice"] for i in listing][0] == "axel" and len(listing) == len(rec["voices"])
    by = {i["voice"]: i for i in listing}
    assert by["signe"]["sentence"] == "fleurs:1" and by["arvid"]["sentence"] == "fleurs:1" and by["astrid"]["sentence"] == "fleurs:0"
    assert (work.release / "samples" / "signe.wav").read_bytes() == b"sten-signe-fleurs:1-1", "Sten's own seed-1 take, never the base's"

    (work.selection / "report.json").write_text(json.dumps({"hours": {"train": {"nst": 1.0}}, "speakers": {"train": 10}}))
    (work.selection / "voices.json").write_text(json.dumps({v["name"]: {"gender": v["gender"], "region": v["region"]} for v in rec["voices"]}))
    (work.shards / "train.json").write_text(json.dumps({"hours": 420.0, "records": 194769, "voices": {}}))
    card = modelcard.model_card(rec, work, {"request": {"steps": 1}, "seconds": 60, "result": {}})
    head = card.split("## Usage")[0]
    assert f'src="https://huggingface.co/{rec["publish"]["repo"]}/resolve/main/samples/axel.wav"' in head
    assert head.index("**Axel**") < head.index("**Astrid**")
    assert "Signe and Arvid read a different held-out sentence" in head and head.count("> Mening") == 2, "each sentence shown once"
    assert f"https://huggingface.co/datasets/{rec['dataset']['repo']}" in head and "420 hours" in head
