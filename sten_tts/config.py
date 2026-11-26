# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in building speech
# datasets and training text-to-speech models, you can procure our services
# by sending an email to info@swedishembedded.com.

"""The recipe file and the work directory every stage reads and writes.

The recipe is validated strictly: an unknown or missing key is an error naming
it, so a typo never silently falls back to a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_TOP = {"name", "seed", "base", "judge", "language", "sources", "selection", "audio", "voices", "train", "evaluate", "publish"}
# Stages a recipe may leave out (v1 had none of them).
_OPTIONAL_TOP = {"segment", "liveliness", "dataset"}
_SEGMENT = {"source", "min_seconds", "max_seconds", "min_pause_seconds", "silence_db", "max_cer", "asr_language"}
# A compact variant names the full recipe it is carved from and takes the data, voices, evaluation and gates
# from it; these are all it may set.
_VARIANT_TOP = {"name", "title", "variant_of", "teacher", "compress", "train", "evaluate", "publish"}
_TEACHER = {"model", "title", "repo", "parameters"}
_COMPRESS = {"max_parameters", "records", "held_out_records", "install_as", "plan"}
_TRAIN_VARIANT = {"install_as", "epochs", "lr", "weight_decay", "warmup_fraction", "batch_utterances", "grad_accum", "max_frames",
                  "mtp_weight", "streaming_fraction", "eval_every", "eval_batches", "patience", "keep_best", "distill"}
_DISTILL = {"gt_weight", "talker_kd_weight", "mtp_kd_weight", "hidden_kd_weight", "kd_temperature"}
_PARITY = {"gate", "teacher_evaluation", "cer_max_delta", "similarity_max_drop", "identity_max_drop", "failure_max_delta"}
_SPEED = {"sentences", "voice", "warmup"}


class RecipeError(ValueError):
    """The recipe file is malformed."""


def _require(d: dict, keys: set[str], where: str, optional: set[str] = frozenset()) -> None:
    if not isinstance(d, dict):
        raise RecipeError(f"{where}: expected a mapping, got {type(d).__name__}")
    unknown = set(d) - keys - set(optional)
    if unknown:
        raise RecipeError(f"{where}: unknown keys {sorted(unknown)}")
    missing = keys - set(d)
    if missing:
        raise RecipeError(f"{where}: missing keys {sorted(missing)}")


@dataclass(frozen=True)
class Recipe:
    """A loaded, validated recipe. `raw` holds the whole mapping, for stages to read."""

    raw: dict[str, Any]
    path: Path

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def is_variant(self) -> bool:
        """A compact variant of a full model, carved from it and distilled from it."""
        return "variant_of" in self.raw

    @property
    def title(self) -> str:
        """The model's name in its card."""
        return self.raw.get("title", "Sten-TTS-1.7B")


def _load_variant(path: Path, raw: dict) -> Recipe:
    """A variant over its full recipe: `train` replaces the full recipe's, `evaluate` and `publish` add to theirs."""
    _require(raw, _VARIANT_TOP, str(path))
    parent = load(path.parent / raw["variant_of"])
    if parent.is_variant:
        raise RecipeError(f"{path}: variant_of names another variant; a variant is carved from a full model")
    _require(raw["teacher"], _TEACHER, f"{path}: teacher")
    c = raw["compress"]
    _require(c, _COMPRESS, f"{path}: compress", optional={"corpus", "from"})
    if "from" in c:
        _require(c["from"], {"model", "title"}, f"{path}: compress.from")
    if c.get("corpus", "inventory") != "inventory":
        raise RecipeError(f"{path}: compress.corpus must be `inventory` (the full run's candidate utterance texts) or left out")
    if not isinstance(c["plan"], list) or not c["plan"] or not all(isinstance(e, dict) and len(e) == 1 for e in c["plan"]):
        raise RecipeError(f"{path}: compress.plan must be a non-empty list of one-key edits")
    if "corpus" in c and not any("trim_vocabulary" in e for e in c["plan"]):
        raise RecipeError(f"{path}: compress.corpus serves a trim_vocabulary edit, and the plan has none")
    if not isinstance(c["max_parameters"], int) or c["max_parameters"] <= 0:
        raise RecipeError(f"{path}: compress.max_parameters must be a positive integer")
    t = raw["train"]
    _require(t, _TRAIN_VARIANT, f"{path}: train", optional={"start_from"})
    _require(t["distill"], _DISTILL, f"{path}: train.distill", optional={"hidden_kd_layers"})
    if c["install_as"] == t["install_as"]:
        raise RecipeError(f"{path}: compress.install_as and train.install_as must differ")
    ev = raw["evaluate"]
    if not isinstance(ev, dict) or "parity" not in ev:
        raise RecipeError(f"{path}: evaluate.parity is required: a variant is judged against its full model")
    _require(ev, {"parity"}, f"{path}: evaluate", optional={"speed"})
    _require(ev["parity"], _PARITY, f"{path}: evaluate.parity")
    if not isinstance(ev["parity"]["gate"], bool):
        raise RecipeError(f"{path}: evaluate.parity.gate must be true or false: whether parity with the full model blocks the release")
    if "speed" in ev:
        _require(ev["speed"], _SPEED, f"{path}: evaluate.speed")
        if ev["speed"]["voice"] not in {v["name"] for v in parent["voices"]}:
            raise RecipeError(f"{path}: evaluate.speed.voice {ev['speed']['voice']!r} is not a voice")
    if not isinstance(raw["publish"], dict) or "subfolder" not in raw["publish"]:
        raise RecipeError(f"{path}: publish.subfolder is required: a variant is published beside its full model")
    _require(raw["publish"], {"subfolder"}, f"{path}: publish")
    merged = {**parent.raw, "name": raw["name"], "title": raw["title"], "variant_of": raw["variant_of"], "teacher": raw["teacher"],
              "compress": c, "train": t, "evaluate": {**parent["evaluate"], **ev}, "publish": {**parent["publish"], **raw["publish"]}}
    return Recipe(raw=merged, path=path)


def load(path: str | Path) -> Recipe:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "variant_of" in raw:
        return _load_variant(path, raw)
    _require(raw, _TOP, str(path), optional=_OPTIONAL_TOP)
    for name in ("base", "judge"):
        _require(raw[name], {"repo", "revision"}, f"{path}: {name}", optional={"license", "served_as"})
    _require(raw["language"], {"name", "init_from"}, f"{path}: language", optional={"base_languages"})
    for src, spec in raw["sources"].items():
        _require(spec, {"repo", "revision", "license", "files", "format"}, f"{path}: sources.{src}", optional={"attribution"})
        if len(spec["revision"]) != 40:
            raise RecipeError(f"{path}: sources.{src}.revision must be a full 40-hex commit, got {spec['revision']!r}")
    if "segment" in raw:
        seg = raw["segment"]
        _require(seg, _SEGMENT, f"{path}: segment")
        if seg["source"] not in raw["sources"]:
            raise RecipeError(f"{path}: segment.source {seg['source']!r} is not a source")
        if not 0 < seg["min_seconds"] < seg["max_seconds"]:
            raise RecipeError(f"{path}: segment: need 0 < min_seconds < max_seconds")
    if "liveliness" in raw:
        live = raw["liveliness"]
        _require(live, {"min_minutes", "utterances", "weights"}, f"{path}: liveliness")
        _require(live["weights"], {"f0_range", "f0_sd", "energy_sd", "syllable_rate"}, f"{path}: liveliness.weights")
    if "dataset" in raw:
        _require(raw["dataset"], {"repo", "name", "rows_per_shard"}, f"{path}: dataset")
    _require(raw["publish"], {"repo", "private"}, f"{path}: publish", optional={"samples", "subfolder", "family", "comparison"})
    family = raw["publish"].get("family", [])
    if not isinstance(family, list) or not all(isinstance(f, str) for f in family):
        raise RecipeError(f"{path}: publish.family must list variant recipe files (relative to this one)")
    if "comparison" in raw["publish"]:
        cmp = raw["publish"]["comparison"]
        _require(cmp, {"seed", "sentences"}, f"{path}: publish.comparison")
        if "samples" not in raw["publish"] or not family:
            raise RecipeError(f"{path}: publish.comparison needs publish.samples (the voice order) and publish.family (the sizes)")
        if cmp["seed"] not in raw["evaluate"]["seeds"]:
            raise RecipeError(f"{path}: publish.comparison.seed {cmp['seed']} is not an evaluation seed {raw['evaluate']['seeds']}")
        if not cmp["sentences"]:
            raise RecipeError(f"{path}: publish.comparison.sentences is empty")
        for x in cmp["sentences"]:
            if isinstance(x, dict):
                _require(x, {"id", "text"}, f"{path}: publish.comparison.sentences", optional={"references"})
            elif not isinstance(x, str):
                raise RecipeError(f"{path}: publish.comparison.sentences: an evaluation sentence id or {{id, text, references}}, got {x!r}")
    if "parity" in raw["evaluate"]:
        raise RecipeError(f"{path}: evaluate.parity belongs to a variant recipe (one with variant_of)")
    if "samples" in raw["publish"]:
        smp = raw["publish"]["samples"]
        _require(smp, {"seed", "order"}, f"{path}: publish.samples", optional={"sentence"})
        if smp["seed"] not in raw["evaluate"]["seeds"]:
            raise RecipeError(f"{path}: publish.samples.seed {smp['seed']} is not an evaluation seed {raw['evaluate']['seeds']}")
        if sorted(smp["order"]) != sorted(v["name"] for v in raw["voices"]):
            raise RecipeError(f"{path}: publish.samples.order must name every voice exactly once")
        unknown = set(smp.get("sentence", {})) - {v["name"] for v in raw["voices"]}
        if unknown:
            raise RecipeError(f"{path}: publish.samples.sentence names {sorted(unknown)}, which are not voices")
    sel = raw["selection"]
    if not isinstance(sel, dict) or not isinstance(sel.get("share"), dict):
        raise RecipeError(f"{path}: selection.share must be a mapping of source to share")
    # One block per source that has a share, with at least its per-speaker cap.
    _require(sel, {"hours", "share", "oversample", "duration_seconds", "held_out", *sel["share"]}, f"{path}: selection")
    if abs(sum(sel["share"].values()) - 1.0) > 1e-9:
        raise RecipeError(f"{path}: selection.share must sum to 1, got {sel['share']}")
    for src in sel["share"]:
        if src not in raw["sources"]:
            raise RecipeError(f"{path}: selection.share names {src!r}, which is not a source")
        if "cap_minutes_per_speaker" not in sel[src]:
            raise RecipeError(f"{path}: selection.{src}: missing keys ['cap_minutes_per_speaker']")
    unknown = set(sel["held_out"]["unseen_speakers"]) - set(sel["share"])
    if unknown:
        raise RecipeError(f"{path}: selection.held_out.unseen_speakers: {sorted(unknown)} have no share")
    names = [v["name"] for v in raw["voices"]]
    if len(set(names)) != len(names):
        raise RecipeError(f"{path}: voices: duplicate names in {names}")
    for v in raw["voices"]:
        _require(v, {"name", "gender", "region"}, f"{path}: voice {v.get('name')}")
        if v["gender"] not in ("female", "male"):
            raise RecipeError(f"{path}: voice {v['name']}: gender must be female or male")
    return Recipe(raw=raw, path=path)


@dataclass(frozen=True)
class Workdir:
    """Where a run's state lives. Every stage takes it as a flag; nothing is global."""

    root: Path

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            object.__setattr__(self, "root", self.root.resolve())

    def raw(self, source: str) -> Path:
        return self.root / "raw" / source

    @property
    def inventory(self) -> Path:
        return self.root / "inventory"

    @property
    def selection(self) -> Path:
        return self.root / "selection"

    @property
    def segments(self) -> Path:
        return self.root / "segments"

    @property
    def audio(self) -> Path:
        return self.root / "audio"

    @property
    def shards(self) -> Path:
        return self.root / "shards"

    @property
    def train(self) -> Path:
        return self.root / "train"

    @property
    def release(self) -> Path:
        return self.root / "release"

    @property
    def compress(self) -> Path:
        return self.root / "compress"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"
