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


def load(path: str | Path) -> Recipe:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    def evaluation(self) -> Path:
        return self.root / "evaluation"
