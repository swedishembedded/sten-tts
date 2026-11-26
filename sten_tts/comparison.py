# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements compression of speech models to the hardware
# a product has, without audible loss, for its clients. If your team needs
# expertise in shrinking text-to-speech models, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Every voice in every size, on the same sentences: the full model's card lets a listener compare.

`publish.comparison` names the evaluation sentences and the seed. For the
full model and each of its compact variants (`publish.family`) the clip of
every voice reading every named sentence comes from that model's own
evaluation (`evaluation/clips.parquet`, a variant's under
`compact/<name>/evaluation/`): the take the evaluation scored, unpicked, so
all sizes read the same text with the same seed. A sentence given as
`{id, text, references}` instead is spoken fresh by every size through the
brain server, with the same seed, and the recogniser scores each clip against
the closest reference. The clips go to
`release/comparison/<model>/` with a listing, and the card plays them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from concurrent.futures import ThreadPoolExecutor

from .config import Recipe, Workdir

LISTING = "comparison/comparison.json"


def _models(recipe: Recipe, work: Workdir) -> list[tuple[str, str, Path, str]]:
    """(name, title, clips, served id) of the full model and each family variant, in the recipe's order."""
    from . import config
    out = [(recipe.name, recipe.title, work.evaluation / "clips.parquet", "local/" + recipe["train"]["install_as"])]
    for f in recipe["publish"].get("family", []):
        v = config.load(recipe.path.parent / f)
        out.append((v.name, v.title, work.root / "compact" / v.name / "evaluation" / "clips.parquet", "local/" + v["train"]["install_as"]))
    return out


def _pcm16k(wav: bytes) -> bytes:
    from .evaluate import pcm16k
    return pcm16k(wav)


def _speak(recipe: Recipe, pool, model: str, voices: list[str], sentence: dict, seed: int, concurrency: int) -> dict[str, dict]:
    """Every voice of `model` reading `sentence` with `seed`, heard by the recipe's recogniser."""
    from . import metrics
    from .evaluate import ASR_RATE
    judge = recipe["judge"].get("served_as", recipe["judge"]["repo"])
    lang = recipe["language"]["name"]

    def one(v: str) -> dict:
        wav = pool.get().run(model, "design", {"text": sentence["text"], "speaker": v, "lang": lang, "seed": seed}).blobs["audio"]
        heard = pool.get().run(judge, "transcribe", {"sample_rate": ASR_RATE, "language": recipe["evaluate"]["judge_language"]},
                               blobs={"audio": _pcm16k(wav)}, meta={"audio": {"media": "audio", "sample_rate": ASR_RATE}}).text("text")
        refs = sentence.get("references", [sentence["text"]])
        return {"voice": v, "sentence": sentence["id"], "text": sentence["text"], "heard": heard,
                "cer": min(metrics.cer(r, heard) for r in refs), "wav": wav}

    with ThreadPoolExecutor(concurrency) as ex:
        return {(r["voice"], r["sentence"]): r for r in ex.map(one, voices)}


def write(recipe: Recipe, work: Workdir, pool=None, concurrency: int = 8) -> dict:
    spec = recipe["publish"]["comparison"]
    voices = recipe["publish"]["samples"]["order"]
    entries = [e if isinstance(e, dict) else {"id": e} for e in spec["sentences"]]
    fresh = [e for e in entries if "text" in e]
    if fresh and pool is None:
        raise RuntimeError(f"publish.comparison: {[e['id'] for e in fresh]} are spoken fresh and need a brain server (--brain-bus)")
    wanted = [e["id"] for e in entries]
    stored = [e["id"] for e in entries if "text" not in e]
    out = work.release / "comparison"
    if out.exists():
        for old in out.rglob("*.wav"):
            old.unlink()
    models, texts, table = [], {}, {s: {v: [] for v in voices} for s in wanted}
    for name, title, path, served in _models(recipe, work):
        rows = pq.read_table(path, columns=["system", "voice", "sentence", "seed", "text", "heard", "cer", "wav"],
                             filters=[("system", "=", "sten"), ("seed", "=", spec["seed"]), ("sentence", "in", stored)]).to_pylist() if stored else []
        found = {(r["voice"], r["sentence"]): r for r in rows}
        for e in fresh:
            found.update(_speak(recipe, pool, served, voices, e, spec["seed"], concurrency))
        missing = sorted({(v, s) for v in voices for s in wanted} - set(found))
        if missing:
            raise RuntimeError(f"{path}: no seed-{spec['seed']} clip of {title} for {missing[:6]}{' ...' if len(missing) > 6 else ''}")
        (out / name).mkdir(parents=True, exist_ok=True)
        models.append({"name": name, "title": title})
        for s in wanted:
            for v in voices:
                r = found[(v, s)]
                texts.setdefault(s, r["text"])
                file = f"comparison/{name}/{v}-{s.replace(':', '-')}.wav"
                (work.release / file).write_bytes(r["wav"])
                table[s][v].append({"file": file, "cer": round(float(r["cer"]), 4), "heard": r["heard"]})
    listing = {"seed": spec["seed"], "models": models, "sentences": [{"id": s, "text": texts[s]} for s in wanted], "clips": table}
    (work.release / LISTING).write_text(json.dumps(listing, ensure_ascii=False, indent=1))
    return listing


def section(recipe: Recipe, work: Workdir) -> str:
    """The card's comparison section, from the listing `write` left in the release; empty without one."""
    path = work.release / LISTING
    if "comparison" not in recipe["publish"] or not path.exists():
        return ""
    from .modelcard import _region
    listing = json.loads(path.read_text())
    region = {v["name"]: v["region"] for v in recipe["voices"]}
    base = f"https://huggingface.co/{recipe['publish']['repo']}/resolve/main"
    titles = [m["title"] for m in listing["models"]]
    lines = ["", "## Listen across sizes", "",
             f"Every voice in {', '.join(titles[:-1])} and {titles[-1]}, reading the same sentences with the same seed: the "
             f"first take each model's evaluation made (seed {listing['seed']}), unpicked, on text none of them was trained on. "
             "CER is the recogniser's character error rate on that one clip; the means over all clips are in each model's card.", ""]
    for s in listing["sentences"]:
        lines += [f"> {s['text']}", "", "| voice | region | " + " | ".join(titles) + " |", "|---|---|" + "---|" * len(titles)]
        for v, cells in listing["clips"][s["id"]].items():
            audio = [f'<audio controls preload="none" src="{base}/{c["file"]}"></audio><br>CER {c["cer"]:.3f}' for c in cells]
            lines.append(f"| **{v.capitalize()}** | {_region(region[v])} | " + " | ".join(audio) + " |")
        lines.append("")
    return "\n".join(lines)
