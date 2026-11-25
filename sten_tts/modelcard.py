# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in releasing
# text-to-speech models, you can procure our services by sending an email to
# info@swedishembedded.com.

"""The release's model card, written from the run's own reports."""

from __future__ import annotations

import json

from .config import Recipe, Workdir


SOURCE_NAMES = {"nst": "NST Swedish (read sentences, ten Swedish regions)", "rixvox": "RixVox-v2 (parliament debates)",
                "librivox": "LibriVox Swedish audiobooks (cut and checked by `segment`)", "fleurs": "FLEURS sv_se train and dev (read Wikipedia sentences)",
                "libritts_r": "LibriTTS-R train-clean-100 (English audiobooks, 24 kHz)"}
SET_NAMES = {"fleurs": "FLEURS sv_se test", "code_switching": "Swedish with English words and names", "numbers": "numbers, dates, units, abbreviations"}


def _evaluation(recipe: Recipe, work: Workdir) -> str:
    path = work.evaluation / "report.json"
    if not path.exists():
        return "Pending: the gated evaluation of this checkpoint has not finished; this card is updated with its results.\n"
    r = json.loads(path.read_text())
    lines = ["| gate | value | threshold | pass |", "|---|---|---|---|"]
    for name, g in r["gates"].items():
        lines.append(f"| {name} | {g['value']} | {g['threshold']} | {'yes' if g['pass'] else 'no'} |")
    lines += ["", "| voice | CER (FLEURS) | similarity to own recordings | identified as itself |", "|---|---|---|---|"]
    for v, m in r["voices"].items():
        lines.append(f"| {v} | {m['cer']:.3f} | {m['similarity']:.3f} | {m['identity_accuracy']:.2f} |")
    lines += ["", f"Base model on the same sentences: CER {r['base']['cer']:.3f}. Recogniser on the real held-out recordings: CER {r['floor']['cer']:.3f}."]
    sets = r.get("sets", {})
    if sets:
        names = list(sets)
        lines += ["", "CER per evaluation set (lower is better; the base model clones each voice from one recording):", "",
                  "| set | Sten | base |", "|---|---|---|"]
        lines += [f"| {n} | {sets[n]['cer']:.3f} | {'-' if sets[n]['base_cer'] is None else format(sets[n]['base_cer'], '.3f')} |" for n in names]
        lines += ["", "| voice | region | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for v in recipe["voices"]:
            lines.append(f"| {v['name']} | {v['region']} | " + " | ".join(f"{sets[n]['voices'].get(v['name'], float('nan')):.3f}" for n in names) + " |")
        regions = sorted({g for n in names for g in sets[n]["regions"]})
        lines += ["", "| region | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for g in regions:
            lines.append(f"| {g} | " + " | ".join(f"{sets[n]['regions'].get(g, float('nan')):.3f}" for n in names) + " |")
        lines += ["", "Sets: " + "; ".join(f"`{n}`: {SET_NAMES.get(n, n)}" for n in names) + "."]
    return "\n".join(lines) + "\n"


def model_card(recipe: Recipe, work: Workdir, train_report: dict) -> str:
    sel = json.loads((work.selection / "report.json").read_text())
    shard = json.loads((work.shards / "train.json").read_text())
    voices = json.loads((work.selection / "voices.json").read_text())
    train_hours = sel["hours"]["train"]
    sources = [s for s in recipe["selection"]["share"]]
    languages = [recipe["language"]["name"], *recipe["language"].get("base_languages", [])]
    iso = {"swedish": "sv", "english": "en"}
    lang_yaml = "\n".join(f"- {iso.get(l, l)}" for l in languages)
    data_rows = "\n".join(f"| {SOURCE_NAMES.get(s, s)} | {recipe['sources'][s]['license']} | {train_hours.get(s, 0):.1f} |" for s in sources)
    per_lang = shard.get("hours_per_language", {languages[0]: shard["hours"]})
    lang_text = ", ".join(f"{h} h {l}" for l, h in per_lang.items())
    has_score = all("score" in v for v in voices.values())
    voice_rows = "\n".join(f"| {name} | {v['gender']} | {v['region']} | {shard['voices'].get(name, 0)} |" + (f" {v['score']:.2f} |" if has_score else "")
                            for name, v in voices.items())
    voice_head = "| voice | gender | region of upbringing | training utterances |" + (" liveliness |" if has_score else "")
    voice_rule = "|---|---|---|---|" + ("---|" if has_score else "")
    attributions = " ".join(recipe["sources"][s]["attribution"] for s in sources if "attribution" in recipe["sources"][s])
    licences = "; ".join(f"{SOURCE_NAMES.get(s, s).split(' (')[0]} {recipe['sources'][s]['license']}" for s in sources)
    return f"""---
license: apache-2.0
language:
{lang_yaml}
base_model: {recipe['base']['repo']}
pipeline_tag: text-to-speech
tags:
- text-to-speech
- swedish
- qwen3-tts
---

# Sten-TTS-1.7B

A Swedish text-to-speech model with twelve named voices, fine-tuned from
[{recipe['base']['repo']}](https://huggingface.co/{recipe['base']['repo']}) (revision
`{recipe['base']['revision']}`) on {shard['hours']} hours of speech ({lang_text}) with brain.
It is a Qwen3-TTS `custom_voice` checkpoint in the upstream layout and adds the
language `swedish`; English is trained alongside, so English words and names inside
Swedish sentences keep their English pronunciation.

## Use

With the upstream package (`pip install qwen-tts==0.1.1`):

```python
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained("{recipe['publish']['repo']}")
wavs, sr = model.generate_custom_voice(text="Hej och välkommen!", speaker="astrid", language="swedish")
sf.write("astrid.wav", wavs[0], sr)
```

With brain, through its OpenAI-compatible HTTP API:

```bash
brain pull {recipe['publish']['repo']}          # the repository is private for now: `hf auth login` with access first
brain serve --openai 127.0.0.1:8080             # prints the API key it generates (or pass --api-key-file)
curl http://127.0.0.1:8080/v1/audio/speech \\
  -H "Authorization: Bearer $BRAIN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{recipe['publish']['repo']}", "voice": "astrid", "language": "swedish",
       "input": "Hej och välkommen till Sten!", "response_format": "wav"}}' \\
  -o astrid.wav
```

`voice` is any of the twelve below. Always send `"language": "swedish"`: without it brain speaks the text as English.

## Voices

{voice_head}
{voice_rule}
{voice_rows}

The voices are anonymous speakers of the NST Swedish speech database (CC0), each the
liveliest-sounding speaker of their gender and region with enough recordings (pitch range
and variability, loudness dynamics and speaking rate, measured from their own audio; the
score is a weighted sum of z-scores among the speakers of the same gender). Their
recordings are 16 kHz, so the 24 kHz output is band-limited.

## Training

| source | licence | hours selected |
|---|---|---|
{data_rows}

- Data: {shard['hours']} h after cleaning and the cut, {shard['records']} utterances, {sel['speakers']['train']} speakers.
  RixVox speakers and the English speakers are never named voices.
- Objective: full fine-tune of the Talker and the code predictor (teacher-forced codebook-0 and residual
  cross-entropy, residual weight {recipe['train']['mtp_weight']}), both text layouts the model is served with; the text
  embedding and projection frozen; new codec rows for `swedish` and every voice; English uses the base's own row.
- Schedule: {recipe['train']['epochs']} epochs, {train_report['request']['steps']} steps of {recipe['train']['batch_utterances']} x {recipe['train']['grad_accum']} utterances,
  learning rate {recipe['train']['lr']} (cosine, {recipe['train']['warmup_fraction']:.0%} warm-up), seed {recipe.seed}.
- Run: one NVIDIA GH200, {train_report['seconds'] / 3600:.1f} h wall clock; loss {train_report['result'].get('initial_loss', float('nan')):.2f} -> {train_report['result'].get('final_loss', float('nan')):.2f}.
- Recipe: the `sten-tts` repository, recipe `{recipe.name}`, every source pinned to a revision.

## Evaluation

Every voice reads {recipe['evaluate']['fleurs_sentences']} FLEURS sv_se test sentences (never trained on) and the
recipe's hand-written sets; the recogniser is {recipe['judge']['repo']}.

{_evaluation(recipe, work)}
## Limitations

- Most Swedish training audio is 16 kHz, so the output lacks the top of the 24 kHz band.
- The voices are read-speech speakers recorded around 1999; no style instruction is supported.
- Swedish text is read as written: numbers, abbreviations and symbols follow whatever the training text held
  (no separate text normalisation); the `numbers` set above measures how well.
- Finland Swedish is not covered.

## Licence and attribution

Apache-2.0, as the base model. Sources: {licences}. {attributions}
"""
