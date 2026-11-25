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


def _evaluation(work: Workdir) -> str:
    path = work.evaluation / "report.json"
    if not path.exists():
        return "Pending: the gated evaluation of this checkpoint has not finished; this card is updated with its results.\n"
    r = json.loads(path.read_text())
    lines = ["| gate | value | threshold | pass |", "|---|---|---|---|"]
    for name, g in r["gates"].items():
        lines.append(f"| {name} | {g['value']} | {g['threshold']} | {'yes' if g['pass'] else 'no'} |")
    lines.append("")
    lines.append("| voice | CER | similarity to own recordings | identified as itself |")
    lines.append("|---|---|---|---|")
    for v, m in r["voices"].items():
        lines.append(f"| {v} | {m['cer']:.3f} | {m['similarity']:.3f} | {m['identity_accuracy']:.2f} |")
    lines.append("")
    lines.append(f"Base model on the same sentences: CER {r['base']['cer']:.3f}. Recogniser on the real held-out recordings: CER {r['floor']['cer']:.3f}.")
    return "\n".join(lines) + "\n"


def model_card(recipe: Recipe, work: Workdir, train_report: dict) -> str:
    sel = json.loads((work.selection / "report.json").read_text())
    shard = json.loads((work.shards / "train.json").read_text())
    voices = json.loads((work.selection / "voices.json").read_text())
    rix = recipe["sources"]["rixvox"]
    voice_rows = "\n".join(f"| {name} | {v['gender']} | {v['region']} | {shard['voices'].get(name, 0)} |" for name, v in voices.items())
    return f"""---
license: apache-2.0
language:
- sv
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
`{recipe['base']['revision']}`) on {shard['hours']} hours of Swedish speech with brain.
It is a Qwen3-TTS `custom_voice` checkpoint in the upstream layout and adds the
language `swedish`.

## Use

With the upstream package (`pip install qwen-tts==0.1.1`):

```python
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained("{recipe['publish']['repo']}")
wavs, sr = model.generate_custom_voice(text="Hej och välkommen!", speaker="astrid", language="swedish")
sf.write("astrid.wav", wavs[0], sr)
```

With brain: `brain pull {recipe['publish']['repo']}`, then
`POST /v1/audio/speech` with `"model": "{recipe['publish']['repo']}"`, `"voice": "astrid"`,
`"language": "swedish"`.

## Voices

| voice | gender | region of upbringing | training utterances |
|---|---|---|---|
{voice_rows}

The voices are anonymous speakers of the NST Swedish speech database (CC0).
Recordings are 16 kHz, so the 24 kHz output is band-limited.

## Training

- Data: {shard['hours']} h of Swedish speech, {shard['records']} utterances, {sel['speakers']['train']} speakers: NST
  (CC0, read sentences, ten Swedish regions) and RixVox-v2 (ODC-BY, parliament debates); {sel['hours']['train'].get('nst', 0):.1f} h
  and {sel['hours']['train'].get('rixvox', 0):.1f} h were selected, then cleaned and cut to {shard['hours']} h. RixVox speakers
  are never named voices.
- Objective: full fine-tune of the Talker and the code predictor (teacher-forced codebook-0 and residual
  cross-entropy, residual weight {recipe['train']['mtp_weight']}), both text layouts the model is served with; the text
  embedding and projection frozen; new codec rows for `swedish` and every voice.
- Schedule: {recipe['train']['epochs']} epochs, {train_report['request']['steps']} steps of {recipe['train']['batch_utterances']} x {recipe['train']['grad_accum']} utterances,
  learning rate {recipe['train']['lr']} (cosine, {recipe['train']['warmup_fraction']:.0%} warm-up), seed {recipe.seed}.
- Run: one NVIDIA GH200, {train_report['seconds'] / 3600:.1f} h wall clock; loss {train_report['result'].get('initial_loss', float('nan')):.2f} -> {train_report['result'].get('final_loss', float('nan')):.2f}.
- Recipe: the `sten-tts` repository, recipe `{recipe.name}`, every source pinned to a revision.

## Evaluation

Every voice reads {recipe['evaluate']['fleurs_sentences']} FLEURS sv_se test sentences (never trained on); the
recogniser is {recipe['judge']['repo']}.

{_evaluation(work)}
## Limitations

- Trained on 16 kHz recordings, so the output lacks the top of the 24 kHz band.
- The voices are read-speech speakers recorded around 1999; expressive, conversational or emotional styles
  were not trained, and no style instruction is supported.
- Swedish text is read as written: numbers, abbreviations and symbols follow whatever the training text held
  (no separate text normalisation).
- Finland Swedish is not covered.

## Licence and attribution

Apache-2.0, as the base model. NST is CC0-1.0. {rix['attribution']}
"""
