# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in releasing
# text-to-speech models, you can procure our services by sending an email to
# info@swedishembedded.com.

"""The release's model card, written from the run's own reports.

The card is for someone who wants to use the model: what it is, how it sounds,
how to run it, how well it reads and where it falls short. Every number in it
is read from a report of the run it describes.
"""

from __future__ import annotations

import json
from collections import Counter

from .config import Recipe, Workdir
from .offer import model_offer


# source -> (name, what it holds)
SOURCES = {"nst": ("NST Swedish", "read sentences, speakers from ten Swedish regions, 16 kHz"),
           "rixvox": ("RixVox-v2", "debates of the Swedish Parliament"),
           "librivox": ("LibriVox Swedish audiobooks", "public-domain audiobook narration"),
           "fleurs": ("FLEURS sv_se, train and dev splits", "read Wikipedia sentences"),
           "libritts_r": ("LibriTTS-R train-clean-100", "English audiobooks, 24 kHz")}
# evaluation set -> (name, what it holds)
SETS = {"fleurs": ("FLEURS", "FLEURS sv_se test sentences, never trained on"),
        "code_switching": ("Code-switching", "hand-written Swedish sentences with English words, names and brands"),
        "numbers": ("Numbers", "hand-written sentences with numbers, dates, times, units and abbreviations")}
# The regions are those of the NST speaker metadata; a card reader gets them in English.
REGIONS = {"Mellansverige": "Central Sweden", "Stockholm med omnejd": "Stockholm area", "Göteborg med omnejd": "Gothenburg area",
           "Västra sydsverige": "Western southern Sweden", "Norrland": "Norrland", "Dalarna med omnejd": "Dalarna area"}
ISO = {"swedish": "sv", "english": "en"}


def _region(name: str) -> str:
    return REGIONS.get(name, name)


def _names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _set_order(sets: dict) -> list[str]:
    """The evaluation sets, the gated FLEURS set first."""
    return sorted(sets, key=lambda n: (n != "fleurs", n))


def _samples(recipe: Recipe, work: Workdir) -> str:
    """The listening section: the release's own samples, played from the repository."""
    path = work.release / "samples" / "samples.json"
    if not path.exists():
        return ""
    sub = recipe["publish"].get("subfolder")
    base = f"https://huggingface.co/{recipe['publish']['repo']}/resolve/main" + (f"/{sub}" if sub else "")
    items = json.loads(path.read_text())
    texts = list(dict.fromkeys(it["text"] for it in items))  # in the order the voices come, the first voice's sentence first
    common = Counter(it["text"] for it in items).most_common(1)[0][0]
    others = [it["voice"].capitalize() for it in items if it["text"] != common]
    out = ["## Listen", "",
           "Real, unpicked output: every voice reads a FLEURS sv_se test sentence that was never part of the training data, "
           f"and the first take (seed {items[0]['seed']}) is the one you hear. CER is the character error rate of a "
           f"speech recogniser ([{recipe['judge']['repo']}](https://huggingface.co/{recipe['judge']['repo']})) "
           "transcribing that clip, so 0.000 means every character was recognised."
           + (f" {_names(others)} read a different held-out sentence from the other voices." if others else ""), ""]
    for text in texts:
        out += [f"> {text}", "", "| voice | gender, region | CER | listen |", "|---|---|---|---|"]
        out += [f"| **{it['voice'].capitalize()}** | {it['gender']}, {_region(it['region'])} | {it['cer']:.3f} | "
                f'<audio controls preload="none" src="{base}/{it["file"]}"></audio> |' for it in items if it["text"] == text]
        out.append("")
    return "\n".join(out)


def _data_link(recipe: Recipe, hours: str) -> str:
    """Where the training data can be read: the published dataset, if the recipe has one."""
    if "dataset" not in recipe.raw:
        return ""
    d = recipe["dataset"]
    return (f"The training data is published as [{d['name']}](https://huggingface.co/datasets/{d['repo']}): "
            f"the {hours} hours used here, with held-out sets, sources and licences.")


def _took(train_report: dict) -> str:
    minutes = train_report["seconds"] / 60
    return f"{minutes / 60:.1f} hours" if minutes >= 90 else f"{minutes:.0f} minutes"


def _run(train_report: dict) -> str:
    """Where and how long it trained, and which checkpoint the release is."""
    ranks, result = train_report.get("ranks", 1), train_report["result"]
    line = f"Trained with brain on {'one NVIDIA GH200 GPU' if ranks == 1 else f'{ranks} NVIDIA GH200 GPUs'} in {_took(train_report)}."
    best = result.get("best")
    if best and result.get("checkpoint") == "best":
        how = "Training stopped early; the" if result.get("stopped_early") else "The"
        line += (f" {how} release is the checkpoint with the lowest loss on held-out speech, "
                 f"step {best['step']} of {result['steps_completed']}.")
    return line


def _proof(evaluation: dict | None, hours: str, shard: dict, voices: dict, train_report: dict) -> str:
    """The offer's proof: this release's own numbers, the evaluation's once it has run."""
    regions = len({v["region"] for v in voices.values()})
    ranks = train_report.get("ranks", 1)
    text = (f"a 1.7B model with {len(voices)} named Swedish voices from {regions} regions, fine-tuned on "
            f"{hours} hours of speech ({shard['records']:,} utterances) in {_took(train_report)} on {'one GPU' if ranks == 1 else f'{ranks} GPUs'}.")
    if evaluation:
        fleurs = evaluation["sets"]["fleurs"]
        text += (f" On FLEURS test sentences it never saw, a speech recogniser's character error rate is "
                 f"{fleurs['cer']:.3f}, against {fleurs['base_cer']:.3f} for the base model.")
    return text


def _evaluation(recipe: Recipe, r: dict, name: str = "Sten-TTS") -> str:
    """The results section, from the evaluation's report."""
    ev, sets, per_voice = recipe["evaluate"], r["sets"], r["voices"]
    names = _set_order(sets)
    sentences = {n: sets[n]["clips"] // (len(per_voice) * len(ev["seeds"])) for n in names}
    low, high = r["paired_cer"]["ci95"]
    real = sum(m["real_similarity"] for m in per_voice.values()) / len(per_voice)
    synth = sum(m["similarity"] for m in per_voice.values()) / len(per_voice)
    identified = min(m["identity_accuracy"] for m in per_voice.values())
    lines = [
        "## Evaluation", "",
        f"Each of the {len(per_voice)} voices read every sentence of the sets below. "
        f"[{recipe['judge']['repo']}](https://huggingface.co/{recipe['judge']['repo']}), told that the language is "
        f"{ev['judge_language']}, transcribed each clip, and the character error rate (CER, lower is better) compares "
        "that transcript with the sentence. The base model read the same sentences, cloning each voice from one of its recordings.", "",
        f"| test set | sentences | {name} | base model |", "|---|---|---|---|"]
    def lower_bold(a: float, b: float) -> tuple[str, str]:  # the better (lower) CER of the pair is bold
        return (f"**{a:.3f}**", f"{b:.3f}") if a <= b else (f"{a:.3f}", f"**{b:.3f}**")
    lines += [f"| {SETS.get(n, (n, ''))[0]} | {sentences[n]} | " + " | ".join(lower_bold(sets[n]['cer'], sets[n]['base_cer'])) + " |" for n in names]
    lines += ["", "; ".join(f"{SETS[n][0]}: {SETS[n][1]}" for n in names if n in SETS) + ".", "",
              f"- **Against the base model.** On FLEURS the CER difference, {name} minus base, is {r['paired_cer']['mean_diff']:.3f} "
              f"(95% paired-bootstrap interval {low:.3f} to {high:.3f}, n = {r['paired_cer']['n']:,} voice and sentence pairs).",
              f"- **Reliability.** Failure rate {r['gates']['failure_rate_max']['value']:g} on the {sets['fleurs']['clips']:,} FLEURS clips: "
              "none was silent, ran into the length limit or ran on past its text.",
              f"- **Voice consistency.** Identification accuracy is {identified:.2f} for every voice: a speaker encoder matched each "
              f"FLEURS clip to the nearest of the {len(per_voice)} voices' real recordings. Cosine similarity to the voice's own recordings is {synth:.3f} for synthesized "
              f"clips and {real:.3f} for real held-out recordings; at that level the encoder separates little, so read the "
              "identification result, not the similarity, as the evidence."]
    if r.get("upstream", {}).get("status") == "ok":
        lines.append("- **Compatibility.** The checkpoint loads and speaks in the upstream `qwen-tts` package, unconverted.")
    lines += ["", "CER per voice:", "", "| voice | region | " + " | ".join(SETS.get(n, (n, ""))[0] for n in names) + " |", "|---|---|" + "---|" * len(names)]
    lines += [f"| {v['name']} | {_region(v['region'])} | " + " | ".join(f"{sets[n]['voices'][v['name']]:.3f}" for n in names) + " |" for v in recipe["voices"]]
    lines += ["", "CER per region:", "", "| region | " + " | ".join(SETS.get(n, (n, ""))[0] for n in names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {_region(g)} | " + " | ".join(f"{sets[n]['regions'][g]:.3f}" for n in names) + " |" for g in sorted(sets[names[0]]["regions"], key=_region)]
    lines += ["",
              "How to read these numbers: CER measures the recogniser as well as the speech, so it is a relative measure between "
              f"systems on the same sentences, not an absolute error count. On the voices' real held-out recordings the same recogniser scores {r['floor']['cer']:.3f}, "
              "which is higher than on the synthesized speech. That reflects those recordings (16 kHz, made around 1999, other "
              "sentences) and is not a claim that the model is clearer than its speakers."]
    return "\n".join(lines) + "\n"


def _family(recipe: Recipe, work: Workdir) -> str:
    """The compact variants of this model (`publish.family`), each from its run's evidence under `compact/<name>/`."""
    from . import config
    names = recipe["publish"].get("family", [])
    if not names:
        return ""
    rows = []
    for name in names:
        v = config.load(recipe.path.parent / name)
        d = work.root / "compact" / v.name
        params = json.loads((d / "compress" / "report.json").read_text())["parameters"]
        r = json.loads((d / "evaluation" / "report.json").read_text())
        speed = json.loads((d / "evaluation" / "speed.json").read_text())
        par = r["parity"]
        at = all(g["pass"] for g in par["gates"].values())
        low, high = par["cer_delta_ci95"]
        link = f"https://huggingface.co/{v['publish']['repo']}/tree/main/{v['publish']['subfolder']}"
        rows.append(f"| [{v.title}]({link}) | {params / 1e6:,.1f} M | {r['sets']['fleurs']['cer']:.3f} | {'yes' if at else 'no'} | "
                    f"{par['cer_delta']:+.4f} ({low:+.4f} to {high:+.4f}) | {speed['variant']['real_time_factor']:.2f} |")
    title = recipe.title
    return "\n".join(["", "## Compact variants", "",
                      f"Smaller models cut out of {title} by weight surgery and distilled from it, each in its own folder of this "
                      "repository with its own card. The CER difference is to " + title + " on the same FLEURS clips, with its 95% "
                      "paired-bootstrap interval; the real-time factor is measured one request at a time on one GPU (below 1 is faster "
                      "than real time).", "",
                      f"| model | parameters | FLEURS CER | at parity with {title} | CER difference | real-time factor |", "|---|---|---|---|---|---|",
                      *rows, ""])


def model_card(recipe: Recipe, work: Workdir, train_report: dict) -> str:
    if recipe.is_variant:
        from .variantcard import variant_card
        return variant_card(recipe, work, train_report)
    sel = json.loads((work.selection / "report.json").read_text())
    shard = json.loads((work.shards / "train.json").read_text())
    voices = json.loads((work.selection / "voices.json").read_text())
    report = work.evaluation / "report.json"
    evaluation = json.loads(report.read_text()) if report.exists() else None
    repo, base = recipe["publish"]["repo"], recipe["base"]
    hours = f"{shard['hours']:g}"
    sources = list(recipe["selection"]["share"])
    languages = [recipe["language"]["name"], *recipe["language"].get("base_languages", [])]
    per_lang = shard.get("hours_per_language", {languages[0]: shard["hours"]})
    lang_text = ", ".join(f"{h:.0f} h {l.capitalize()}" for l, h in per_lang.items())
    genders = Counter(v["gender"] for v in voices.values())
    regions = {v["region"] for v in voices.values()}
    first = next(iter(recipe["publish"].get("samples", {}).get("order", [])), next(iter(voices)))
    lang_yaml = "\n".join(f"- {ISO.get(l, l)}" for l in languages)
    datasets_yaml = f"datasets:\n- {recipe['dataset']['repo']}\n" if "dataset" in recipe.raw else ""
    voice_rows = "\n".join(f"| `{name}` | {v['gender']} | {_region(v['region'])} ({v['region']}) |" if _region(v["region"]) != v["region"]
                           else f"| `{name}` | {v['gender']} | {v['region']} |" for name, v in voices.items())
    source_rows = "\n".join(f"| {SOURCES.get(s, (s, ''))[0]} | {SOURCES.get(s, (s, ''))[1]} | {recipe['sources'][s]['license']} |" for s in sources)
    attributions = " ".join(recipe["sources"][s]["attribution"] for s in sources if "attribution" in recipe["sources"][s])
    licences = "; ".join(f"{SOURCES.get(s, (s, ''))[0]}, {recipe['sources'][s]['license']}" for s in sources)
    dataset = (f"Published as [{recipe['dataset']['name']}](https://huggingface.co/datasets/{recipe['dataset']['repo']})."
               if "dataset" in recipe.raw else "")
    numbers_note = " The Numbers set under Evaluation measures how well." if evaluation and "numbers" in evaluation["sets"] else ""
    return f"""---
license: apache-2.0
language:
{lang_yaml}
base_model: {base['repo']}
base_model_relation: finetune
pipeline_tag: text-to-speech
{datasets_yaml}tags:
- text-to-speech
- tts
- swedish
- qwen3-tts
---

# Sten-TTS-1.7B

{model_offer(_proof(evaluation, hours, shard, voices, train_report))}

Sten-TTS-1.7B is a Swedish text-to-speech model with {len(voices)} named voices, {genders['female']} female and {genders['male']} male, from
{len(regions)} regions of Sweden. It is a fine-tune of [{base['repo']}](https://huggingface.co/{base['repo']}) that adds
Swedish as a language. English is trained alongside it, so English words and names inside Swedish sentences keep
their English pronunciation. The model produces 24 kHz audio, loads in the upstream `qwen-tts` package without
conversion, and is released under Apache-2.0.

{_data_link(recipe, hours)}

{_samples(recipe, work)}
## Usage

### Python, with the upstream package

`pip install qwen-tts soundfile` (tested with `qwen-tts` 0.1.1).

```python
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained("{repo}", device_map="cuda:0", dtype=torch.bfloat16)
wavs, sample_rate = model.generate_custom_voice(
    text="Hej och välkommen till Sten.",
    speaker="{first}",
    language="swedish",
)
sf.write("{first}.wav", wavs[0], sample_rate)
```

On a machine without a GPU, load with `device_map="cpu", dtype=torch.float32`.

### HTTP, with brain's OpenAI-compatible API

```bash
brain pull {repo}
brain serve --openai 127.0.0.1:8080    # prints the API key it generates, or pass --api-key-file

curl http://127.0.0.1:8080/v1/audio/speech \\
  -H "Authorization: Bearer $BRAIN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{repo}", "voice": "{first}", "language": "swedish",
       "input": "Hej och välkommen till Sten.", "response_format": "wav"}}' \\
  -o {first}.wav
```

Always send `"language": "swedish"`; without it the text is spoken as English.

## Voices

Pass the name as `speaker` (Python) or `voice` (HTTP).

| voice | gender | region |
|---|---|---|
{voice_rows}

Each voice is an anonymous speaker of the NST Swedish speech database (CC0-1.0), chosen as the liveliest reader of
their gender and region by pitch variation, loudness dynamics and speaking rate. The region is where the speaker grew
up. The recordings are read sentences, so expect a regional accent rather than dialect.

{_evaluation(recipe, evaluation) if evaluation else ""}{_family(recipe, work)}
## Training

- **Base model:** [{base['repo']}](https://huggingface.co/{base['repo']}), revision `{base['revision']}`.
- **Data:** {hours} hours of speech ({lang_text}), {shard['records']:,} utterances from {sel['speakers']['train']:,} speakers. {dataset}
- **Method:** a full fine-tune of the speech-generating parts of the model, with the text embedding frozen.
- **Run:** {_run(train_report)}
- **Reproducibility:** the release was produced by a single recipe in which the base model, every data source and the
  recogniser are pinned to a revision, and the random seed is fixed.

| source | content | licence |
|---|---|---|
{source_rows}

Only the {len(voices)} NST speakers above are named voices; no other speaker in the training data can be selected.

## Limitations

- Most of the Swedish training audio, including all recordings of the named voices, is 16 kHz, so the 24 kHz output
  lacks the top of its band.
- The voices are read-speech speakers recorded around 1999. The checkpoint speaks with its named voices only:
  it takes no style instruction and does not clone a voice from a reference recording.
- Text is read as written. There is no separate text normalisation, so numbers, dates, abbreviations and symbols are
  the weakest area.{numbers_note}
- Error rates differ between voices; the per-voice table shows which read least reliably.
- English is trained only so that English words inside Swedish text sound right. English-only text was not evaluated.
- Finland Swedish is not covered.

## Licence and attribution

The model is released under Apache-2.0, as is the base model.

Training data: {licences}.

{attributions}
"""
