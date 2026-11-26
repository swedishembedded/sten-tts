# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements compression of speech models to the hardware
# a product has, without audible loss, for its clients. If your team needs
# expertise in shrinking text-to-speech models, you can procure our services by
# sending an email to info@swedishembedded.com.

"""The model card of a compact variant, written from its run's own reports.

It follows the full model's card, for the same reader: what the model is, how
it sounds, how to load it (from its subfolder of the full model's repository),
how well it reads. It adds what a smaller model owes its user: its size beside
the full model's, the speed measured on one card, the parity table against
the full model on the very same clips, and how it was cut and recovered (every
edit of the surgery manifest with its measured cost before recovery).
"""

from __future__ import annotations

import json
from collections import Counter

from .compress import parameters
from .config import Recipe, Workdir
from .modelcard import ISO, SETS, _evaluation, _names, _region, _run, _samples, _took
from .offer import model_offer

# A stack or stream of the checkpoint, as the card names it.
PARTS = {"talker": "Talker", "code_predictor": "code predictor", "text": "text"}


def _m(n: float) -> str:
    return f"{n / 1e6:,.1f} M"


def _part(target: str) -> str:
    return PARTS.get(target.split(".")[0], target)


def _layers(numbers: list[int]) -> str:
    items = [str(n) for n in numbers]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def describe(change: dict, method: str, vocabulary: dict) -> str:
    """One surgery edit in words, from its manifest entry (what it changed in the config, and what it kept)."""
    kind, config = change["edit"], change.get("config", {})
    (key, (before, after)), *_ = config.items() if config else [("", (0, 0))]
    part = _part(change.get("targets", [""])[0])
    greedy = method.startswith("greedy_")
    how = {"fisher": "Fisher", "taylor": "group Taylor", "act_rms": "activation RMS", "wanda": "Wanda", "kl": "KL divergence",
           "loss": "loss", "cosine": "stream cosine"}.get(method.replace("greedy_", "").replace("ablation_", ""), method)
    if kind == "trim_vocabulary":
        return (f"Trim the text vocabulary to {vocabulary['kept']:,} of {vocabulary['of']:,} tokens: every token of the "
                "training texts and of the other candidate utterances of its sources, and the tokens they are merged from")
    if kind == "select_channels":
        unit = "attention heads" if "head" in key else "channels"
        return f"Keep {after:,} of {before:,} {unit} in every {part} {'MLP' if unit == 'channels' else 'attention'}, ranked by {how} score"
    if kind == "remove_layers":
        stack = next(iter(change["kept"]))
        removed = sorted(set(range(before)) - set(change["kept"][stack]))
        chosen = "chosen one at a time, each measured with the earlier ones removed" if greedy else "the ones it misses least alone"
        return f"Remove {before - after} of {before} {part} layers ({_layers(removed)}), {chosen}, by {how}"
    if kind == "rotate_stream":
        return f"Rotate the {part} residual stream into its principal directions and keep {after:,} of {before:,}"
    if kind == "fold_embedding":
        return f"Fold the text embedding into its projection, keeping {after:,} of {before:,} channels"
    return f"{kind} ({method})"


def _hidden(distill: dict) -> str:
    """The hidden-state term of the recovery, in words, when it has one."""
    if not distill.get("hidden_kd_weight"):
        return ""
    pairs = ", ".join(f"{s} to {t}" for s, t in distill["hidden_kd_layers"])
    return (f", and a mean squared error (weight {distill['hidden_kd_weight']:g}) between residual streams of this model and "
            f"the full model through a learned projection (its layers {pairs}; the projection is not part of the release)")


def _cut(recipe: Recipe, compress: dict) -> str:
    out = compress["result"]
    manifest = out["manifest"]["manifest"]["edits"]  # the job reports surgery.json: {"manifest", "outcome"}
    source = recipe["compress"].get("from", {}).get("title", "the full model")
    rows = ["| edit | parameters after | loss | KL to the full model |", "|---|---|---|---|",
            f"| {source} | {_m(out['params_before'])} | {out['baseline']['loss']:.3f} | {out['baseline'].get('kl', 0.0):.3f} |"]
    for entry, measured in zip(manifest, out["edits"]):
        rows.append(f"| {describe(entry['change'], entry['method'], out.get('vocabulary', {}))} | {_m(measured['params_after'])} | "
                    f"{measured['loss']:.3f} | {measured['kl']:.3f} |")
    return "\n".join(rows)


def _short_of(full: str, r: dict) -> str:
    """The plain statement of a variant that misses parity: which checks it fails, with the CER gap and its interval."""
    par = r["parity"]
    failed = [k for k, g in par["gates"].items() if not g["pass"]]
    low, high = par["cer_delta_ci95"]
    text = f"It is not at parity with {full}"
    if "cer_delta_max" in failed:
        text += (f": its FLEURS CER is {par['cer_delta']:.4f} higher (95% paired-bootstrap interval {low:.4f} to {high:.4f}), "
                 f"more than the {par['gates']['cer_delta_max']['threshold']:g} parity allows")
    others = {"similarity_drop_max": "voice similarity", "identity_drop_max": "voice identification", "failure_delta_max": "reliability"}
    rest = [others[k] for k in failed if k in others]
    if rest:
        text += (" and" if "cer_delta_max" in failed else ":") + " it falls short in " + _names(rest)
    return text + "."


def _parity(recipe: Recipe, r: dict, teacher_sets: dict | None) -> str:
    par, full = r["parity"], recipe["teacher"]["title"]
    low, high = par["cer_delta_ci95"]
    lines = ([_short_of(full, r), ""] if not all(g["pass"] for g in par["gates"].values()) else []) + [f"Both models read the same FLEURS test sentences with the same voices and seeds ({par['clips']:,} clips each), "
             "and the same recogniser and speaker encoder judged both, so the comparison is clip for clip.", "",
             f"- **Intelligibility.** FLEURS CER is {r['sets']['fleurs']['cer']:.3f} against {full}'s "
             f"{r['sets']['fleurs']['cer'] - par['cer_delta']:.3f}: a difference of {par['cer_delta']:+.4f} "
             f"(95% paired-bootstrap interval {low:+.4f} to {high:+.4f}).",
             f"- **Reliability.** The failure rate differs by {par['failure_delta']:+.4f}."]
    if teacher_sets:
        names = [n for n in ("fleurs", "code_switching", "numbers") if n in r["sets"] and n in teacher_sets]
        lines.append("- **Every set.** Mean CER of this model against " + full + "'s: "
                     + "; ".join(f"{SETS.get(n, (n, ''))[0]} {r['sets'][n]['cer']:.3f} against {teacher_sets[n]['cer']:.3f}" for n in names) + ".")
    lines += ["",
             f"| voice | region | CER | {full} CER | similarity drop | identification drop |", "|---|---|---|---|---|---|"]
    region = {v["name"]: v["region"] for v in recipe["voices"]}
    lines += [f"| {v} | {_region(region[v])} | {m['cer']:.3f} | {m['teacher_cer']:.3f} | {m['similarity_drop']:+.4f} | {m['identity_drop']:+.3f} |"
              for v, m in par["voices"].items()]
    lines += ["", "Similarity drop is the full model's mean cosine similarity to the voice's real recordings minus this model's; "
              "identification drop is the same for the share of clips matched to the right voice. Positive means this model is lower."]
    return "\n".join(lines)


def _speed(recipe: Recipe, speed: dict, own: int, full_params: int, codec: int) -> str:
    full, title = recipe["teacher"]["title"], recipe.title
    v, t = speed["variant"], speed["teacher"]
    return "\n".join([
        f"| | {title} | {full} |", "|---|---|---|",
        f"| parameters (model.safetensors) | {_m(own)} | {_m(full_params)} |",
        f"| speech codec (speech_tokenizer) | {_m(codec)}, shared | {_m(codec)}, shared |",
        f"| real-time factor | {v['real_time_factor']:.2f} | {t['real_time_factor']:.2f} |",
        f"| codec frames per second | {v['frames_per_second']:.1f} | {t['frames_per_second']:.1f} |", "",
        f"Measured on one {speed['device']} with brain, one request at a time through its API (wall time per request, as "
        f"a client sees it), {speed['sentences']} FLEURS sentences read by {speed['voice']} after a warm-up. The real-time factor is "
        "the time taken over the length of the audio, so below 1 is faster than real time; a frame is 80 ms of audio "
        f"(12.5 frames a second). {title} speaks {speed['speedup']:.2f} times as fast as {full} here.", "",
        f"The speech codec ({_m(codec)} parameters, in `speech_tokenizer/`) turns codec frames into audio. It is the same "
        "codec in both models, untouched by the compression and counted separately from the parameters above."])


def variant_card(recipe: Recipe, work: Workdir, train_report: dict) -> str:
    shard = json.loads((work.shards / "train.json").read_text())
    voices = json.loads((work.selection / "voices.json").read_text())
    compress = json.loads((work.compress / "report.json").read_text())
    report = work.evaluation / "report.json"
    evaluation = json.loads(report.read_text()) if report.exists() else None
    speed_path = work.evaluation / "speed.json"
    speed = json.loads(speed_path.read_text()) if speed_path.exists() else None
    t, title, full = recipe["teacher"], recipe.title, recipe["teacher"]["title"]
    # brain serves a model-store folder under its path, so the downloaded subfolder is served under its own name.
    repo, sub = recipe["publish"]["repo"], recipe["publish"]["subfolder"]
    model_id = f"local/{sub}"
    own = parameters(work.release / "model.safetensors")
    codec = parameters(work.release / "speech_tokenizer" / "model.safetensors")
    full_params = t["parameters"]
    source = recipe["compress"].get("from")
    frac = own / full_params
    size = "half the size" if 0.4 < frac < 0.6 else "a quarter of the size" if 0.2 < frac < 0.3 else f"{frac:.0%} of the size"
    origin = (f"It was cut out of {source['title']}, the previous compact model, itself cut from {full}, by removing the parts it "
              "needs least, measured on its own speech against " + full + "," if source else
              f"It was cut out of {full} by removing the parts it needs least, measured on its own speech,")
    hours = f"{shard['hours']:g}"
    languages = [recipe["language"]["name"], *recipe["language"].get("base_languages", [])]
    lang_yaml = "\n".join(f"- {ISO.get(l, l)}" for l in languages)
    datasets_yaml = f"datasets:\n- {recipe['dataset']['repo']}\n" if "dataset" in recipe.raw else ""
    genders = Counter(v["gender"] for v in voices.values())
    first = next(iter(recipe["publish"].get("samples", {}).get("order", [])), next(iter(voices)))
    voice_rows = "\n".join(f"| `{name}` | {v['gender']} | {_region(v['region'])} |" for name, v in voices.items())
    result = train_report["result"]
    distill = recipe["train"]["distill"]
    passed = evaluation is not None and all(g["pass"] for g in evaluation["parity"]["gates"].values())
    proof = (f"{title}, {_m(own)} parameters cut out of the {_m(full_params)} {full} by weight surgery and recovered by "
             f"distillation in {_took(train_report)} on {train_report.get('ranks', 1)} GPUs.")
    if evaluation:
        proof += (f" On FLEURS test sentences its character error rate is {evaluation['sets']['fleurs']['cer']:.3f}, "
                  f"{evaluation['parity']['cer_delta']:+.4f} from {full}'s on the same clips.")
    if speed:
        proof += f" It speaks {speed['speedup']:.2f} times as fast on one GH200."
    teacher_report = work.root / recipe["evaluate"]["parity"]["teacher_evaluation"] / "report.json"
    teacher_sets = json.loads(teacher_report.read_text()).get("sets") if teacher_report.exists() else None
    # Sets beyond the gated FLEURS one where this model reads clearly worse than the full model: said, not hidden.
    margin = recipe["evaluate"]["parity"]["cer_max_delta"]
    weaker = [n for n in (evaluation or {}).get("sets", {}) if n != "fleurs" and teacher_sets and n in teacher_sets
              and evaluation["sets"][n]["cer"] - teacher_sets[n]["cer"] > margin]
    weaker_text = (" It reads " + _names([SETS.get(n, (n, ""))[0].lower() for n in weaker]) + " sentences less clearly than "
                   f"{full} (CER " + ", ".join(f"{evaluation['sets'][n]['cer']:.3f} against {teacher_sets[n]['cer']:.3f}" for n in weaker) + ").") if weaker else ""
    standing = (f"On FLEURS test sentences it is at parity with {full}: no worse in intelligibility, voice similarity, voice "
                "identification or reliability than the margins in the parity table below." + weaker_text if passed else
                _short_of(full, evaluation) + " The parity table below shows where." if evaluation else "")
    sections = [f"""---
license: apache-2.0
language:
{lang_yaml}
base_model: {t['repo']}
base_model_relation: finetune
pipeline_tag: text-to-speech
{datasets_yaml}tags:
- text-to-speech
- tts
- swedish
- qwen3-tts
- distillation
- pruning
---

# {title}

{model_offer(proof)}

{title} is [{full}](https://huggingface.co/{t['repo']}) at {size}: {_m(own)} parameters against {_m(full_params)},
with the same {len(voices)} named Swedish voices ({genders['female']} female, {genders['male']} male), the same languages and
the same 24 kHz output. {origin}
and then trained to predict what {full} predicts on the same {hours} hours of speech. {standing} It loads in the upstream
`qwen-tts` package without conversion and lives in the `{sub}/` folder of the {full} repository.
The speech codec ({_m(codec)} parameters) is shared with {full} and counted separately.
""", _samples(recipe, work), f"""## Usage

### Python, with the upstream package

`pip install qwen-tts soundfile huggingface_hub` (tested with `qwen-tts` 0.1.1). Download only this model's folder and load it
from there:

```python
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

path = snapshot_download("{repo}", allow_patterns="{sub}/*")
model = Qwen3TTSModel.from_pretrained(f"{{path}}/{sub}", device_map="cuda:0", dtype=torch.bfloat16)
wavs, sample_rate = model.generate_custom_voice(
    text="Hej och välkommen till Sten.",
    speaker="{first}",
    language="swedish",
)
sf.write("{first}.wav", wavs[0], sample_rate)
```

On a machine without a GPU, load with `device_map="cpu", dtype=torch.float32`.

### HTTP, with brain's OpenAI-compatible API

Put the folder into brain's model store, where it is served as `{model_id}`:

```bash
hf download {repo} --include "{sub}/*" --local-dir sten-tts
mkdir -p ~/.local/share/brain/models/local && mv sten-tts/{sub} ~/.local/share/brain/models/{model_id}
brain serve --openai 127.0.0.1:8080    # prints the API key it generates, or pass --api-key-file

curl http://127.0.0.1:8080/v1/audio/speech \\
  -H "Authorization: Bearer $BRAIN_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{model_id}", "voice": "{first}", "language": "swedish",
       "input": "Hej och välkommen till Sten.", "response_format": "wav"}}' \\
  -o {first}.wav
```

Always send `"language": "swedish"`; without it the text is spoken as English.

## Voices

The voices of {full}; pass the name as `speaker` (Python) or `voice` (HTTP).

| voice | gender | region |
|---|---|---|
{voice_rows}
"""]
    if speed:
        sections.append("## Size and speed\n\n" + _speed(recipe, speed, own, full_params, codec) + "\n")
    if evaluation:
        sections.append(f"## Parity with {full}\n\n" + _parity(recipe, evaluation, teacher_sets) + "\n")
        sections.append(_evaluation(recipe, evaluation, title))
    best = result.get("best") or {}
    sections.append(f"""## How it was made

**The cut.** Weight surgery on {source['title'] if source else full}, calibrated on {compress['request']['records']} training utterances and measured
on {compress['request']['held_out_records']} held-out ones. Each edit keeps the upstream layout (fewer layers, narrower
dimensions, a smaller vocabulary), so the result is an ordinary checkpoint. Loss and KL divergence from {full}'s
predictions are measured after each edit, before any training:

{_cut(recipe, compress)}

**The recovery.** Distillation from {full} on its {hours} hours of training speech ({shard['records']:,} utterances): the loss is the
ground-truth cross-entropy (weight {distill['gt_weight']:g}) plus the KL divergence from {full}'s predictions for the first
codebook (weight {distill['talker_kd_weight']:g}) and the 15 residual codebooks (weight {distill['mtp_kd_weight']:g}), at
temperature {distill['kd_temperature']:g}{_hidden(distill)}, with learning rate {train_report['request']['lr']:g}. {_run(train_report)}
{f"On held-out speech its distillation loss (the KL terms to {full}{' and the hidden-state error' if distill.get('hidden_kd_weight') else ''}) is {best['held_out_loss']:.4f}." if "held_out_loss" in best else ""}

## Limitations

- Everything said about {full}'s limits holds here: 16 kHz training audio for most Swedish speech, read-speech voices
  recorded around 1999, no style instructions or voice cloning, text read as written (numbers and abbreviations are the
  weakest area), English trained only for English words inside Swedish text, no Finland Swedish.
- A smaller model keeps less: compare the per-voice parity table before choosing it over {full} for a particular voice.{chr(10) + "- " + weaker_text.strip() + " Prefer " + full + " where that matters." if weaker else ""}

## Licence and attribution

The model is released under Apache-2.0, as is {full}, and was trained on the same data; see the
[{full} model card](https://huggingface.co/{t['repo']}) for the sources and their attributions.
""")
    return "\n".join(sections)
