# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for publishing curated, licence-clean
# speech datasets for its clients. If your team needs expertise in dataset
# licensing, attribution and documentation, you can procure our services by
# sending an email to info@swedishembedded.com.

"""The dataset card (README.md) and terms (TERMS.md) of the exported training collection.

Both are written from the recipe (sources, revisions, licences, attributions)
and the export's own manifest (rows, hours, speakers), so every number in the
card is the export's.
"""

from __future__ import annotations

from .config import Recipe

# What each source is, for the card; licence, revision and attribution come from the recipe.
ABOUT = {
    "nst": ("NST Swedish ASR Database", "read sentences, about 900 recording sessions from ten Swedish dialect regions, recorded around 1999 at 16 kHz"),
    "rixvox": ("RixVox-v2", "debates of the Swedish Parliament (Riksdagen) at 16 kHz, aligned to the protocols; the 100 most recent shards"),
    "librivox": ("LibriVox Swedish (5-minute chunks)", "public-domain Swedish audiobooks at 16 kHz, cut into utterances and aligned by this recipe"),
    "fleurs": ("FLEURS sv_se", "read Wikipedia sentences at 16 kHz; train and dev only, the test split is the evaluation set"),
    "libritts_r": ("LibriTTS-R train-clean-100", "English audiobook speech, restored, at 24 kHz"),
}
HF_LICENSE = {"CC0-1.0": "cc0-1.0", "CC-BY-4.0": "cc-by-4.0", "ODC-BY-1.0": "odc-by"}

LIBRITTS_R_CITATION = '''@ARTICLE{Koizumi2023-hs,
  title         = "{LibriTTS-R}: A restored multi-speaker text-to-speech corpus",
  author        = "Koizumi, Yuma and Zen, Heiga and Karita, Shigeki and Ding,
                   Yifan and Yatabe, Kohei and Morioka, Nobuyuki and Bacchiani,
                   Michiel and Zhang, Yu and Han, Wei and Bapna, Ankur",
  month         =  may,
  year          =  2023,
  archivePrefix = "arXiv",
  primaryClass  = "eess.AS",
  eprint        = "2305.18802"
}'''
FLEURS_CITATION = '''@article{fleurs2022arxiv,
  title = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal={arXiv preprint arXiv:2205.12446},
  url = {https://arxiv.org/abs/2205.12446},
  year = {2022}
}'''


def terms(recipe: Recipe) -> str:
    name = recipe["dataset"]["name"]
    lic = ", ".join(f"{ABOUT[s][0]} {recipe['sources'][s]['license']}" for s in recipe["selection"]["share"])
    return f"""# {name}: terms

1. **Our contribution.** The selection, cleaning, segmentation and alignment, the transcripts this
   work produced (the aligned LibriVox spans), the metadata, the voice assignments and the dataset
   card are © 2026 Swedish Embedded AB, licensed under
   [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
2. **The source material.** Each clip and its source transcript stay under the licence of the
   source they come from, named on every row (`source`, `source_license`, `source_revision`):
   {lic}. Nothing in these terms restricts any right those licences grant.
3. **Attribution.** Redistributing the dataset, or publishing a model trained on it, credits
   "{name} (Swedish Embedded AB)" and carries the attributions the sources require; the dataset
   card gives the RixVox-v2 notice and the LibriTTS-R and FLEURS citations verbatim.
4. **Responsible use.** Swedish Embedded AB's policy for its contribution: do not use the dataset
   to imitate or impersonate an identifiable person (the RixVox-v2 speakers are public persons),
   and disclose synthetic speech wherever listeners could take it for a real person. This binds
   the use of our contribution; for the source material it is a request made alongside the source
   licences, not a restriction of them.
5. **No warranty.** The dataset is provided as is, without warranty of any kind. Users are
   responsible for complying with applicable law, including privacy and personality rights.
"""


def _hours_table(stats: dict, key: str, label: str) -> str:
    splits = [s for s in stats if stats[s]["hours"].get(key)]
    names = sorted({n for s in splits for n in stats[s]["hours"][key]})
    lines = [f"| {label} | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
    lines += [f"| {n} | " + " | ".join(f"{stats[s]['hours'][key].get(n, 0):.2f}" for s in splits) + " |" for n in names]
    return "\n".join(lines)


def card(recipe: Recipe, manifest: dict, voices: dict, segment_report: dict | None) -> str:
    spec, sel = recipe["dataset"], recipe["selection"]
    sources = list(sel["share"])
    licences = sorted({HF_LICENSE.get(recipe["sources"][s]["license"], recipe["sources"][s]["license"].lower()) for s in sources})
    splits = [s for s in manifest["splits"] if manifest["splits"][s]["rows"]]
    configs = "\n".join(f"  - split: {s}\n    path: data/{s}-*" for s in splits)
    stats = manifest["stats"]
    src_rows = "\n".join(
        f"| [{ABOUT[s][0]}](https://huggingface.co/datasets/{recipe['sources'][s]['repo']}) | `{recipe['sources'][s]['revision'][:12]}` | "
        f"{recipe['sources'][s]['license']} | {ABOUT[s][1]} | {stats['train']['hours']['source'].get(s, 0):.1f} |" for s in sources)
    split_rows = "\n".join(f"| {s} | {manifest['splits'][s]['rows']} | {manifest['splits'][s]['hours']:.1f} |" for s in splits)
    speakers = "\n".join(f"| {s} | " + " | ".join(str(stats[sp]["speakers"].get(s, 0)) for sp in splits) + " |" for s in sources)
    voice_rows = "\n".join(f"| {n} | {v['gender']} | {v['region']} | `{v['speaker']}` | {v.get('score', float('nan')):.2f} | "
                           f"{v['measures']['f0_range']:.1f} | {v['measures']['f0_sd']:.1f} | {v['measures']['energy_sd']:.1f} | {v['measures']['syllable_rate']:.2f} |"
                           for n, v in voices.items() if "measures" in v)
    seg = ""
    if segment_report:
        kept = segment_report["outcome"].get("kept", {"segments": 0, "hours": 0})
        rejected = ", ".join(f"{k} {v['hours']:.1f} h" for k, v in segment_report["outcome"].items() if k != "kept")
        seg = (f" Of {segment_report['source_hours']:.1f} h of chunks, {kept['hours']:.1f} h ({kept['segments']} segments) were kept; "
               f"rejected: {rejected}.")
    rix = recipe["sources"]["rixvox"].get("attribution", "")
    seg_cfg = recipe["segment"]
    held = sel["held_out"]
    return f"""---
license: other
license_name: sten-tts-voice-terms
license_link: https://huggingface.co/datasets/{recipe['dataset']['repo']}/blob/main/TERMS.md
language:
- sv
- en
task_categories:
- text-to-speech
pretty_name: {spec['name']}
tags:
- speech
- swedish
- text-to-speech
- code-switching
source_datasets:
{chr(10).join(f"- {recipe['sources'][s]['repo']}" for s in sources)}
configs:
- config_name: default
  data_files:
{configs}
---

# {spec['name']}

The training collection of Sten-TTS-1.7B v2, a Swedish text-to-speech model with twelve named voices:
{manifest['splits']['train']['hours']:.0f} hours of Swedish and English speech from five open sources, cleaned,
trimmed and transcribed, with the held-out sets it was evaluated on. Every row names its source, the
source's licence and the pinned revision it came from. Licences per source:
{", ".join(licences)}; our own contribution is CC-BY-4.0 (see [Licence and terms](#licence-and-terms)).

Built by the open `sten-tts` recipe (recipe `{recipe.name}`, seed {recipe.seed}); every step below is a
deterministic stage of it.

## Use

```python
from datasets import load_dataset

ds = load_dataset("{spec['repo']}", split="train")
row = ds[0]
row["audio"]["array"], row["audio"]["sampling_rate"], row["text"], row["language"], row["voice"]
```

Columns: `id`, `audio` (FLAC, at the source's own rate), `text` (the training transcript), `language`
(`swedish` or `english`), `speaker`, `gender`, `region` (NST: dialect region of upbringing; RixVox:
electoral district), `source`, `source_license`, `source_revision` (`repo@commit`), `split`, `voice`
(the named voice a row belongs to, else empty), `duration` (seconds).

| split | rows | hours |
|---|---|---|
{split_rows}

- `train`: what the model was trained on (cut at exactly {sel['hours']:.0f} h).
- `validation`: {held['validation_fraction']:.0%} of the training speakers' utterances, for checkpoint choice.
- `voice_test`: {held['voice_test_utterances']} recordings per named voice whose texts appear nowhere in training.
- `unseen_test`: every recording of whole speakers kept out of training ({", ".join(f"{n} {s}" for s, n in held['unseen_speakers'].items())}).

## Sources

| source | revision | licence | what | train hours |
|---|---|---|---|---|
{src_rows}

Speakers per split:

| source | {" | ".join(splits)} |
|---|{"---|" * len(splits)}
{speakers}

FLEURS does not identify its readers, so each FLEURS recording counts as its own speaker. A LibriVox
"speaker" is the reader of one audiobook (four parts of one novel may share a reader). LibriTTS-R
carries no gender here (`unspecified`).

## Hours

{_hours_table(stats, "language", "language")}

{_hours_table(stats, "gender", "gender")}

NST hours by dialect region of upbringing:

{_hours_table(stats, "region", "region")}

## How it was made

1. **Fetch** exactly the files the recipe names, at pinned revisions (sizes checked).
2. **Segment LibriVox.** Each 5-minute chunk is cut at pauses (20 ms frames more than
   {seg_cfg['silence_db']:.0f} dB below the chunk's 95th-percentile level for at least {seg_cfg['min_pause_seconds']} s) into
   {seg_cfg['min_seconds']:.0f}-{seg_cfg['max_seconds']:.0f} s segments, each ending mid-way in the longest pause of its window. Qwen3-ASR-1.7B
   (served by brain, told the language) transcribes every segment; the chunk's recognised words are aligned in
   order to its known text (word-level edit distance over the normalised strings), and a segment is kept only when
   its recognition matches its aligned span within CER {seg_cfg['max_cer']}. Its transcript is the span as the text prints
   it, never the recognition; segments next to reference words neither neighbour recognised are dropped.{seg}
3. **Eligibility.** NST sentences of at least {sel['nst']['min_words']} words, without letter-by-letter spelling items,
   keyed by recording session (NST reuses speaker ids); RixVox segments whose protocol and Whisper transcript agree
   (WER at most {sel['rixvox']['max_wer_whisper']}, Swedish with probability at least {sel['rixvox']['min_lang_prob_sv']}), with the verbatim
   Whisper transcript; durations {sel['duration_seconds']['min']}-{sel['duration_seconds']['max']} s.
4. **Voices.** Each of the twelve named voices is an anonymous NST speaker, never a RixVox speaker: the
   liveliest-sounding speaker of its gender and region with at least {recipe['liveliness']['min_minutes']:.0f} minutes of
   eligible audio. Liveliness is a weighted sum of z-scores, among the candidates of the same gender, of F0
   range (10th-90th percentile, semitones), F0 variability, loudness dynamics over voiced frames and syllables
   a second, measured on {recipe['liveliness']['utterances']} of the speaker's own recordings (`voices.json` has the
   chosen speakers' and the runners-up's measures).
5. **Held out and banned texts.** Whole speakers per source (drawn among those with at most the median audio),
   {held['voice_test_utterances']} unseen-text recordings per voice, and every text of an evaluation set (the FLEURS sv_se test
   sentences and the recipe's committed sets of Swedish sentences with English words, and with numbers) are
   kept out of `train` and `validation`; no text trains more than three times.
6. **Shares.** {", ".join(f"{s} {sh:.0%}" for s, sh in sel['share'].items())} of the hours, with per-speaker caps
   ({", ".join(f"{s} {sel[s]['cap_minutes_per_speaker']:.0f} min" for s in sources)}), round-robin over gender and region;
   a source that runs short leaves its shortfall to the others.
7. **Clean.** Every clip is mixed to mono, trimmed to its speech (frames {-recipe['audio']['silence_trim_db']:.0f} dB below its peak cut, {recipe['audio']['keep_margin_seconds']} s margin kept)
   and dropped when clipped, quiet (below {recipe['audio']['min_rms_dbfs']} dBFS) or noisy (energy-percentile SNR below
   {recipe['audio']['min_snr_db']} dB). `reports/` holds the selection, audio and segmentation reports with every rejection reason.

## Named voices

| voice | gender | region | NST session | liveliness | F0 range (st) | F0 sd (st) | level sd (dB) | syllables/s |
|---|---|---|---|---|---|---|---|---|
{voice_rows}

## Limitations

- The Swedish sources are 16 kHz recordings: a model trained on them lacks the top of a 24 kHz band.
  Only the English LibriTTS-R audio is 24 kHz.
- NST is read speech recorded around 1999; RixVox is parliamentary debate; LibriVox is narration of old
  books in older spelling. Conversational and emotional speech are scarce.
- Transcripts are the sources' own (RixVox: Whisper's verbatim transcript, which can still err; LibriVox:
  the source text aligned to the audio). Numbers and abbreviations appear as each source wrote them.
- No Finland Swedish.

## What not to use it for

The RixVox-v2 speakers are identifiable public persons (members of the Swedish Parliament). Do not use their
recordings, or this dataset, to imitate or impersonate them or any other identifiable person, and disclose
synthetic speech wherever listeners could take it for a real person.

## Licence and terms

The terms are in [TERMS.md](TERMS.md). In short: our contribution (selection, cleaning, segmentation and
alignment, the transcripts we produced, metadata, voice assignments, this card) is © 2026 Swedish Embedded AB
under CC-BY-4.0; every clip and its source transcript stay under the source's licence, named on its row
(NST CC0-1.0, LibriVox CC0-1.0, RixVox-v2 ODC-BY-1.0, LibriTTS-R CC-BY-4.0, FLEURS CC-BY-4.0), and nothing in our
terms restricts what those licences allow. Redistributing the dataset, or publishing a model trained on it,
credits "{spec['name']} (Swedish Embedded AB)" and carries the sources' attributions below. The responsible-use
policy above binds the use of our contribution and is a request for the source material. No warranty.

### Attributions the sources require

RixVox-v2 (ODC-BY-1.0):

> {rix}

LibriTTS-R (CC-BY-4.0), derived from LibriTTS and LibriSpeech:

```bibtex
{LIBRITTS_R_CITATION}
```

FLEURS (CC-BY-4.0):

```bibtex
{FLEURS_CITATION}
```

NST Swedish ASR Database (CC0-1.0), created by Nordic Language Technology (reorganised by KTH as
[KTH/nst](https://huggingface.co/datasets/KTH/nst)). LibriVox Swedish recordings (CC0-1.0, public domain), via
[datadriven-company/TTS-Swedish](https://huggingface.co/datasets/datadriven-company/TTS-Swedish).
"""
