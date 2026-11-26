<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Sten-TTS-1.7B

> **A Swedish or Nordic voice for your product, trained on your data and run on your infrastructure**
>
> For product teams, public-sector services and device makers that need speech in Swedish, another Nordic
> language or a regional dialect, and cannot send their users' text to a cloud service.
>
> Swedish Embedded AB will:
> - build your brand voice, language or dialect with a recipe like this one, from data selection to release;
> - adapt pronunciation to your domain: product names, numbers, terminology, measured on your own sentences;
> - deploy the model privately, behind an OpenAI-compatible speech API or on an edge device.
>
> This repository is the whole recipe: pinned open sources, licence-checked selection, training across a
> GPU fabric and a gated evaluation, every stage reproducible from one command.
>
> Tell Martin Schröder about your use case on [LinkedIn](https://www.linkedin.com/in/martinschroder/), or email info@swedishembedded.com.

Built by Martin Schröder <info@swedishembedded.com> ([LinkedIn](https://www.linkedin.com/in/martinschroder/)), Swedish Embedded AB.

The complete recipe that produces **Sten-TTS-1.7B**, a Swedish multi-speaker
text-to-speech model with twelve named voices, by fine-tuning
[Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
on 420 hours of open speech (Swedish, with 10 % English so that English words
inside Swedish sentences keep their pronunciation). Training runs entirely inside
[brain](https://github.com/swedishembedded/brain); this repository only selects
and prepares data, drives brain over its API (through `brain-py`), checks the
result and publishes it.

The release is an ordinary Qwen3-TTS `custom_voice` checkpoint in the upstream
layout. It loads unchanged in brain and in the upstream `qwen-tts` package
(transformers 4.57.3) and adds `swedish` as a language.

This is the v2 recipe (`recipe/sten-tts-1.7b-v2.yaml`, the default). It fixes
v1's problems: v1 trained on 100 h of Swedish only (English words came out
badly), its voices were the NST speakers with the most audio (20-35 minutes of
flat read speech each), and all of it was 16 kHz. `recipe/sten-tts-1.7b.yaml`
reproduces v1.

Swedish Embedded AB implements solutions for training speech models on your own
data, in your own languages and voices, for its clients. If your team needs
expertise in text-to-speech or speech-model training, you can procure our
services by sending an email to info@swedishembedded.com.

## What you get

| | |
|---|---|
| Base | Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0), pinned revision in the recipe |
| Training audio | 420 h after filtering and the cut: NST Swedish (CC0, read speech, 10 regions), RixVox-v2 (ODC-BY, parliament debates), LibriVox Swedish audiobooks (CC0, segmented and aligned here), FLEURS sv_se train+dev (CC-BY-4.0) and LibriTTS-R English (CC-BY-4.0, 24 kHz) |
| Speakers | about 4300 in training (2674 of them single FLEURS recordings); 12 NST speakers become named voices |
| Voices | astrid, erik (Mellansverige), linnea, viktor (Stockholm), elsa, oskar (Göteborg), maja, nils (Västra sydsverige), signe, arvid (Norrland), ingrid, axel (Dalarna) |
| Languages | `swedish`, a new codec language row learnt in training; `english`, the base's own row, trained on alongside |
| Output | 24 kHz, 12.5 Hz codec frames |
| Held out | 23 whole unseen speakers (NST, RixVox, LibriVox, LibriTTS-R), 20 unseen sentences per voice, 200 FLEURS sv_se test sentences, 60 Swedish sentences with English words and 40 with numbers (never trained on) |
| Dataset | the training collection is published as the dataset `swedishembedded/sten-tts-voice` (stages `export`, `publish-dataset`) |

## Requirements

- **brain**, built from a revision that has the speech fine-tuning job
  (`brain/train` action `speech_sft`), the `encode` action of Qwen3-TTS and
  language selection on `/v1/audio/speech`. Build it as brain's README says;
  CUDA is needed for training at a useful speed (one GPU with about 60 GB).
- **The base, in brain's model store:**
  `brain pull Qwen/Qwen3-TTS-12Hz-1.7B-Base` and, for the evaluation,
  `brain pull Qwen/Qwen3-ASR-1.7B`.
- **brain-py**, brain's Python client: `pip install -e <brain>/brain-py`.
- **Python 3.11+**, and about 500 GB of free disk for the work directory
  (about 350 GB of sources, the rest segments, cleaned audio, shards,
  checkpoints, reports and the dataset export). A fast local disk is best.
- **A Hugging Face token** (`hf auth login`) to download the sources and, for
  `publish` only, write access to the target organisation.
- For the upstream compatibility check in `evaluate`: a second environment
  with `torch` (CPU is enough) and `qwen-tts==0.1.1`, named by `--upstream-python`.

## Training on brain

brain is the training backend. The recipe never runs a training loop or
loads model weights itself: it prepares data, then asks a running brain server
to do the model work, over D-Bus, through brain's Python client `brain-py`.

| work | who does it |
|---|---|
| choosing, downloading, cleaning and splitting the audio | this recipe (`fetch` .. `audio`) |
| codec codes, speaker x-vectors and token ids of every clip | brain, Qwen3-TTS action `encode` (called by `shards`) |
| writing them into training shards | this recipe (brain's `brain-tts-shard-v1` format; brain validates it) |
| the fine-tune: Talker and code predictor, new `swedish` row and twelve voice rows, checkpoints, resume | brain, `brain/train` action `speech_sft` (called by `train`) |
| exporting the checkpoint in the upstream layout and installing it | brain, at the end of the same job |
| synthesis, recognition and speaker similarity for the gates | brain, served models (called by `evaluate`) |

### 1. Build brain

```bash
git clone <brain repository> brain && cd brain
make build/release                      # ./target/release/brain; CUDA is found at run time
```

Training needs one NVIDIA GPU with about 60 GB free (the run is charged four
f32 copies of the Talker and code predictor plus a step's activations; brain
refuses a card it does not fit on). It also runs on the CPU backend with
`allow_cpu`, which is only useful for a smoke test.

### 2. Put the models in brain's model store

```bash
./target/release/brain pull Qwen/Qwen3-TTS-12Hz-1.7B-Base    # the base that is fine-tuned
./target/release/brain pull Qwen/Qwen3-ASR-1.7B              # the judge used by `evaluate`
```

`brain models list` should show the base with family `qwen3tts`. Add
`--brain-data-dir DIR` to every brain command if the store is not in brain's
default data directory.

### 3. Start the brain server

```bash
WORK=/fast/disk/sten-run                 # the recipe's work directory; any disk with ~200 GB
mkdir -p "$WORK/artifacts"
dbus-run-session -- bash                 # on a headless machine: a session bus for server and recipe
./target/release/brain serve --dbus --training --artifact-store "$WORK/artifacts" --device gpu0,cpu &
```

- `--training` turns on the `brain/train` job service; without it `train` is refused.
- `--artifact-store DIR` is where brain keeps shards it is handed, its run state
  (for resuming) and the checkpoint files it publishes. The recipe's `train`
  stage is given the same directory to collect the release from.
- `--device gpu0,cpu` pins the server to one card (and the host for host-side work); leave it out to let brain choose.
- Run the recipe in the same D-Bus session (the shell `dbus-run-session` opened).

### 4. Prepare the data and encode it with brain

```bash
make env && .venv/bin/pip install -e <brain>/brain-py
make fetch inventory select audio WORKDIR="$WORK" JOBS=16
make shards WORKDIR="$WORK" CONCURRENCY=8
```

`shards` sends the kept clips (16-bit WAVs at their source rate, with their
transcripts), 32 to a request, to the Qwen3-TTS model's `encode` action, which
encodes each request's clips in one batch on the card, and gets back each clip's
`[frames, 16]` codec codes, the speaker encoder's x-vector and the token ids of
`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n` from the
checkpoint's own tokenizer. It cuts the training set at exactly 100 hours
(the named voices first, then by the seed's order) and writes
`shards/train.bttshrd` and `shards/validation.bttshrd`. On one GH200 the
encode runs at about 700 audio seconds per second with 8 requests in flight,
about 9 minutes for the corpus. Results are saved every 2000 clips in
`shards/cache/`, so an interrupted run resumes; delete that directory when
brain's encoder changes, since the cache does not know which build made it.

### 5. Train

```bash
make train WORKDIR="$WORK" ARTIFACTS="$WORK/artifacts"
```

`train` first sends the job with `validate_only: true`. brain then checks
the shards (format, codebook count, x-vector width), every language and voice
name against the base, and whether the run fits the card, and answers what it
would train without taking the card. Then it submits the real job and follows
it. The request `train` builds from the recipe is:

```json
{
  "model": "brain/train",
  "action": "speech_sft",
  "params": {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "languages": "{\"swedish\": {\"new\": true, \"init_from\": [\"german\", \"english\"]}, \"english\": {\"new\": false}}",
    "voices": "{\"astrid\": {\"new\": true}, \"erik\": {\"new\": true}, ...}",
    "steps": 5560,
    "lr": 1e-05,
    "grad_accum": 2,
    "residual_weight": 0.3,
    "eval_every": 250,
    "seed": 20261125,
    "cards": 1,
    "install_as": "sten-tts-1.7b",
    "resume_key": "sten-tts-1.7b-v1",
    "checkpoint_every": 500
  },
  "blobs": {"dataset": "shards/train.bttshrd", "held_out": "shards/validation.bttshrd"}
}
```

`steps` (5560 for this corpus) is the recipe's `epochs` times the training utterances divided by
`batch_utterances * grad_accum`. The shards travel as D-Bus file descriptors,
so nothing is copied through a socket.

While it runs, every optimiser step arrives as a progress event (`step`,
`steps`, `loss`, `lr`, `grad_norm`, `seconds`, the Talker's and the code
predictor's losses, and the held-out loss on evaluating steps). `train` prints
them and appends them to `train/events.jsonl`.

**Interrupted?** Run `make train` again. The same `resume_key` and the same
request make brain continue from its last checkpoint (every
`checkpoint_every` steps) with the exact optimiser, data-order and random
state. A changed request under the same key is refused. Change `resume_key`
in the recipe to start over. **Ctrl-C** cancels the job; brain stops at the
next step and keeps the last checkpoint.

When the job finishes, brain has:

1. exported the trained weights over the base in the upstream layout: one BF16
   `model.safetensors` in the base's tensor names, `config.json` with
   `codec_language_id.swedish` and the twelve voices in `spk_id` as a
   `custom_voice` checkpoint, the tokenizer and `speech_tokenizer/` unchanged;
2. published those files to the artifact store with a checkpoint manifest
   (`checkpoint_digest` in the result);
3. installed the checkpoint in its model store as `local/sten-tts-1.7b`.

`train` then copies the manifest's files from the artifact store into
`release/`, checks every digest, adds the model card (`README.md`, with the
dataset attribution) and writes `train/report.json`, which holds the job's
result and the codec row of `swedish` and of every voice.

### 6. Use the trained model with brain

Restart the server so it loads the installed model. It is then served like
any other speech model:

```bash
curl -s http://127.0.0.1:8788/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "local/sten-tts-1.7b", "voice": "astrid", "language": "swedish",
       "input": "Välkommen till Sten, en svensk röst.", "response_format": "wav"}' -o astrid.wav
```

That needs the HTTP surface (`--openai 127.0.0.1:8788`). Over D-Bus, use
brain-py: `Brain().run("local/sten-tts-1.7b", "design", {"text": ..., "speaker": "astrid", "lang": "swedish"})`.
brain also loads `release/` directly, as does upstream `qwen-tts`:
`Qwen3TTSModel.from_pretrained("release").generate_custom_voice(text, speaker="astrid", language="swedish")`.

### 7. Evaluate and publish

```bash
make evaluate WORKDIR="$WORK"
make samples  WORKDIR="$WORK"    # only after changing publish.samples: evaluate already writes them
make publish  WORKDIR="$WORK" HUB_DIR=<local clone of the Hugging Face repository>
```

### Doing it by hand

Every stage is a few calls on brain-py, so the same training can be driven
from a notebook:

```python
from pathlib import Path
from brain_py import Brain

with Brain() as brain:                                    # D-Bus, `brain serve --dbus`
    enc = brain.encode_audio(Path("clip.wav").read_bytes(), "Hej, jag heter Astrid.")
    # enc.codes [frames, 16] u16, enc.xvector [2048] f32, enc.text_ids u32: write shards with sten_tts.shard.ShardWriter
    params = {"base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "languages": '{"swedish": {"new": true}}',
              "voices": '{"astrid": {"new": true}}', "steps": 1000, "resume_key": "my-run", "install_as": "my-model"}
    blobs = {"dataset": Path("train.bttshrd").read_bytes()}
    print(brain.run("brain/train", "speech_sft", {**params, "validate_only": True}, blobs=blobs).outputs)
    out = brain.subscribe("brain/train", "speech_sft", params, blobs=blobs,
                          on_progress=lambda step, total, msg: print(step, total, msg), timeout=7 * 86400)
    print(out.outputs["checkpoint_digest"])
```

## Running it end to end

```bash
make all WORKDIR="$WORK" ARTIFACTS="$WORK/artifacts"    # fetch .. evaluate
make publish WORKDIR="$WORK" HUB_DIR=<local mirror>
```

Each stage is also `python -m sten_tts.cli <stage> --workdir DIR` (see
`--help`) and can be rerun on its own; it reads only what earlier stages wrote.

### What each stage does and leaves behind

| stage | reads | writes (under the work directory) |
|---|---|---|
| `fetch` | the recipe's `sources` | `raw/<source>/...`, `raw/<source>/fetched.json` (files, sizes, blob ids) |
| `segment` | `raw/librivox/`, the brain server (recogniser) | `segments/segments.parquet` (every cut: times, book text, recognition, CER, rejection), `segments/audio-NNNN.parquet`, `segments/report.json` |
| `inventory` | `raw/`, `segments/` | `inventory/<source>.parquet`: every utterance, its speaker, metadata, language (`swedish`/`english`), transcript, duration |
| `voices` | `inventory/nst.parquet`, `raw/nst/` | `selection/voices.json`: per voice slot the liveliest NST speaker with enough audio, their measures and score, and the runners-up |
| `select` | `inventory/`, FLEURS transcripts | `selection/selection.parquet` (utterance -> split, voice), `selection/voices.json`, `selection/report.json`, `evaluation/sentences.json` |
| `audio` | `selection/`, `raw/` | `audio/<split>-NNNN.parquet` (trimmed 16-bit WAV + measures), `audio/rejected.parquet`, `audio/report.json` |
| `shards` | `audio/`, the brain server | `shards/{train,validation}.bttshrd` (brain's `brain-tts-shard-v1`), `shards/report.json` |
| `compress` | a variant recipe, `shards/`, `evaluation/sentences.json`, the brain server | `compress/report.json` (the surgery manifest and each edit's measured loss and KL), `compress/checkpoint/` (the cut, installed as `compress.install_as`) |
| `train` | `shards/`, the brain server | the job's events in `train/events.jsonl`, `release/` (the checkpoint), `train/report.json` |
| `evaluate` | `release/`, the brain server | `evaluation/clips.parquet`, `evaluation/report.json` with every gate, the samples |
| `speed` | a variant recipe, the first brain server | `evaluation/speed.json`: real-time factor and codec frames a second of the variant and the full model, one request at a time on one card |
| `samples` | `evaluation/clips.parquet`, `publish.samples` | `release/samples/`: one unpicked clip of an unseen FLEURS sentence per voice, played by the model card |
| `publish` | `release/`, `evaluation/report.json` | the model repository: created private if missing, uploaded, then made public when `publish.private` is false |
| `export` | `selection/`, `audio/`, `segments/report.json` | `export/`: the training collection as a Hugging Face dataset (`data/<split>-*.parquet` with FLAC audio, `README.md` card, `TERMS.md`, `voices.json`, `reports/`, `manifest.json`) |
| `publish-dataset` | `export/` | the dataset repository: created private, uploaded, checked file for file and row for row against the manifest, then made public |

### Selection, in short

`select` is a pure function of the inventories, the voices stage's choice, the
evaluation sentences and the recipe's seed; rerunning it gives a
byte-identical selection. With the v2 recipe it:

- keeps NST read sentences of at least three words (no letter-by-letter
  spelling items), RixVox segments whose protocol text and Whisper transcript
  agree (WER at most 0.10, Swedish with probability at least 0.98, the verbatim
  Whisper transcript), the LibriVox segments that `segment` checked, FLEURS
  train and dev, and LibriTTS-R English (language `english`, never a voice);
- keys NST speakers by recording session, because NST reuses one speaker id for
  different people;
- takes each voice from `selection/voices.json` (the liveliest NST speaker of
  its gender and region with at least 20 minutes, see `voices`);
- holds out whole speakers per source (spread over genders, drawn among the
  speakers with at most the median audio), 20 sentences per voice, and bans
  every FLEURS test sentence, every committed evaluation sentence and every
  held-out text from training;
- fills each source's share of 420 h (NST 45 %, RixVox 35 %, LibriVox 8 %,
  FLEURS 2 %, LibriTTS-R 10 %) with a per-speaker cap (20 min NST, 30 min
  RixVox, 5 h a LibriVox book, 20 min LibriTTS-R), round-robin over gender and
  region; a source that runs short leaves its shortfall to the others; then
  oversamples by 15 % so the audio filters can drop clips and the shards are
  cut at exactly 420 h; 1 % of the training speakers' utterances validate.

`selection/report.json` lists hours per split, source, language, gender and
region, speakers per source, every voice's speaker and hours, and every
rejection reason with its count.

## Checking the result

`make test` runs the recipe's own tests (no data or server needed): the
selection's determinism and disjointness, caps and budgets, the shard layout,
audio trimming and rejection, measures, and the publish guard.

`evaluate` decides whether the release is good enough. Every voice reads the
200 FLEURS sentences with two seeds; the base model reads the same sentences
(cloning each voice from one of its held-out recordings) as the comparison.
The judge is Qwen3-ASR-1.7B served by brain. The gates, all in the recipe:

| gate | passes when |
|---|---|
| `cer_over_floor_max` | mean recogniser CER is at most 0.03 above the recogniser's own CER on the voices' real held-out recordings |
| `cer_better_than_base` | Sten's CER is below the base's on the same sentences (paired bootstrap, 95 % interval below zero) |
| `voice_similarity_gap_max` | for each voice, its clips are at most 0.05 less similar (speaker encoder) to its held-out recordings than those recordings are to each other |
| `voice_identity_accuracy_min` | at least 90 % of clips are nearest to their own voice among the twelve |
| `failure_rate_max` | at most 2 % of clips hit the length cap, are silent or loop |
| `upstream_loads` | the release loads in upstream `qwen-tts` and speaks Swedish with a named voice |

The recogniser's own CER on the real held-out recordings is reported beside
these as the floor that synthesised speech is compared against.

Besides FLEURS, every voice (and the base, for comparison) reads two
committed, hand-written sets that test v1's failures, scored the same way:
`recipe/evalsets/code-switching.tsv` (60 Swedish sentences with English words,
names and brands: technical, everyday, products) and
`recipe/evalsets/numbers.tsv` (40 with numbers, dates, times, units and
abbreviations, each with its spoken form; a clip counts against whichever of
the two forms the recogniser's transcript is closer to). Their texts never
train. `evaluation/report.json` has, under `sets`, the CER of each set overall,
for the base, per voice and per region, and the model card shows the per-voice
and per-region tables. `publish`
refuses unless every gate passed.

## Compact variants

A variant recipe (`recipe/sten-tts-950m.yaml`) makes a smaller model out of the
trained Sten-TTS-1.7B. It names the full recipe (`variant_of`) and takes the
data, the voices, the evaluation sets and the release gates from it; it adds:

- `teacher`: the full model in brain's model store;
- `compress`: a weight-surgery plan for brain's `speech_surgery` (vocabulary
  trim, MLP channel selection by Fisher score, layer removal by measured KL,
  optionally chosen greedily with `greedy: true`), and the most parameters
  `model.safetensors` may hold;
- `train`: distillation from the full model (`train.distill`: the weights of
  the ground-truth loss and of the KL terms of the first and the residual
  codebooks, and the temperature); the held-out KL to the full model picks
  the checkpoint and stops the run;
- `evaluate.parity`: the full model's evaluation directory and the margins the
  variant must hold against it, clip for clip (FLEURS CER, and per voice
  similarity, identification and failures), as extra gates;
- `evaluate.speed`: the sentences and voice the `speed` stage times;
- `publish.subfolder`: the folder of the full model's repository it goes to.

The variant's work directory needs from the full run only what evaluation
reads: `selection/`, `audio/voice_test-*.parquet`, `shards/` (the shards
themselves, or links to them), `evaluation/sentences.json`, and the full
model's evaluation (`clips.parquet`, `report.json`, `sentences.json`) under
`evaluate.parity.teacher_evaluation`. The base model's clips and the
recogniser floor are taken from that evaluation instead of being made again,
when it provably did the same computation: the same sentences file, voices
and seed, and the same real-recording similarity per voice (the same
reference recordings and encoder); otherwise they are made again.

```bash
R=recipe/sten-tts-950m.yaml
scripts/with-brain.sh --brain BIN --workdir "$WORK" --recipe $R --device gpu0,cpu compress
scripts/with-brain.sh --brain BIN --workdir "$WORK" --recipe $R --device gpu,cpu --cards 4 train
scripts/with-brain.sh --brain BIN --workdir "$WORK" --recipe $R --upstream-python PY \
    --device gpu0,cpu --device gpu1,cpu --device gpu2,cpu --device gpu3,cpu evaluate speed samples
make publish WORKDIR="$WORK" RECIPE=$R HUB_DIR=<local mirror>
```

## Status

| stage | state |
|---|---|
| `fetch`, `segment`, `inventory`, `voices`, `select`, `audio`, `shards` | done for v2 on the real sources: 478 h selected, 454 h kept after the audio filters, the shards cut at 420 h (194 769 records, 375.8 h Swedish, 44.2 h English) |
| `train`, `evaluate` | the v2 run trains on those shards |
| `export`, `publish-dataset` | the training collection as the public dataset `swedishembedded/sten-tts-voice` |
| `publish` | done (guarded; refuses until `evaluate` passes) |
| `compress`, `train`, `evaluate`, `speed` for `recipe/sten-tts-950m.yaml` | the first compact rung, Sten-TTS-950M: cut to 952.6 M parameters (held-out KL 0.864 to the full model before recovery), distilled on 4 cards for 6 epochs with hidden-state matching; every release and parity gate passes (FLEURS CER 0.0635 against 0.0637) |
| `compress`, `train`, `evaluate`, `speed` for `recipe/sten-tts-475m.yaml` | the second rung, cut from the first to 469.6 M parameters (KL 2.858 to the full model before recovery): every gate but CER parity holds; after four recoveries FLEURS CER stays 0.0125 above the full model's (parity allows 0.0025), so it is not released |

### The v2 corpus

| source | train h (selected) | speakers | notes |
|---|---|---|---|
| NST | 231.8 | 833 | read speech, 10 dialect regions; 20 min cap per session; voices take all their audio |
| RixVox-v2 | 167.4 | 543 | 100 shards (2477 h, 700 speakers) before filtering; WER <= 0.10 keeps far fewer |
| LibriVox | 22.3 | 8 books | `segment` kept 23.1 h of 40.4 h (CER rejections 15.6 h, edge words 1.6 h) |
| FLEURS sv_se train+dev | 9.1 | 2674 recordings | readers not identified |
| LibriTTS-R | 47.8 | 238 | English, 24 kHz |

By language 430.6 h Swedish and 47.8 h English; by gender 217.7 h female,
212.9 h male, 47.8 h unspecified (LibriTTS-R). These are selected hours with
the 15 % oversample; the audio filters dropped 490 clips and trimming the rest
left 454 h, cut to 420 h for the shards.

The voices (`selection/voices.json`, the liveliest NST speaker per slot with at
least 20 minutes): their liveliness scores run from 0.92 (ingrid) to 2.38
(linnea); every slot had 15-51 candidates, and every chosen voice scores above
the mean of its gender (score 0).

## Deviations from the original plan

- **Open data only.** No authorised persona recordings exist yet, so the twelve
  voices are anonymous NST speakers (CC0). The Swedish recordings are 16 kHz, so
  the 24 kHz output is band-limited until wideband Swedish material is added
  (only LibriTTS-R, English, is 24 kHz).
- **LibriVox yields less than proposed.** The recogniser's CER against the book
  text is above 0.10 for 39 % of the LibriVox audio (old spelling and names),
  so `segment` keeps 23 h instead of ~35 h; the selection gives the shortfall to
  the other sources.
- **The LibriVox recognition pass is bound by brain's Qwen3-ASR**, which decodes
  one request at a time per instance (about 11 audio seconds a second on a
  GH200); spreading it over eight servers on six cards (two per card on n80, one per card
  on n460) took it to about 75.
- **Voices need 20 minutes, not 25.** At 25 minutes a slot has two to eight
  candidates and two slots' best scored below their gender's mean; at 20 it has
  15-51.
- **No Finland Swedish.** NST has no Finland Swedish speakers; the last voice
  pair is from Dalarna.
- **RixVox speakers are not voices.** They are identifiable public persons, so
  they only teach Swedish through their x-vectors, never as a named voice.
- **Text path frozen.** The text embedding and its projection keep the base's
  weights in this release; the Talker, the code predictor and every codec table
  are trained.

## Licences and attribution

The recipe is Apache-2.0 (`LICENSE`). The base model is Apache-2.0. NST and
LibriVox are CC0-1.0. FLEURS and LibriTTS-R are CC-BY-4.0 (the model card and
the dataset card carry their citations). The release carries this notice,
which RixVox-v2's licence requires:

> Contains information from the dataset RixVox-v2
> (https://huggingface.co/datasets/KBLab/rixvox-v2), made available under the
> ODC Attribution License, sourced from the Swedish Parliament's open data.
