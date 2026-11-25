<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Sten-TTS-1.7B

The complete recipe that produces **Sten-TTS-1.7B**, a Swedish multi-speaker
text-to-speech model with twelve named voices, by fine-tuning
[Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
on 100 hours of open Swedish speech. Training runs entirely inside
[brain](https://github.com/swedishembedded/brain); this repository only selects
and prepares data, drives brain over its API (through `brain-py`), checks the
result and publishes it.

The release is an ordinary Qwen3-TTS `custom_voice` checkpoint in the upstream
layout. It loads unchanged in brain and in the upstream `qwen-tts` package
(transformers 4.57.3) and adds `swedish` as a language.

Swedish Embedded AB implements solutions for training speech models on your own
data, in your own languages and voices, for its clients. If your team needs
expertise in text-to-speech or speech-model training, you can procure our
services by sending an email to info@swedishembedded.com.

## What you get

| | |
|---|---|
| Base | Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0), pinned revision in the recipe |
| Training audio | 100 h after filtering: NST Swedish (CC0, read speech, 10 regions) and RixVox-v2 (ODC-BY, parliament) |
| Speakers | about 790 in training; 12 of them become named voices |
| Voices | astrid, erik (Mellansverige), linnea, viktor (Stockholm), elsa, oskar (Göteborg), maja, nils (Västra sydsverige), signe, arvid (Norrland), ingrid, axel (Dalarna) |
| Language | `swedish`, a new codec language row learnt in training |
| Output | 24 kHz, 12.5 Hz codec frames |
| Held out | 18 whole unseen speakers, 20 unseen sentences per voice, 200 FLEURS sv_se sentences (never trained on) |

## Requirements

- **brain**, built from a revision that has the speech fine-tuning job
  (`brain/train` action `speech_sft`), the `encode` action of Qwen3-TTS and
  language selection on `/v1/audio/speech`. Build it as brain's README says;
  CUDA is needed for training at a useful speed (one GPU with about 60 GB).
- **The base, in brain's model store:**
  `brain pull Qwen/Qwen3-TTS-12Hz-1.7B-Base` and, for the evaluation,
  `brain pull Qwen/Qwen3-ASR-1.7B`.
- **brain-py**, brain's Python client: `pip install -e <brain>/brain-py`.
- **Python 3.11+**, and about 200 GB of free disk for the work directory
  (120 GB of sources, the rest shards, checkpoints and reports). A fast local
  disk is best.
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
./target/release/brain serve --dbus --training --artifact-store "$WORK/artifacts" --device gpu0 &
```

- `--training` turns on the `brain/train` job service; without it `train` is refused.
- `--artifact-store DIR` is where brain keeps shards it is handed, its run state
  (for resuming) and the checkpoint files it publishes. The recipe's `train`
  stage is given the same directory to collect the release from.
- `--device gpu0` pins the server to one card; leave it out to let brain choose.
- Run the recipe in the same D-Bus session (the shell `dbus-run-session` opened).

### 4. Prepare the data and encode it with brain

```bash
make env && .venv/bin/pip install -e <brain>/brain-py
make fetch inventory select audio WORKDIR="$WORK" JOBS=16
make shards WORKDIR="$WORK" CONCURRENCY=8
```

`shards` sends every kept clip (a 16-bit WAV at its source rate, with its
transcript) to the Qwen3-TTS model's `encode` action and gets back its
`[frames, 16]` codec codes, the speaker encoder's x-vector and the token ids of
`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n` from the
checkpoint's own tokenizer. It cuts the training set at exactly 100 hours
(the named voices first, then by the seed's order) and writes
`shards/train.bttshrd` and `shards/validation.bttshrd`. On the CPU backend
encoding runs at about 6.7 audio seconds per second with four streams (about
15 hours for the corpus); on a GPU it is much faster.

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
    "languages": "{\"swedish\": {\"new\": true, \"init_from\": [\"german\", \"english\"]}}",
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
| `inventory` | `raw/` | `inventory/<source>.parquet`: every utterance, its speaker, metadata, transcript, duration |
| `select` | `inventory/`, FLEURS transcripts | `selection/selection.parquet` (utterance -> split, voice), `selection/voices.json`, `selection/report.json`, `evaluation/sentences.json` |
| `audio` | `selection/`, `raw/` | `audio/<split>-NNNN.parquet` (trimmed 16-bit WAV + measures), `audio/rejected.parquet`, `audio/report.json` |
| `shards` | `audio/`, the brain server | `shards/{train,validation}.bttshrd` (brain's `brain-tts-shard-v1`), `shards/report.json` |
| `train` | `shards/`, the brain server | the job's events in `train/events.jsonl`, `release/` (the checkpoint), `train/report.json` |
| `evaluate` | `release/`, the brain server | `evaluation/clips/`, `evaluation/report.json` with every gate |
| `publish` | `release/`, `evaluation/report.json` | the private Hugging Face repository |

### Selection, in short

`select` is a pure function of the inventories, the FLEURS transcripts and the
recipe's seed; rerunning it gives a byte-identical selection. It:

- keeps NST read sentences of at least three words (no letter-by-letter
  spelling items) and RixVox segments whose protocol text and Whisper transcript
  agree (WER at most 0.10), Swedish with probability at least 0.98, using the
  verbatim Whisper transcript;
- keys NST speakers by recording session, because NST reuses one speaker id for
  different people;
- picks each voice as the NST speaker with the most eligible audio in its gender
  and region;
- holds out whole speakers (balanced by gender), 20 sentences per voice, and
  bans every FLEURS sentence and every held-out text from training;
- fills 75 % NST and 25 % RixVox with at most 8 (NST) or 20 (RixVox) minutes per
  auxiliary speaker, round-robin over gender and region, then oversamples by
  15 % so the audio filters can drop clips and the shards are cut at exactly
  100 h.

`selection/report.json` lists hours per split, source, gender and region, every
voice's speaker and hours, and every rejection reason with its count.

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
these as the floor that synthesised speech is compared against. `publish`
refuses unless every gate passed.

## Status

| stage | state |
|---|---|
| `fetch`, `inventory`, `select`, `audio` | done; run on the real sources (105 h kept for training before the 100 h cut) |
| `shards`, `train`, `evaluate` | being written against brain's `encode` and `speech_sft` actions, which are landing in brain now |
| `publish` | done (guarded; refuses until `evaluate` passes) |

## Deviations from the original plan

- **Open data only.** No authorised persona recordings exist yet, so the twelve
  voices are anonymous NST speakers (CC0). The recordings are 16 kHz, so the
  24 kHz output is band-limited until wideband material is added.
- **No Finland Swedish.** NST has no Finland Swedish speakers; the last voice
  pair is from Dalarna.
- **RixVox speakers are not voices.** They are identifiable public persons, so
  they only teach Swedish through their x-vectors, never as a named voice.
- **Text path frozen.** The text embedding and its projection keep the base's
  weights in this release; the Talker, the code predictor and every codec table
  are trained.

## Licences and attribution

The recipe is Apache-2.0 (`LICENSE`). The base model is Apache-2.0. NST is
CC0-1.0. FLEURS (evaluation only) is CC-BY-4.0. The release carries this
notice, which RixVox-v2's licence requires:

> Contains information from the dataset RixVox-v2
> (https://huggingface.co/datasets/KBLab/rixvox-v2), made available under the
> ODC Attribution License, sourced from the Swedish Parliament's open data.
