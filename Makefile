# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Every stage of the Sten-TTS-1.7B recipe. WORKDIR is required and holds all of
# a run's data (about 350 GB of sources with the v2 recipe, plus shards, checkpoints and reports):
#
#   make all WORKDIR=/fast/disk/sten-run
#
# PYTHON is the interpreter of an environment with this package installed
# (`make env` makes one in .venv).

WORKDIR ?=
PYTHON ?= .venv/bin/python
RECIPE ?= recipe/sten-tts-1.7b-v2.yaml
JOBS ?= 16
CONCURRENCY ?= 8
HUB_DIR ?=
ARTIFACTS ?=
UPSTREAM_PYTHON ?=
STEN := $(PYTHON) -m sten_tts.cli
ARGS = --workdir "$(WORKDIR)" --recipe "$(RECIPE)"

.PHONY: help env test all fetch segment inventory voices select audio shards train evaluate publish export publish-dataset check-workdir

## help - list the targets
help:
	@grep -E '^## ' Makefile | sed 's/^## //'

## env - a virtual environment in .venv with this package and its test tools
env:
	python3 -m venv .venv && .venv/bin/pip install -q -e '.[test]'

## test - the recipe's own tests (no data, no model server)
test:
	$(PYTHON) -m pytest -q

check-workdir:
	@test -n "$(WORKDIR)" || { echo "set WORKDIR=<a directory with ~500 GB free>"; exit 2; }

## all - every stage in order
all: fetch segment inventory voices select audio shards train evaluate

## fetch - download the pinned source files (only the ones the recipe names)
fetch: check-workdir
	$(STEN) fetch $(ARGS)

## segment - cut the LibriVox chunks at pauses, recognise each piece with the brain server, keep the ones matching the book text
segment: check-workdir
	$(STEN) segment $(ARGS) --concurrency $(CONCURRENCY) --jobs $(JOBS)

## inventory - one table per source of every candidate utterance, without audio
inventory: check-workdir
	$(STEN) inventory $(ARGS) --jobs $(JOBS)

## voices - the liveliest NST speaker of each voice slot, with the scores of the runners-up
voices: check-workdir
	$(STEN) voices $(ARGS) --jobs $(JOBS)

## select - the deterministic training selection, held-out sets and voices
select: check-workdir
	$(STEN) select $(ARGS)

## audio - measure, trim and filter the selected clips
audio: check-workdir
	$(STEN) audio $(ARGS) --jobs $(JOBS)

## shards - codes, x-vectors and token ids from the brain server, into training shards
shards: check-workdir
	$(STEN) shards $(ARGS) --concurrency $(CONCURRENCY)

## train - the served fine-tuning job on the brain server, and the release directory
train: check-workdir
	@test -n "$(ARTIFACTS)" || { echo "set ARTIFACTS=<the brain server's --artifact-store directory>"; exit 2; }
	$(STEN) train $(ARGS) --artifact-store "$(ARTIFACTS)"

## evaluate - synthesis by every voice, recognition, similarity, gates; upstream loader check
evaluate: check-workdir
	$(STEN) evaluate $(ARGS) --concurrency $(CONCURRENCY) $(if $(UPSTREAM_PYTHON),--upstream-python "$(UPSTREAM_PYTHON)")

## publish - upload the release to its private Hugging Face repository (HUB_DIR: a local mirror)
publish: check-workdir
	$(STEN) publish $(ARGS) $(if $(HUB_DIR),--hub-dir "$(HUB_DIR)")

## export - the training collection as a Hugging Face dataset in WORKDIR/export (FLAC in parquet, card, terms)
export: check-workdir
	$(STEN) export $(ARGS) --jobs $(JOBS)

## publish-dataset - upload the export private, verify it on the hub, then make it public (HUB_DIR: a durable local copy)
publish-dataset: check-workdir
	$(STEN) publish-dataset $(ARGS) $(if $(HUB_DIR),--hub-dir "$(HUB_DIR)")
