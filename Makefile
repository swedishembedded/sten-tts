# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Every stage of the Sten-TTS-1.7B recipe. WORKDIR is required and holds all of
# a run's data (about 120 GB of sources plus shards, checkpoints and reports):
#
#   make all WORKDIR=/fast/disk/sten-run
#
# PYTHON is the interpreter of an environment with this package installed
# (`make env` makes one in .venv).

WORKDIR ?=
PYTHON ?= .venv/bin/python
RECIPE ?= recipe/sten-tts-1.7b.yaml
JOBS ?= 16
CONCURRENCY ?= 8
HUB_DIR ?=
ARTIFACTS ?=
UPSTREAM_PYTHON ?=
STEN := $(PYTHON) -m sten_tts.cli
ARGS = --workdir "$(WORKDIR)" --recipe "$(RECIPE)"

.PHONY: help env test all fetch inventory select audio shards train evaluate publish check-workdir

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
	@test -n "$(WORKDIR)" || { echo "set WORKDIR=<a directory with ~200 GB free>"; exit 2; }

## all - every stage in order
all: fetch inventory select audio shards train evaluate

## fetch - download the pinned source files (only the ones the recipe names)
fetch: check-workdir
	$(STEN) fetch $(ARGS)

## inventory - one table per source of every candidate utterance, without audio
inventory: check-workdir
	$(STEN) inventory $(ARGS) --jobs $(JOBS)

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
