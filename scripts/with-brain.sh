#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Run the recipe's model stages against a brain server of their own, in a
# private D-Bus session, and stop the server afterwards.
#
#   scripts/with-brain.sh --brain BIN --workdir DIR [--recipe FILE] [--device SPEC]
#                         [--brain-data-dir DIR] [--python PY] [--upstream-python PY]
#                         [--allow-cpu]         # let the fine-tune run on the CPU backend (smoke runs)
#                         [--concurrency N]     # requests in flight to the server (lower it on a CPU)
#                         [stage ...]          # default: shards train evaluate
#
# The server runs with served training and its artifact store in
# DIR/artifacts. After `train` it is restarted, because a model it installs is
# served from the next start. Server logs go to DIR/brain-serve.log.
set -euo pipefail

brain= workdir= recipe=recipe/sten-tts-1.7b.yaml device= data_dir= python=.venv/bin/python upstream= allow_cpu= concurrency= stages=()
while [ $# -gt 0 ]; do
    case "$1" in
        --brain) brain=$2; shift 2 ;;
        --workdir) workdir=$2; shift 2 ;;
        --recipe) recipe=$2; shift 2 ;;
        --device) device=$2; shift 2 ;;
        --brain-data-dir) data_dir=$2; shift 2 ;;
        --python) python=$2; shift 2 ;;
        --upstream-python) upstream=$2; shift 2 ;;
        --allow-cpu) allow_cpu=1; shift ;;
        --concurrency) concurrency=$2; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        -*) echo "with-brain: unknown flag $1" >&2; exit 2 ;;
        *) stages+=("$1"); shift ;;
    esac
done
[ -n "$brain" ] && [ -x "$brain" ] || { echo "with-brain: --brain must name the brain binary" >&2; exit 2; }
[ -n "$workdir" ] || { echo "with-brain: --workdir is required" >&2; exit 2; }
[ ${#stages[@]} -gt 0 ] || stages=(shards train evaluate)

if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ] || [ "${WITH_BRAIN_PRIVATE_BUS:-}" != 1 ]; then
    # A private session bus, so the server and the client find each other and nothing else does.
    exec env WITH_BRAIN_PRIVATE_BUS=1 dbus-run-session -- "$0" --brain "$brain" --workdir "$workdir" --recipe "$recipe" \
        ${device:+--device "$device"} ${data_dir:+--brain-data-dir "$data_dir"} --python "$python" \
        ${upstream:+--upstream-python "$upstream"} ${allow_cpu:+--allow-cpu} ${concurrency:+--concurrency "$concurrency"} "${stages[@]}"
fi

mkdir -p "$workdir/artifacts"
serve_args=(serve --dbus --training --artifact-store "$workdir/artifacts")
[ -n "$device" ] && serve_args+=(--device "$device")
[ -n "$data_dir" ] && serve_args=(--brain-data-dir "$data_dir" "${serve_args[@]}")
pid=

start() {
    "$brain" "${serve_args[@]}" >> "$workdir/brain-serve.log" 2>&1 &
    pid=$!
    for _ in $(seq 600); do
        if "$python" -c "from brain_py import Brain; b = Brain(); b.models()" > /dev/null 2>&1; then return 0; fi
        kill -0 "$pid" 2> /dev/null || { echo "with-brain: the server exited; see $workdir/brain-serve.log" >&2; exit 1; }
        sleep 1
    done
    echo "with-brain: the server did not answer within 10 minutes" >&2
    exit 1
}
stop() { [ -n "$pid" ] && kill "$pid" 2> /dev/null && wait "$pid" 2> /dev/null || true; pid=; }
trap stop EXIT

start
for s in "${stages[@]}"; do
    extra=()
    case "$s" in
        train) extra=(--artifact-store "$workdir/artifacts"); [ -n "$allow_cpu" ] && extra+=(--allow-cpu) ;;
        evaluate) [ -n "$upstream" ] && extra=(--upstream-python "$upstream") ;;
    esac
    case "$s" in shards|evaluate) [ -n "$concurrency" ] && extra+=(--concurrency "$concurrency") ;; esac
    "$python" -m sten_tts.cli "$s" --workdir "$workdir" --recipe "$recipe" "${extra[@]}"
    if [ "$s" = train ]; then stop; start; fi
done
