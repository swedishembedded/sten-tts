#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Run the recipe's model stages against brain servers of their own, each on a
# private D-Bus bus, and stop them afterwards.
#
#   scripts/with-brain.sh --brain BIN --workdir DIR [--recipe FILE] [--device SPEC ...]
#                         [--brain-data-dir DIR] [--python PY] [--upstream-python PY]
#                         [--allow-cpu]         # let the fine-tune run on the CPU backend (smoke runs)
#                         [--concurrency N]     # requests in flight to the servers (lower it on a CPU)
#                         [stage ...]          # default: shards train evaluate
#
# Every --device starts one server pinned to SPEC (gpu0, gpu0,cpu, ...); without
# one, a single server lets brain choose. Name a card twice for two servers on
# it, which helps a model that serves one request at a time (the recogniser in
# `segment`). The first server runs served training with its artifact store in
# DIR/artifacts and is the one `train` uses; the other stages spread their
# requests over all of them. After `train` the servers are restarted, because a
# model it installs is served from the next start. Server logs go to
# DIR/brain-serve-N.log.
set -euo pipefail

brain= workdir= recipe=recipe/sten-tts-1.7b.yaml data_dir= python=.venv/bin/python upstream= allow_cpu= concurrency= stages=() devices=()
while [ $# -gt 0 ]; do
    case "$1" in
        --brain) brain=$2; shift 2 ;;
        --workdir) workdir=$2; shift 2 ;;
        --recipe) recipe=$2; shift 2 ;;
        --device) devices+=("$2"); shift 2 ;;
        --brain-data-dir) data_dir=$2; shift 2 ;;
        --python) python=$2; shift 2 ;;
        --upstream-python) upstream=$2; shift 2 ;;
        --allow-cpu) allow_cpu=1; shift ;;
        --concurrency) concurrency=$2; shift 2 ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        -*) echo "with-brain: unknown flag $1" >&2; exit 2 ;;
        *) stages+=("$1"); shift ;;
    esac
done
[ -n "$brain" ] && [ -x "$brain" ] || { echo "with-brain: --brain must name the brain binary" >&2; exit 2; }
[ -n "$workdir" ] || { echo "with-brain: --workdir is required" >&2; exit 2; }
[ ${#stages[@]} -gt 0 ] || stages=(shards train evaluate)
[ ${#devices[@]} -gt 0 ] || devices=("")
command -v dbus-daemon > /dev/null || { echo "with-brain: dbus-daemon is not installed" >&2; exit 2; }

mkdir -p "$workdir/artifacts"
busdir=$(mktemp -d "$workdir/.bus.XXXXXX")   # mode 700: the buses are this user's only
buses=() daemons=() pids=()

for i in "${!devices[@]}"; do
    out=$(dbus-daemon --session --fork --address="unix:dir=$busdir" --print-address=1 --print-pid=1 2> /dev/null)
    buses+=("$(sed -n 1p <<< "$out")")
    daemons+=("$(sed -n 2p <<< "$out")")
done

serve_args() {
    local i=$1 args=(serve --dbus)
    [ "$i" = 0 ] && args+=(--training --artifact-store "$workdir/artifacts")
    [ -n "${devices[$i]}" ] && args+=(--device "${devices[$i]}")
    [ -n "$data_dir" ] && args=(--brain-data-dir "$data_dir" "${args[@]}")
    printf '%s\n' "${args[@]}"
}

start() {
    pids=()
    for i in "${!buses[@]}"; do
        mapfile -t args < <(serve_args "$i")
        DBUS_SESSION_BUS_ADDRESS=${buses[$i]} "$brain" "${args[@]}" >> "$workdir/brain-serve-$i.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!buses[@]}"; do
        for n in $(seq 601); do
            [ "$n" = 601 ] && { echo "with-brain: server $i did not answer within 10 minutes" >&2; exit 1; }
            if "$python" -c "import sys; from brain_py import Brain; Brain(bus=sys.argv[1]).models()" "${buses[$i]}" > /dev/null 2>&1; then break; fi
            kill -0 "${pids[$i]}" 2> /dev/null || { echo "with-brain: server $i exited; see $workdir/brain-serve-$i.log" >&2; exit 1; }
            sleep 1
        done
    done
}
stop() {
    for p in "${pids[@]}"; do kill "$p" 2> /dev/null || true; done
    for p in "${pids[@]}"; do wait "$p" 2> /dev/null || true; done
    pids=()
}
cleanup() {
    stop
    for d in "${daemons[@]}"; do kill "$d" 2> /dev/null || true; done
    rm -rf "$busdir"
}
trap cleanup EXIT

bus_flags=()
for b in "${buses[@]}"; do bus_flags+=(--brain-bus "$b"); done

start
for s in "${stages[@]}"; do
    extra=()
    case "$s" in
        train) extra=(--artifact-store "$workdir/artifacts"); [ -n "$allow_cpu" ] && extra+=(--allow-cpu) ;;
        evaluate) [ -n "$upstream" ] && extra=(--upstream-python "$upstream") ;;
    esac
    case "$s" in shards|evaluate|segment) [ -n "$concurrency" ] && extra+=(--concurrency "$concurrency") ;; esac
    "$python" -m sten_tts.cli "$s" --workdir "$workdir" --recipe "$recipe" "${bus_flags[@]}" "${extra[@]}"
    if [ "$s" = train ]; then stop; start; fi
done
