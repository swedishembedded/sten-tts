# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Several brain servers behind one pool: worker threads spread over them in turn."""

from __future__ import annotations

import threading
from collections import Counter

import pytest

from sten_tts.brainclient import BrainPool


def test_threads_are_spread_over_the_servers_and_addresses_kept_verbatim() -> None:
    made = []
    addrs = ["unix:path=/run/a,guid=1", "unix:path=/run/b,guid=2"]
    pool = BrainPool("dbus", addrs, factory=lambda transport, bus: made.append(bus) or bus)
    got = []
    threads = [threading.Thread(target=lambda: got.append(pool.get())) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert Counter(got) == {addrs[0]: 3, addrs[1]: 3}
    assert pool.first() == addrs[0], "the first server is the one training uses"
    assert BrainPool("dbus", ["session"], factory=lambda transport, bus: bus).get() == "SESSION"


def test_several_servers_need_dbus() -> None:
    with pytest.raises(ValueError, match="dbus"):
        BrainPool("jsonl", ["a", "b"])
