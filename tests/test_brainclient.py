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


def test_a_remote_server_is_reached_over_https_with_its_key(tmp_path, monkeypatch) -> None:
    import base64
    import json
    import urllib.request

    seen = []

    class _Resp:
        def __init__(self, body: dict) -> None:
            self.body = json.dumps(body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a) -> None:
            pass

        def read(self) -> bytes:
            return self.body

    def urlopen(req, timeout, context):
        seen.append(req)
        if req.full_url.endswith("/v1/capabilities"):
            return _Resp({"data": [{"model": "m", "action": "encode", "inputs": [{"name": "clips", "media": "bytes"}]}]})
        body = json.loads(req.data)
        assert body["blobs"]["clips"] == {"media": "bytes", "data": base64.b64encode(b"wav").decode(), "meta": {}}
        return _Resp({"outputs": {"frames": 3}, "blobs": {"codes": {"media": "bytes", "data": base64.b64encode(b"\x01\x00").decode(), "meta": {}}}})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    (tmp_path / "key").write_text("sekret\n")
    pool = BrainPool("dbus", ["https://node:8801"], key_file=tmp_path / "key")
    pool.require_model("m", "encode")
    out = pool.get().run("m", "encode", {}, blobs={"clips": b"wav"})
    assert out.outputs == {"frames": 3} and out.blobs["codes"] == b"\x01\x00"
    assert all(r.get_header("Authorization") == "Bearer sekret" for r in seen)
    with pytest.raises(ValueError, match="plaintext"):
        BrainPool("dbus", ["http://node:8801"], key_file=tmp_path / "key")
