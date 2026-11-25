# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in training
# text-to-speech models, you can procure our services by sending an email to
# info@swedishembedded.com.

"""The brain servers the model stages talk to, through brain-py.

One connection per worker thread: the D-Bus client is not shared between
threads. A stage can drive several servers (one per card, or several on one
card for a model that serves one request at a time): each worker thread is
bound to one of them in turn, so the load spreads evenly. A server is a D-Bus
bus (`session`, `system` or an address `unix:path=...`, for a server on this
machine) or an `https://host:port` URL of a server on another machine, reached
through brain's generic action route (`POST /v1/run`) with an API key and the
server's certificate. The servers must already run; nothing here starts or
configures them.
"""

from __future__ import annotations

import base64
import itertools
import json
import ssl
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _dbus_bus(bus: str) -> str:
    """brain-py's (jeepney's) name for a bus: SESSION, SYSTEM, or the address unchanged."""
    return bus.upper() if bus in ("session", "system") else bus


def _http_client(url: str, key: str, context: ssl.SSLContext) -> Any:
    """A brain-py client of a server's HTTP surface: brain-py's capability API (encode_audio_batch, ...)
    over `GET /v1/capabilities` and `POST /v1/run`, so a stage need not know the transport."""
    from brain_py import BrainBase, BrainError, Outcome

    class HttpBrain(BrainBase):
        def __init__(self) -> None:
            self._caps: list[dict] | None = None

        def _call(self, method: str, path: str, body: dict | None, timeout: float) -> dict:
            req = urllib.request.Request(url.rstrip("/") + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout, context=context) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                raise BrainError(f"{url}{path}: HTTP {e.code}: {e.read()[:2000].decode('utf-8', 'replace')}") from None

        def _capabilities(self) -> list[dict]:
            if self._caps is None:
                self._caps = self._call("GET", "/v1/capabilities", None, 60)["data"]
            return self._caps

        def manifests(self) -> list[dict]:
            models: dict[str, list[dict]] = {}
            for a in self._capabilities():
                models.setdefault(a["model"], []).append({"name": a["action"]})
            return [{"model": m, "actions": acts} for m, acts in models.items()]

        def run(self, model, action, params=None, *, blobs=None, meta=None, timeout=1800.0):
            spec = next((a for a in self._capabilities() if a["model"] == model and a["action"] == action), None)
            if spec is None:
                raise BrainError(f"{url} serves no {model} action {action}")
            media = {i["name"]: i["media"] for i in spec["inputs"]}
            wire = {name: {"media": media.get(name, "bytes"), "data": base64.b64encode(data).decode(), "meta": (meta or {}).get(name, {})}
                    for name, data in (blobs or {}).items()}
            out = self._call("POST", "/v1/run", {"model": model, "action": action, "params": params or {}, "blobs": wire}, timeout)
            return Outcome(outputs=out.get("outputs", {}), blobs={n: base64.b64decode(b["data"]) for n, b in out.get("blobs", {}).items()},
                           meta={n: {"media": b.get("media"), "meta": b.get("meta")} for n, b in out.get("blobs", {}).items()})

    return HttpBrain()


@dataclass
class BrainPool:
    transport: str
    buses: list[str]
    factory: Callable[..., Any] | None = None  # brain_py.Brain; injectable for tests
    key_file: Path | None = None  # the API key of the https:// servers
    ca_file: Path | None = None  # the certificate (or CA) they present
    _local: threading.local = field(default_factory=threading.local)
    _turn: itertools.count = field(default_factory=itertools.count)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not self.buses:
            raise ValueError("BrainPool needs at least one bus")
        if self.transport != "dbus" and len(self.buses) > 1:
            raise ValueError(f"{len(self.buses)} servers need the dbus transport, not {self.transport}")
        if any(b.startswith("http://") for b in self.buses):
            raise ValueError("a remote brain server is reached over https://, never plaintext")
        if any(b.startswith("https://") for b in self.buses) and self.key_file is None:
            raise ValueError("an https:// brain server needs --brain-key-file")

    def _connect(self, bus: str) -> Any:
        if bus.startswith("https://"):
            context = ssl.create_default_context(cafile=str(self.ca_file) if self.ca_file else None)
            return _http_client(bus, self.key_file.read_text().strip(), context)
        factory = self.factory
        if factory is None:
            from brain_py import Brain
            factory = Brain
        return factory(transport=self.transport, bus=_dbus_bus(bus)) if self.transport == "dbus" else factory(transport=self.transport)

    def get(self) -> Any:
        """This thread's connection; a new thread is bound to the next server in turn."""
        brain = getattr(self._local, "brain", None)
        if brain is None:
            with self._lock:
                bus = self.buses[next(self._turn) % len(self.buses)]
            brain = self._local.brain = self._connect(bus)
        return brain

    def first(self) -> Any:
        """A connection to the first server, for a job that must run on exactly one (training)."""
        brain = getattr(self._local, "first", None)
        if brain is None:
            brain = self._local.first = self._connect(self.buses[0])
        return brain

    def require_model(self, model: str, action: str) -> None:
        """Refuse early, naming what is missing, when a server lacks `model` or it lacks `action`."""
        for bus in self.buses:
            brain = self._connect(bus)
            models = brain.models()
            if model not in models:
                raise RuntimeError(f"the brain server on {bus} does not serve {model!r} (it serves {sorted(models)}); `brain pull` it and restart the server")
            if action not in brain.actions(model):
                raise RuntimeError(f"{model} on {bus} has no action {action!r}; this brain is older than the recipe needs")


def connect(transport: str, buses: list[str], key_file: Path | None = None, ca_file: Path | None = None) -> BrainPool:
    try:
        import brain_py  # noqa: F401
    except ImportError as e:
        raise RuntimeError("brain-py is not installed: pip install -e <brain>/brain-py") from e
    return BrainPool(transport=transport, buses=list(buses), key_file=key_file, ca_file=ca_file)
