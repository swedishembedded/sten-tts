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
bound to one of them in turn, so the load spreads evenly. A bus is `session`,
`system` or a D-Bus address (`unix:path=...`). The servers must already run
(`brain serve --dbus ...`); nothing here starts or configures them.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


def _dbus_bus(bus: str) -> str:
    """brain-py's (jeepney's) name for a bus: SESSION, SYSTEM, or the address unchanged."""
    return bus.upper() if bus in ("session", "system") else bus


@dataclass
class BrainPool:
    transport: str
    buses: list[str]
    factory: Callable[..., Any] | None = None  # brain_py.Brain; injectable for tests
    _local: threading.local = field(default_factory=threading.local)
    _turn: itertools.count = field(default_factory=itertools.count)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not self.buses:
            raise ValueError("BrainPool needs at least one bus")
        if self.transport != "dbus" and len(self.buses) > 1:
            raise ValueError(f"{len(self.buses)} servers need the dbus transport, not {self.transport}")

    def _connect(self, bus: str) -> Any:
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


def connect(transport: str, buses: list[str]) -> BrainPool:
    try:
        import brain_py  # noqa: F401
    except ImportError as e:
        raise RuntimeError("brain-py is not installed: pip install -e <brain>/brain-py") from e
    return BrainPool(transport=transport, buses=list(buses))
