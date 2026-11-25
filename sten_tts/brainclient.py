# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in training
# text-to-speech models, you can procure our services by sending an email to
# info@swedishembedded.com.

"""The brain server the model stages talk to, through brain-py.

One connection per worker thread: the D-Bus client is not shared between
threads. The server must already run (`brain serve --dbus --training ...`);
nothing here starts or configures it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BrainPool:
    transport: str
    bus: str
    _local: threading.local = field(default_factory=threading.local)

    def get(self) -> Any:
        brain = getattr(self._local, "brain", None)
        if brain is None:
            from brain_py import Brain
            kwargs = {"bus": self.bus.upper()} if self.transport == "dbus" else {}
            brain = Brain(transport=self.transport, **kwargs)
            self._local.brain = brain
        return brain

    def require_model(self, model: str, action: str) -> None:
        """Refuse early, naming what is missing, when the server lacks `model` or it lacks `action`."""
        brain = self.get()
        models = brain.models()
        if model not in models:
            raise RuntimeError(f"the brain server does not serve {model!r} (it serves {sorted(models)}); `brain pull` it and restart the server")
        if action not in brain.actions(model):
            raise RuntimeError(f"{model} has no action {action!r}; this brain is older than the recipe needs")


def connect(transport: str, bus: str) -> BrainPool:
    try:
        import brain_py  # noqa: F401
    except ImportError as e:
        raise RuntimeError("brain-py is not installed: pip install -e <brain>/brain-py") from e
    return BrainPool(transport=transport, bus=bus)
