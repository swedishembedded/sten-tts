# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible training recipes for speech
# models for its clients. If your team needs expertise in speech training
# data, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Writer (and a reader for tests) of brain's `brain-tts-shard-v1` training file.

The format is brain's (crate qwen3tts, module shard); brain validates every
shard it is given, so this writer is checked by the model server, not trusted.
Layout: the 8-byte magic, a little-endian u64 header length, the JSON header
`{"meta": ..., "columns": {name: {"dtype", "offset", "len"}}}`, then the
columns, each 8-byte aligned from the end of the header.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"BTTSHRD1"
FORMAT = "brain-tts-shard-v1"
NO_VOICE = 0xFFFFFFFF
COLUMNS = [("text", "u32"), ("text_off", "u64"), ("codes", "u16"), ("codes_off", "u64"), ("xvec", "f32"),
           ("voice", "u32"), ("language", "u32"), ("complete", "u32"), ("ids", "u8")]
DTYPES = {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64, "f32": np.float32}


@dataclass
class ShardWriter:
    codebooks: int
    codebook_size: int
    xvector_dim: int
    frame_rate: float
    languages: list[str]
    voices: list[str]
    producers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._text: list[np.ndarray] = []
        self._codes: list[np.ndarray] = []
        self._xvec: list[np.ndarray] = []
        self._voice: list[int] = []
        self._language: list[int] = []
        self._complete: list[int] = []
        self._ids: list[str] = []

    def __len__(self) -> int:
        return len(self._voice)

    def push(self, *, id: dict, text_ids: np.ndarray, codes: np.ndarray, xvector: np.ndarray, voice: str | None, language: str, complete: bool) -> None:
        codes = np.asarray(codes, dtype=np.uint16).reshape(-1, self.codebooks)
        if len(codes) == 0 or int(codes.max()) >= self.codebook_size:
            raise ValueError(f"{id['id']}: no frames, or a code outside the codebook of {self.codebook_size}")
        xvector = np.asarray(xvector, dtype=np.float32)
        if xvector.shape != (self.xvector_dim,) or not np.all(np.isfinite(xvector)):
            raise ValueError(f"{id['id']}: x-vector of shape {xvector.shape}, expected ({self.xvector_dim},) and finite")
        text_ids = np.asarray(text_ids, dtype=np.uint32)
        if len(text_ids) == 0:
            raise ValueError(f"{id['id']}: no text ids")
        self._text.append(text_ids)
        self._codes.append(codes.reshape(-1))
        self._xvec.append(xvector)
        self._voice.append(NO_VOICE if voice is None else self.voices.index(voice))
        self._language.append(self.languages.index(language))
        self._complete.append(int(complete))
        self._ids.append(json.dumps(id, ensure_ascii=False, separators=(",", ":")))

    def to_bytes(self) -> bytes:
        n = len(self)
        text_off = np.concatenate([[0], np.cumsum([len(t) for t in self._text])]).astype(np.uint64)
        codes_off = np.concatenate([[0], np.cumsum([len(c) // self.codebooks for c in self._codes])]).astype(np.uint64)
        cols = {
            "text": np.concatenate(self._text) if n else np.zeros(0, np.uint32),
            "text_off": text_off,
            "codes": np.concatenate(self._codes) if n else np.zeros(0, np.uint16),
            "codes_off": codes_off,
            "xvec": np.concatenate(self._xvec) if n else np.zeros(0, np.float32),
            "voice": np.asarray(self._voice, np.uint32),
            "language": np.asarray(self._language, np.uint32),
            "complete": np.asarray(self._complete, np.uint32),
            "ids": np.frombuffer(("".join(i + "\n" for i in self._ids)).encode(), np.uint8),
        }
        body, columns = bytearray(), {}
        for name, dtype in COLUMNS:
            a = np.ascontiguousarray(cols[name], dtype=np.dtype(DTYPES[dtype]).newbyteorder("<"))
            columns[name] = {"dtype": dtype, "offset": len(body), "len": int(a.size)}
            body += a.tobytes()
            body += b"\0" * (-len(body) % 8)
        meta = {"format": FORMAT, "codebooks": self.codebooks, "codebook_size": self.codebook_size, "xvector_dim": self.xvector_dim,
                "frame_rate": self.frame_rate, "languages": self.languages, "voices": self.voices, "records": n, "producers": self.producers}
        header = json.dumps({"meta": meta, "columns": columns}, ensure_ascii=False).encode()
        return MAGIC + len(header).to_bytes(8, "little") + header + bytes(body)

    def write(self, path: Path) -> None:
        """Write atomically; an existing file is refused."""
        if path.exists():
            raise FileExistsError(path)
        tmp = path.with_suffix(path.suffix + ".partial")
        with open(tmp, "wb") as fh:
            fh.write(self.to_bytes())
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp, path)


def read(path: Path) -> dict:
    """Header and columns of a shard, for tests and reports."""
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: not a shard file")
    hlen = int.from_bytes(data[8:16], "little")
    header = json.loads(data[16 : 16 + hlen])
    body = memoryview(data)[16 + hlen :]
    cols = {}
    for name, c in header["columns"].items():
        dt = np.dtype(DTYPES[c["dtype"]]).newbyteorder("<")
        cols[name] = np.frombuffer(body, dtype=dt, count=c["len"], offset=c["offset"])
    return {"meta": header["meta"], **cols}
