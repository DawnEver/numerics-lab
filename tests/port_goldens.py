"""The committed port goldens: one JSONL file per port (one record, one line), read by both the Python and the Rust tests.

A golden is generated from the Python by the test that registers its builder, and frozen in
`crates/numerics/tests/goldens/`; the Rust port judges itself against the same file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

__all__ = ['GOLDENS', 'REGISTRY', 'load_golden', 'register', 'write_golden']

#: Where every port golden lives.
GOLDENS = Path(__file__).resolve().parents[1] / 'crates' / 'numerics' / 'tests' / 'goldens'

#: golden path -> the builder that writes it; one writer per golden.
REGISTRY: dict[Path, Callable[[], dict]] = {}


def register(path: Path, build: Callable[[], dict]) -> Path:
    """Declare that ``build()`` produces the golden at ``path``; a second, different writer raises."""
    known = REGISTRY.get(path)
    if known is not None and known.__qualname__ != build.__qualname__:
        msg = f'{path.name} already has a builder ({known.__qualname__}); one golden, one writer'
        raise ValueError(msg)
    REGISTRY[path] = build
    return path


def load_golden(path: Path) -> dict:
    """Read a committed golden: exactly one JSON record on one line."""
    lines = path.read_text(encoding='utf-8').splitlines()
    if len(lines) != 1:
        msg = f'{path} holds {len(lines)} lines; a port golden is ONE JSONL record'
        raise ValueError(msg)
    return json.loads(lines[0])


def write_golden(path: Path, record: dict) -> None:
    """Freeze *record* as the golden at *path*: one JSON line."""
    path.write_text(json.dumps(record, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
