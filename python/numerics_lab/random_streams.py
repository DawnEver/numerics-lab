"""Named, reproducible random streams: ``Stream(seed, path)`` and its named children.

A stream is identified by its root seed and a PATH of names (``('population', 'mutation')``), never
by the order in which streams were created or drawn from. Two runs that ask for the same
``(seed, path)`` get the same numbers, whatever else they drew meanwhile, and adding a new child
stream never shifts an existing one. The path is hashed (SHA-256 over an unambiguous encoding) into
the entropy of a :class:`numpy.random.SeedSequence`, so ``('ab',)`` and ``('a', 'b')`` differ.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

__all__ = ['Stream']


@dataclass(frozen=True)
class Stream:
    """A reproducible random stream at ``path`` under the root ``seed``."""

    seed: int
    path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            msg = f'a stream seed is a nonnegative int, got {self.seed!r}'
            raise ValueError(msg)
        if not isinstance(self.path, tuple) or not all(isinstance(name, str) and name for name in self.path):
            msg = f'a stream path is a tuple of nonempty names, got {self.path!r}'
            raise ValueError(msg)

    def child(self, name: str) -> Stream:
        """The child stream ``name`` below this one."""
        return Stream(self.seed, (*self.path, name))

    def generator(self) -> np.random.Generator:
        """A fresh generator positioned at this stream's start (PCG64)."""
        digest = hashlib.sha256()
        for name in self.path:
            encoded = name.encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
        words = [int.from_bytes(digest.digest()[i : i + 4], 'little') for i in range(0, 32, 4)]
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, *words])))
