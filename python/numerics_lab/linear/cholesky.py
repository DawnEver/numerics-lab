"""Jittered Cholesky: factor ``A + jitter I`` with the smallest jitter on a geometric ladder that succeeds.

A Gram or covariance matrix that is positive semidefinite in exact arithmetic is often not
factorable in floating point. The jitter is a MODIFICATION of the matrix, so it is reported, never
hidden: the caller judges whether the factor it got is the factor of the matrix it meant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, cho_solve, cholesky

from numerics_lab.linear.errors import LinearSolveError

__all__ = ['JitteredCholesky', 'jittered_cholesky']


@dataclass(frozen=True)
class JitteredCholesky:
    """The lower factor ``L`` with ``L L^T = A + jitter I``, and the jitter added (``0.0`` when none)."""

    lower: NDArray[np.float64]
    jitter: float

    def solve(self, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(A + jitter I)^{-1} rhs``."""
        return cho_solve((self.lower, True), rhs)


def jittered_cholesky(
    a: NDArray[np.float64], *, initial: float = 1e-10, growth: float = 10.0, max_tries: int = 10
) -> JitteredCholesky:
    """Factor ``a``; on failure retry with ``jitter = initial * mean(diag(a)) * growth**k``, ``k = 0, 1, ...``.

    The ladder is scaled by the mean diagonal so the jitter is relative to the matrix. Raises
    :class:`~numerics_lab.linear.errors.LinearSolveError` when the last rung still fails: a matrix that far from
    positive definite is not a rounding accident.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        msg = f'jittered_cholesky needs a square matrix, got shape {a.shape}'
        raise ValueError(msg)
    if not np.all(np.isfinite(a)):
        msg = 'jittered_cholesky needs a finite matrix'
        raise ValueError(msg)
    try:
        return JitteredCholesky(cholesky(a, lower=True), 0.0)
    except LinAlgError:
        pass
    scale = float(np.mean(np.abs(np.diag(a)))) or 1.0
    identity = np.eye(a.shape[0])
    jitter = initial * scale
    for _ in range(max_tries):
        try:
            return JitteredCholesky(cholesky(a + jitter * identity, lower=True), jitter)
        except LinAlgError:
            jitter *= growth
    msg = f'not positive definite even with jitter {jitter / growth:.3e} ({max_tries} rungs from {initial * scale:.3e})'
    raise LinearSolveError(msg)
