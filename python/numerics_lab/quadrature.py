"""Quadrature node rules: points and weights on an interval, independent of any discretisation."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

__all__ = ['gauss_legendre_1d']


def gauss_legendre_1d(n: int, lo: float, hi: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``n``-point Gauss-Legendre points and weights mapped onto ``[lo, hi]``.

    Exact for polynomials up to degree ``2n - 1``; the test asserts the BOUNDARY case, since a rule
    checked only on a low-degree integrand cannot tell a correct rule from an over-generous one.
    """
    x, w = _reference_gauss(int(n))
    half = 0.5 * (hi - lo)
    return lo + half * (x + 1.0), half * w


@lru_cache(maxsize=64)
def _reference_gauss(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The nodes and weights on ``[-1, 1]``, which depend on ``n`` and on NOTHING ELSE.

    ``leggauss`` solves an eigenproblem, and the span it is mapped onto is a scaling applied after.
    Assembling one motor asked for the same three or four values of ``n`` **8660 times**: 2.455 s of
    a 9.101 s whole-motor assembly, 27% of it, spent recomputing identical arrays. The cache is on
    the reference rule rather than on the mapped one because the interval is different almost every
    call and would make every entry a miss.

    The arrays are marked read-only: a cached array a caller mutates in place is a defect that
    shows up somewhere else entirely.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    x.flags.writeable = False
    w.flags.writeable = False
    return x, w
