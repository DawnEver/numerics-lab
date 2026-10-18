"""Dense least-squares and triangular solves -- the one owner of both operations.

A caller that needs ``argmin ||A x - b||`` or ``T x = b`` with ``T`` triangular takes it from here,
never from LAPACK directly, so the failure it must catch is the seam's own
:class:`~numerics_lab.linear.errors.LinearSolveError` rather than a backend's exception type.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from numerics_lab.linear.errors import LinearSolveError

__all__ = ['least_squares', 'solve_triangular']


def _finite(name: str, *arrays: NDArray[np.float64]) -> None:
    if not all(np.all(np.isfinite(x)) for x in arrays):
        msg = f'{name} needs finite input'
        raise ValueError(msg)


def least_squares(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """The minimum-norm ``x`` minimising ``||a x - b||_2`` (SVD-based, so a rank-deficient ``a`` is answered).

    ``b`` may be a vector or a matrix of right-hand sides; ``x`` takes its trailing shape.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    _finite('least_squares', a, b)
    try:
        return scipy.linalg.lstsq(a, b)[0]
    except scipy.linalg.LinAlgError as exc:
        msg = f'least_squares: the SVD did not converge ({exc})'
        raise LinearSolveError(msg) from exc


def solve_triangular(t: NDArray[np.float64], b: NDArray[np.float64], *, lower: bool = True) -> NDArray[np.float64]:
    """``x`` with ``t x = b``, reading only the named triangle of ``t``; a zero pivot is refused."""
    t = np.asarray(t, dtype=float)
    b = np.asarray(b, dtype=float)
    _finite('solve_triangular', t, b)
    try:
        return scipy.linalg.solve_triangular(t, b, lower=lower)
    except scipy.linalg.LinAlgError as exc:
        msg = f'solve_triangular: the triangle is singular ({exc})'
        raise LinearSolveError(msg) from exc
