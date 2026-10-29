"""Equality-constrained quadratic systems: the KKT system, dense or sparse.

``[[H, A^T], [A, 0]] [x; y] = [c; d]`` -- the stationarity of ``1/2 x.H x - c.x`` subject to
``A x = d``, with ``y`` the multipliers (sign: ``H x + A^T y = c``). The matrix is symmetric and
INDEFINITE, so the dense path factors it by the symmetric indefinite solve and the sparse path by
the seam's ``direct`` solve under its residual guard (the bare registry backend
alone answers a singular KKT with garbage in about one draw in eight, measured 2026-10-18). A rank-deficient ``A`` or a ``H``
singular on the null space of ``A`` makes the system singular, and that is refused, never answered --
dense or sparse, with the same :class:`~numerics_lab.linear.errors.LinearSolveError`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, LinAlgWarning, solve

from numerics_lab.linear import LinearSolveError, solve_linear_system

__all__ = ['KKTSolution', 'solve_kkt']


@dataclass(frozen=True)
class KKTSolution:
    """The primal ``x`` and the multipliers ``y`` of ``A x = d``."""

    x: NDArray[np.float64]
    multipliers: NDArray[np.float64]


def solve_kkt(h, a, c: NDArray[np.float64], d: NDArray[np.float64]) -> KKTSolution:
    """Solve ``H x + A^T y = c``, ``A x = d``; sparse when ``h`` or ``a`` is a scipy sparse matrix, else dense."""
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    n, m = c.size, d.size
    if h.shape != (n, n) or a.shape != (m, n):
        msg = f'KKT shapes disagree: H {h.shape}, A {a.shape}, c ({n},), d ({m},)'
        raise ValueError(msg)
    rhs = np.concatenate([c, d])
    if sp.issparse(h) or sp.issparse(a):
        kkt = sp.bmat([[sp.csr_matrix(h), sp.csr_matrix(a).T], [sp.csr_matrix(a), None]], format='csr')
        solution = np.asarray(solve_linear_system(kkt, rhs, linear_solver='direct'), dtype=float)
    else:
        h = np.asarray(h, dtype=float)
        a = np.asarray(a, dtype=float)
        kkt = np.block([[h, a.T], [a, np.zeros((m, m))]])
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            try:
                solution = solve(kkt, rhs, assume_a='sym')
            except (LinAlgError, LinAlgWarning) as exc:
                msg = 'the KKT system is singular: A is rank deficient or H is singular on its null space'
                raise LinearSolveError(msg) from exc
    return KKTSolution(solution[:n], solution[n:])
