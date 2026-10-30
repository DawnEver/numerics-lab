"""Equality-constrained quadratic systems: the KKT system, dense or sparse.

``[[H, A^T], [A, 0]] [x; y] = [c; d]`` -- the stationarity of ``1/2 x.H x - c.x`` subject to
``A x = d``, with ``y`` the multipliers (sign: ``H x + A^T y = c``). The matrix is symmetric and
INDEFINITE, so the dense path factors it by the symmetric indefinite solve and the sparse path by
SuperLU, whose pivots it judges. A rank-deficient ``A`` or a ``H`` singular on the null space of ``A``
makes the system singular, and that is refused, never answered -- dense or sparse, with the same
:class:`~numerics_lab.linear.errors.LinearSolveError`.

THE SPARSE REFUSAL IS THE FACTORISATION'S PIVOT BAR, not a residual (2026-10-30). Once
``cond * eps >= 1`` floating point cannot tell a singular system from an ill-conditioned one, so a
residual guard answered such draws platform by platform: 47 of 1200 repeated-row draws perturbed by
1e-15 passed it on x86 with ``|x|`` up to 3.6e17, and the exact repeated row passed it on macOS. LU with
partial pivoting is backward stable, so it factors ``K + E`` with ``|E| <~ n eps |K|``; a pivot inside
``n * eps * ||K||_1`` is one that perturbation of the matrix can zero, i.e. ``K`` is singular to working
precision, and it is refused on every platform. Measured on those draws: the largest singular pivot
sits at 0.04 of the bar, the smallest pivot of a regular draw at 6.6e11 of it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, LinAlgWarning, solve
from scipy.sparse.linalg import splu

from numerics_lab.linear import LinearSolveError

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
        kkt = sp.bmat([[sp.csr_matrix(h), sp.csr_matrix(a).T], [sp.csr_matrix(a), None]], format='csc')
        solution = _pivot_barred_solve(kkt, rhs)
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


def _pivot_barred_solve(kkt: sp.csc_matrix, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    """``kkt^{-1} rhs`` by SuperLU, refused when a pivot is inside ``n * eps * ||kkt||_1`` (module docstring)."""
    singular = 'the KKT system is singular: A is rank deficient or H is singular on its null space'
    try:
        factor = splu(kkt)
    except RuntimeError as exc:  # SuperLU reports an exactly zero pivot this way
        msg = f'{singular} (an exactly zero pivot)'
        raise LinearSolveError(msg) from exc
    pivot = float(np.min(np.abs(factor.U.diagonal())))
    bar = kkt.shape[0] * np.finfo(float).eps * sp.linalg.norm(kkt, 1)
    if not pivot > bar:
        msg = f'{singular} (smallest pivot {pivot:.3e} within the bar n*eps*||K||_1 = {bar:.3e})'
        raise LinearSolveError(msg)
    return np.asarray(factor.solve(rhs), dtype=float)
