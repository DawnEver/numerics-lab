"""Factorization reuse for the time march -- physics-blind, roadmap W6.

An implicit step solves ``J du = -R`` with the weighted tangent ``J``. On a fixed grid the march
re-factors the SAME matrix thousands of times: a linear problem's tangent depends only on the
derivative weights (constant while ``dt`` is), and even a switched-linear problem's tangent moves
only at an event. Measured pre-W6 (2026-07-22, laplace ladder): rlc_series paid 600 dense LU
factorizations for 2 distinct matrices; buck_open_loop paid 190902 -- one per Newton solve, while
the matrix itself takes fewer than eleven thousand distinct values over the whole march (the
locator's bisection probes each carry a unique ``dt``, and those are irreducible).

:class:`FactorCache` removes the repeated factorization WITHOUT any physics knowledge. The deal,
kept entirely on the problem's side of the seam:

* The problem MAY expose ``factor_reuse_key(u, u_dot, u_ddot, t, weights) -> key`` where ``key``
  is a value (a tuple of numbers, say) such that **equal keys imply an equal jacobian**. Deriving
  that implication is the problem's -- laplace's ``CircuitDAE`` does it by re-reading the weights
  plus every state-dependent device's tangent stamps, so a ``dt`` change, a switch toggle, or a
  state-dependent tangent each move the key unprompted. gauss never inspects what the key means.
* While the key holds, the cache back-substitutes the last factorization (dense ``lu_solve`` /
  sparse ``splu.solve`` / scalar divide). When it moves, the matrix is built (once -- the build is
  deferred so a hit never pays it), re-factored, and cached.

The counters ARE the verification surface: ``factorizations`` and ``solves`` are what a test pins
(a wall-clock is a fact about a moment; a key count is not). Bit-identity is not gambled on:
``lu_factor`` + ``lu_solve`` is the same LAPACK ``getrf``/``getrs`` pair ``numpy.linalg.solve``
calls (measured bit-equal on 200 random 5x5 systems), and ``splu`` is the SuperLU call ``spsolve``
makes -- so a hit returns the same bits the uncached solve would have.

NAMED HOLE: the cache trusts the key exactly as it trusts ``residual`` / ``jacobian``. A key that
under-reads the jacobian's inputs (say, one that ignores a device state the tangent depends on)
reuses a STALE factor and silently solves the wrong system. That failure class is pinned on the
problem's side (``tests/unit/laplace/solver/test_factor_reuse.py``'s toggle-is-a-new-key anchor), and a
problem that cannot derive the implication should not expose the seam at all -- ``None`` keys and
keyless problems always take the plain per-solve path, which is what every pre-W6 problem does.

This module imports only contracts + numpy/scipy and touches no physics, like the rest of gauss.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu

__all__ = ['FactorCache']

#: Sentinel for 'no factorization cached yet' -- distinct from every legitimate key, None included.
_EMPTY = object()


class FactorCache:
    """One factorization per reuse key: the solve seam the march consults before re-factoring.

    Counters (the machine-independent record a test pins):

    * ``solves`` -- every ``solve(...)`` call, hit or miss;
    * ``factorizations`` -- the misses (a matrix was actually built and factored);
    * ``reuses`` -- the hits (back-substitution only). ``solves == factorizations + reuses``
      always, by construction.

    The cache holds exactly ONE factorization: the march's working set is the current step form,
    so a deeper cache would buy nothing -- a key that returns after leaving (an event rollback
    re-stepping an earlier ``dt``) simply re-factors.
    """

    def __init__(self) -> None:
        self.solves = 0
        self.factorizations = 0
        self._key: Hashable | object = _EMPTY
        self._factor: object = None

    @property
    def reuses(self) -> int:
        """Solves answered by back-substitution against the cached factorization."""
        return self.solves - self.factorizations

    def solve(
        self,
        key: Hashable | None,
        build_matrix: Callable[[], object],
        rhs: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Solve ``matrix @ x = rhs``, reusing the cached factorization while ``key`` holds.

        ``key`` is the problem's derived proof that the matrix is unchanged (see the module
        docstring for who owes that proof); ``None`` means 'cannot prove' and never caches.
        ``build_matrix`` is called ONLY on a miss -- a hit never pays the jacobian build.
        """
        self.solves += 1
        b = np.asarray(rhs, dtype=float)
        if key is not None and key == self._key:
            return np.asarray(self._apply(self._factor, b), dtype=float)
        matrix = build_matrix()
        self._factor = self._factor_of(matrix)
        if key is not None:
            self._key = key
        self.factorizations += 1
        return np.asarray(self._apply(self._factor, b), dtype=float)

    @staticmethod
    def _factor_of(matrix: object) -> object:
        """Factor ``matrix`` in its own storage; mirrors ``_default_solve``'s dispatch."""
        if sp.issparse(matrix):
            return splu(sp.csc_matrix(matrix))
        m = np.asarray(matrix, dtype=float)
        if m.ndim == 0:  # scalar jacobian (1-DOF problem written as a plain number)
            return float(m)
        return lu_factor(m)

    @staticmethod
    def _apply(factor: object, b: NDArray[np.floating]) -> NDArray[np.floating]:
        """Back-substitute a factor produced by :meth:`_factor_of`."""
        if isinstance(factor, float):
            return b / factor
        if isinstance(factor, tuple):  # dense (lu, piv)
            return lu_solve(factor, b)
        return factor.solve(b)  # sparse SuperLU factor
