"""Factorization reuse for the time march -- physics-blind, roadmap W6.

An implicit step solves ``J du = -R`` with the weighted tangent ``J``. On a fixed grid the march
re-factors the SAME matrix thousands of times: a linear problem's tangent depends only on the
derivative weights (constant while ``dt`` is), and even a switched-linear problem's tangent moves
only at an event. Measured pre-W6 (2026-07-22, laplace ladder): rlc_series paid 600 dense LU
factorizations for 2 distinct matrices; `buck_stage`'s `open_loop` point paid 190902 -- one per Newton solve, while
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

WHAT A HEALTHY HIT *IS* JUDGED BY, added 2026-09-06: FINITENESS, via
:func:`~numerics_lab.linear.reject_nonfinite_solution`. A hit used to return back-substitution
output entirely unchecked, so an inf/NaN answer from a reused factor left this class silently --
the exact shape ``invariant.md`` forbids ("never silently wrong"). The reason it was missing is
that the guard's cheap half had no public spelling: it was ``gauss.linear._check_finite``, private
and reached from one line. THE PRICE IS A RATIO, not a value: ``O(n)`` against a back-substitution
already paid at ``O(n^2)`` dense / ``O(nnz)`` sparse, which is why this one is affordable on a hit
where the residual guard below is not.

WHAT A HEALTHY HIT IS STILL NOT JUDGED BY, AND WHAT THAT COSTS. A MISS holds the matrix it just
built, so it repairs and judges the answer against it exactly as ``gauss.linear``'s ``direct`` does.
If that first answer NEEDS repair, later hits through that factor are repaired and judged too;
only a factor whose miss passed unchanged takes the cheap finiteness-only hit path. A healthy HIT
does not rebuild the matrix, and that is the whole point of a hit -- so it is NOT residual-judged,
deliberately and at a measured price.
MEASURED 2026-09-06 through the production CLI over seven ``cases/network/circuit`` nodes:
595,898 solves, of which 211,388 (35%) were hits, and 123,208 of ``class_e``'s 129,749 (95%).
Judging a hit means calling ``build_matrix()`` for it, i.e. reinstating on a third of every
circuit march exactly the work this class exists to remove; that is not a guard, it is a deletion
of the feature. What makes the gap narrow rather than convenient is that the cache starts EMPTY:
the first solve of any march is a miss, so a system that is singular from the start is refused
before any hit can back-substitute against it. Pinned in
``tests/unit/test_a_singular_march_is_refused_where_it_is_factored.py``. What is NOT covered
is a matrix that becomes singular WITHOUT moving its key -- which is the NAMED HOLE below in a
different costume, and has the same owner: the key.

NAMED HOLE: the cache trusts the key exactly as it trusts ``residual`` / ``jacobian``. A key that
under-reads the jacobian's inputs (say, one that ignores a device state the tangent depends on)
reuses a STALE factor and silently solves the wrong system. That failure class is pinned on the
problem's side (``tests/unit/laplace/solver/test_circuit_factor_reuse_key.py``'s toggle-is-a-new-key anchor), and a
problem that cannot derive the implication should not expose the seam at all -- ``None`` keys and
keyless problems always take the plain per-solve path, which is what every pre-W6 problem does.

This module imports only numpy/scipy and touches no physics, like the rest of gauss.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.linalg.lapack import dgetrf, dgetrs
from scipy.sparse.linalg import splu

from numerics_lab.linear import reject_nonfinite_solution, settle_direct_solution
from numerics_lab.linear.native_direct import native_factor

__all__ = ['FactorCache', 'apply_factor', 'factor_matrix']

#: The name the guard's refusal message reports, so a reader lands on THIS seam rather than on
#: ``gauss.linear``'s ``direct``, which never ran.
_BACKEND = 'factor_reuse'

#: Sentinel for 'no factorization cached yet' -- distinct from every legitimate key, None included.
_EMPTY = object()


def factor_matrix(matrix: object) -> object:
    """Factor ``matrix`` in its own storage; mirrors ``_default_solve``'s dispatch.

    THE ONE HOME for "factor this operator however it is stored", shared by :class:`FactorCache`
    and by the bordered field<->circuit tangent
    (a downstream stacked field-circuit ``bordered_solve``), which
    factors its sparse field block ONCE and back-substitutes it for several right-hand sides. A
    second spelling of this dispatch is a second answer to "is a scalar jacobian a matrix", which
    is exactly the branch below.
    """
    if sp.issparse(matrix):
        # The native factor first, with ``splu``'s ``solve(b)`` shape; ``None`` (logged and
        # counted by ``native_direct``) when the extension cannot serve this operator.
        native = native_factor(matrix)
        return native if native is not None else splu(sp.csc_matrix(matrix))
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 0:  # scalar jacobian (1-DOF problem written as a plain number)
        return float(m)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.isfinite(m).all():
        return lu_factor(m)  # its own refusal names the defect
    # THE SAME `getrf` `lu_factor` calls, minus its per-call batching/dispatch wrapper: on a march
    # of tiny dense systems that wrapper cost more than the factorization (measured 2026-10-12,
    # 93,007 misses of an n=12 tangent: 2.5 s of wrapper around LAPACK).
    lu, piv, info = dgetrf(m)
    if info > 0:
        warnings.warn(f'Diagonal number {info} is exactly zero. Singular matrix.', LinAlgWarning, stacklevel=2)
    return lu, piv


def apply_factor(factor: object, b: NDArray[np.floating]) -> NDArray[np.floating]:
    """Back-substitute a factor produced by :func:`factor_matrix` against ``b``.

    ``b`` may be a vector or a MATRIX of right-hand sides (the bordered solve hands it the ``n2``
    columns of the off-diagonal block); all three storages accept both.
    """
    if isinstance(factor, float):
        return b / factor
    if isinstance(factor, tuple):  # dense (lu, piv)
        lu, piv = factor
        rhs = np.asarray(b, dtype=float)
        if not np.isfinite(rhs).all() or rhs.shape[0] != lu.shape[0]:
            return lu_solve(factor, b)  # its own refusal names the defect
        x, info = dgetrs(lu, piv, rhs)
        if info != 0:
            msg = f'illegal value in {-info}-th argument of internal getrs (lapack)'
            raise ValueError(msg)
        return x
    return factor.solve(b)  # sparse factor: native (faer) or SuperLU, both `solve(b)`-shaped


class FactorCache:
    """One factorization per reuse key: the solve seam the march consults before re-factoring.

    Counters (the machine-independent record a test pins):

    * ``solves`` -- every ``solve(...)`` call, hit or miss;
    * ``factorizations`` -- the misses (a matrix was actually built and factored);
    * ``reuses`` -- the hits (back-substitution only). ``solves == factorizations + reuses``
      always, by construction;
    * ``judged`` -- the misses whose answer was put to the residual guard. It is BELOW
      ``factorizations`` by exactly the scalar-jacobian misses, which the direct guard's statistic
      cannot speak about; quoting ``factorizations`` as the guarded count would overstate it.

    The cache holds exactly ONE factorization: the march's working set is the current step form,
    so a deeper cache would buy nothing -- a key that returns after leaving (an event rollback
    re-stepping an earlier ``dt``) simply re-factors.
    """

    def __init__(self) -> None:
        self.solves = 0
        self.factorizations = 0
        self.judged = 0
        #: The CACHED matrix itself, kept (not rebuilt) so the marcher can compute
        #: :func:`~numerics_lab.time_integrators.residual_floor` against the system a HIT just
        #: solved. Holding one reference costs what holding one factorization already costs; the
        #: thing a hit exists to avoid is BUILDING the matrix, and this never builds one.
        self.matrix: object = None
        self._key: Hashable | object = _EMPTY
        self._factor: object = None
        self._repair_hits = False

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
            # THE HALF OF THE GUARD A HIT CAN AFFORD. The residual guard below needs the matrix and
            # a hit exists precisely so it is never built; finiteness needs only the answer, at
            # O(n) against a back-substitution already paid at O(n^2)/O(nnz). A stale-but-FINITE
            # factor is still the key's problem (the NAMED HOLE above) -- this closes the inf/NaN
            # disguise, which was returning silently, and claims nothing about the other one.
            x = reject_nonfinite_solution(np.asarray(apply_factor(self._factor, b), dtype=float), _BACKEND)
            if self._repair_hits:
                # This factor's first answer needed refinement. Every RHS through the
                # same held factor needs that protection; the matrix is already held.
                x = settle_direct_solution(self.matrix, b, x, _BACKEND)
            return x
        matrix = build_matrix()
        self.matrix = matrix
        self._factor = factor_matrix(matrix)
        self._repair_hits = False
        if key is not None:
            self._key = key
        self.factorizations += 1
        # THE MISS IS DELIBERATELY *NOT* FINITENESS-CHECKED FIRST. It has the matrix, so
        # `reject_unsolved_direct_system` below can name the ORDER and both statistics -- and
        # `tests/unit/test_a_singular_march_is_refused_where_it_is_factored.py` pins exactly
        # that. Measured 2026-09-06: adding the cheap check here fired first on a singular MNA and
        # replaced a refusal naming n and the residual with one naming neither. A guard that
        # preempts a better-informed guard is a regression even when it refuses the same system.
        x = np.asarray(apply_factor(self._factor, b), dtype=float)
        # THE MISS IS WHERE THE MATRIX IS IN HAND, so it is the only place the guard is affordable
        # -- see "WHAT A HIT IS NOT JUDGED BY" in the module docstring. A SCALAR jacobian is not a
        # system the direct guard can speak about (``matrix @ x`` is not a matvec), and a divide
        # by an exactly-zero scalar already yields inf/nan the caller sees; it is skipped rather
        # than forced through a statistic that would be 0/0.
        shape = np.shape(matrix)
        if len(shape) == 2 and shape[0] == shape[1]:
            self.judged += 1
            # The plain direct path repairs an ill-conditioned but solvable answer
            # before applying this same guard. A cache miss owns the matrix too, so
            # refusing its first back-substitution without that repair is a different
            # numerical policy solely because the marcher requested reuse.
            original = x
            x = settle_direct_solution(matrix, b, x, _BACKEND)
            self._repair_hits = x is not original and not np.array_equal(x, original)
        return x
