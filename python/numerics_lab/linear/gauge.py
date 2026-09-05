"""Orphan-DOF gauge pinning -- pure algebra on an assembled matrix, no physics, no mesh.

An assembled operator can carry a row that is NUMERICALLY empty: every entry on it is an
explicit zero. A DOF like that couples to nothing, so the operator has a null vector along it
and any direct factorisation is exactly singular. Its value is pure gauge freedom -- pinning it
to zero with an identity row is EXACT, not a regularisation, and it is the only way to pin it
that cannot change the rest of the solution, because the DOF appears in no other equation.

The test is NUMERIC (row sum of ``|entries|`` == 0), not structural: an assembler that drops a
degenerate element still leaves explicit-zero COO triplets, so the row is structurally present
but numerically empty. A healthy operator has no such row and nothing is pinned.

# Why this lives in ``gauss``

It was born as ``hamilton.discretizations.fem._solve._gauge_pin_orphan_dofs`` and was applied by
the two PRIMAL solve paths (``_solve``, ``_newton``) and by nobody else. The DWR adjoint in
``hamilton.error_estimate`` reassembles the SAME stiffness with a different right-hand side and
did not apply it, so on the ten cases whose mesh collapses at the axis the adjoint was exactly
singular where the primal was fine -- the matrix differed from the working primal's by exactly
this pin, and before ``gauss.linear``'s singular guard was reached the adjoint returned NaN and
steered adaptive refinement on it. The decision "what to do about an all-zero row" belongs to one
owner; ``gauss.linear`` already owns "how a sparse system is solved and when the answer is
refused", so it owns this too, and every assembler consults it rather than re-deciding privately.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ['gauge_pin_orphan_rows', 'orphan_rows']


def orphan_rows(mat) -> NDArray[np.intp]:
    """Indices of the numerically empty rows of ``mat`` (``sum |entries| == 0``).

    The separable PREDICATE behind :func:`gauge_pin_orphan_rows`, exposed so a caller (or a test)
    can COUNT the orphans an operator carries instead of only observing that a solve stopped
    failing -- a pin that fired zero times and a pin that was never reached look identical from
    the solve alone.
    """
    abs_row_sum = np.asarray(np.abs(mat).sum(axis=1)).ravel()
    return np.nonzero(abs_row_sum == 0.0)[0].astype(np.intp)


def gauge_pin_orphan_rows(mat, rhs: NDArray[np.float64]):
    """Dirichlet-pin every orphan DOF of ``mat`` to zero so the system is full-rank.

    Args:
        mat: the assembled sparse operator (any scipy sparse format; CSR is returned).
        rhs: the right-hand side, ZEROED IN PLACE on every pinned row.

    Returns:
        the operator as CSR, with an identity row on each orphan DOF. Returned UNCHANGED (and
        ``rhs`` untouched) when there is no orphan row, so a healthy operator is not rebuilt.

    """
    orphans = orphan_rows(mat)
    if orphans.size == 0:
        return mat.tocsr()
    lil = mat.tolil()
    for r in orphans:
        lil[r, r] = 1.0
        rhs[r] = 0.0
    return lil.tocsr()
