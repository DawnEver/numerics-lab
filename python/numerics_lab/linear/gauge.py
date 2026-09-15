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

__all__ = ['gauge_pin_constant_mode', 'gauge_pin_orphan_rows', 'has_no_reference', 'orphan_rows']


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


#: How small a row's ``|(S @ 1)_i|`` must be RELATIVE TO THAT ROW'S OWN scale before the constant
#: vector is read as an EXACT null vector rather than a nearly-cancelling one. A stiffness row sums
#: to zero by construction (the shape functions are a partition of unity, so the constant field
#: carries no energy) and its cancellation is in the last bits, ~1e-16 of the row; a constrained
#: row carries a full ``1.0`` of its own scale. Six orders of margin sit between those two
#: populations and nothing lands in between.
#:
#: PER ROW, not against the matrix's largest row: a global comparison makes the verdict depend on
#: how badly scaled the system is, and would miss a genuine anchor sitting on a small-scale row.
_CONSTANT_MODE_FLOOR = 1e-10


def has_no_reference(mat) -> bool:
    """Whether the CONSTANT vector is an exact null vector of ``mat`` -- i.e. nothing anchors it.

    The separable PREDICATE behind :func:`gauge_pin_constant_mode`, exposed for the same reason
    :func:`orphan_rows` is: a pin that fired zero times and a pin that was never reached look
    identical from the solve alone, so a test needs to be able to ASK.

    Each row is judged against ITS OWN scale, never against a global maximum -- see
    ``_CONSTANT_MODE_FLOOR``. An all-zero (orphan) row imposes no reference and votes on nothing,
    so it is excluded rather than counted as evidence either way.
    """
    n = mat.shape[0]
    if n == 0:
        return False
    residual = np.abs(np.asarray(mat @ np.ones(n)).ravel())
    row_scale = np.asarray(np.abs(mat).sum(axis=1)).ravel()
    live = row_scale > 0.0
    if not live.any():
        return False
    return float(np.max(residual[live] / row_scale[live])) <= _CONSTANT_MODE_FLOOR


def gauge_pin_constant_mode(mat, rhs: NDArray[np.float64]):
    """Dirichlet-pin ONE DOF when the whole system has no reference and the solution floats.

    In 2-D magnetostatics the unknown is the vector potential ``A`` and only ``B = curl A`` is
    physical, so ``A`` is defined up to an additive constant. A stiffness row sums to zero, which
    is exactly the statement that the constant vector lies in the null space; SOMETHING must
    remove it. Normally something does -- a Dirichlet node, or an anti-periodic (``'odd'`` parity)
    pair imposing ``A_s = -A_m``, each contributing a row whose sum is 1 rather than 0. This
    function is for the model where NOTHING does.

    :func:`gauge_pin_orphan_rows` is named for the same physics and cannot reach this case: it
    tests for an all-ZERO row, and the global constant mode leaves every row fully populated.

    MEASURED 2026-09-06 on ``spm_ferrari_luce_rear``, a full-ring (``n_portion = 1``) model::

        dirichlet_values=[]  dirichlet_counts=[]  n_vert=146174
        sym_parities=[]      sym_pairs=[]         sliding_pairs=0

    Not one constraint of any kind -- 146174 free DOFs and no reference at all, so the constant
    was precisely the null space. With 400 increments instrumented inside the newton and picard
    strategies, every one of them was 100.00 % the constant mode (mean-removed 1.0e-06 against raw
    3.5e-02 .. 2.7e-01, against a criterion needing < 1e-4), the energy sat flat at 185.69 to six
    digits, and all four one-hot solves ran to the 100-iteration cap. Each landed on its OWN
    arbitrary constant, which is why the assembled inductance matrix came back NON-PSD at
    364.177 A (eigenvalue -5.76e-05 H) and 19 % asymmetric at 36.4 A -- and ``L_ij = L_ji`` is an
    energy theorem, so that asymmetry was a statement about the MEASUREMENT, not the machine.

    NOTHING SAID SO. ``spsolve`` did not raise ``MatrixRankWarning``, so the "(near-)singular"
    guard in :func:`~motronics.gauss.linear.solve_linear_system` never fired; it returned a
    finite, arbitrary answer and the run reported it as a field.

    THE PIN IS EXACT, NOT AN APPROXIMATION: fixing the free constant is a choice of gauge, and
    every gauge gives the same ``B``. The DOF chosen is the one with the largest ``|diagonal|``
    among the live rows -- the best-connected equation, and never a disconnected DOF, which
    anchors nothing.

    Args:
        mat: the assembled sparse operator (any scipy sparse format; CSR is returned).
        rhs: the right-hand side, ZEROED IN PLACE on the pinned row.

    Returns:
        the operator with one identity row, or UNCHANGED when a reference already exists, so an
        anchored system is never rebuilt and never over-constrained.

    """
    if not has_no_reference(mat):
        return mat  # a reference exists -- leave the assembled system alone
    row_scale = np.asarray(np.abs(mat).sum(axis=1)).ravel()
    live = row_scale > 0.0
    diag = np.abs(mat.diagonal()) * live  # never pin a disconnected DOF: it anchors nothing
    pin = int(np.argmax(diag))  # argmax already breaks ties toward the lowest index
    lil = mat.tolil()
    lil[pin, :] = 0.0
    lil[pin, pin] = 1.0
    rhs[pin] = 0.0
    return lil.tocsr()
