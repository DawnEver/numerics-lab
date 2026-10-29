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

__all__ = [
    'constraint_rows',
    'floats_on_a_constant',
    'gauge_pin_floating_constant',
    'gauge_pin_orphan_rows',
    'orphan_rows',
]


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


def constraint_rows(mat) -> NDArray[np.bool_]:
    """Mask of rows that are CONSTRAINTS (a lone diagonal entry), not equations.

    A paired-DOF fold eliminates each slave by turning its row into an identity row and recovering
    ``A[slave] = sign * A[master]`` after the solve; a Dirichlet pin does the same. Both are
    statements ABOUT a DOF rather than an equation the operator has to satisfy, and
    :func:`floats_on_a_constant` must not read them -- which is the bug this mask exists to stop.
    MEASURED 2026-09-15 on `case_c --scope pole-pair`: of 11995 rows exactly 266 had
    a nonzero row sum, every one of them exactly 1.0, and they were the seam's eliminated slaves.
    A whole-matrix test therefore reported the operator GAUGED (ratio 3.155e-09) while the physical
    problem was not gauged at all.
    """
    csr = mat.tocsr()
    counts = np.diff(csr.indptr)
    mask = np.zeros(csr.shape[0], dtype=bool)
    lone = np.nonzero(counts == 1)[0]
    if lone.size:
        mask[lone] = csr.indices[csr.indptr[lone]] == lone
    return mask


def floats_on_a_constant(mat, *, rtol: float = 1e-8) -> bool:
    """Whether the EQUATION rows annihilate the constant vector.

    HALF of the ungauged question and NOT the whole of it: this says the equations impose no
    reference, which stays true even after a Dirichlet row is added, because that row is a
    CONSTRAINT and is excluded here. The other half -- whether the model declares a Dirichlet at
    all -- is the caller's, and :func:`gauge_pin_floating_constant` requires both.

    A 2-D magnetostatic stiffness is ``INT nu grad(A) . grad(v)``, so adding a constant to ``A``
    changes no equation: the constant vector is an exact null vector. A periodic seam preserves
    that (it adds rows together), and an ANTI-periodic one does not, so a machine folded on an odd
    seam is gauged by its own symmetry and a machine folded on an even one is not. What otherwise
    breaks the degeneracy is a real DIRICHLET boundary -- and `param_model_compile` tags none, so
    every `param_fit` sector reaches the solver ungauged.

    Constraint rows are EXCLUDED (see :func:`constraint_rows`); the tolerance is relative to the
    largest equation row sum, so it means the same in metres and millimetres.
    """
    n = mat.shape[0]
    if n == 0:
        return False
    keep = ~constraint_rows(mat)
    if not keep.any():
        return False
    csr = mat.tocsr()
    row_scale = float(np.abs(csr).sum(axis=1).max())
    if row_scale == 0.0:
        return False
    return float(np.max(np.abs((csr @ np.ones(n))[keep]))) <= rtol * row_scale


def gauge_pin_floating_constant(mat, rhs: NDArray[np.float64], *, has_dirichlet: bool):
    """Pin ONE dof when the problem has no Dirichlet reference, so ``A`` stops floating.

    WHY, measured 2026-09-15 on `case_c --scope pole-pair --point rated_25500`. The
    coefficient-damped Picard ran its full 100 iterations with the ENERGY converged to 9.2e-05 percent and
    the iterate wandering by up to 200x its own norm:

        i=92 alpha=0.050000 |x|=1.96638e+02 E=28.339107 err%=0.000131 inc=1.9358e+02
        i=93 alpha=0.050000 |x|=1.82129e+00 E=28.339143 err%=0.000124 inc=1.0086e+00
        i=99 alpha=0.050000 |x|=5.55864e+00 E=28.339321 err%=0.000092 inc=4.8307e+00

    and the arithmetic names the direction: a uniform shift ``c`` over 11995 DOFs has norm
    ``109.5 c``, so the observed ``||dx|| ~ 4`` is ``c ~ 0.037``, which moves ``||x||`` from 1.8 to
    4.4 -- exactly the swing. The solve was converged in every observable and wandering in the
    gauge, and `_has_converged` requires BOTH the energy and the increment, so it could never
    report the convergence it had already reached.

    ORPHAN PINNING DOES NOT COVER THIS: :func:`gauge_pin_orphan_rows` pins rows coupled to nothing,
    and here every row is fully coupled while the operator is still singular in one global
    direction.

    WHAT IT CANNOT CHANGE: ``B = curl A``, so a constant shift in ``A`` leaves every flux density,
    energy and torque untouched. The pin is EXACT for every observable, not a regularisation.

    WHY ``has_dirichlet`` IS THE CALLER'S TO ANSWER AND NOT THIS FUNCTION'S. A real Dirichlet row
    and a fold's eliminated slave row are the SAME OBJECT -- a lone 1.0 on the diagonal -- so no
    test on the matrix can tell them apart, and a detector that guessed would eventually pin a
    SECOND dof on a properly constrained model and return a silently wrong field. The unambiguous
    fact lives one layer up: whether the model DECLARES any Dirichlet boundary. It is a required
    keyword so that no caller can arrive at the wrong default by omission.

    Returns the operator UNCHANGED when the model is gauged -- by a declared Dirichlet boundary or
    by an anti-periodic seam -- so a model that determines its own potential is never touched.
    """
    if has_dirichlet or not floats_on_a_constant(mat):
        return mat.tocsr()
    csr = mat.tocsr()
    free = ~constraint_rows(csr)
    diag = np.abs(csr.diagonal()) * free
    # THE BEST-CONDITIONED FREE DOF. Pinning is exact wherever it lands, but a sliver cell's tiny
    # diagonal makes the factorisation worse for nothing, and a CONSTRAINT row must not be chosen:
    # overwriting one would drop the seam pairing it encodes.
    r = int(np.argmax(diag))
    lil = csr.tolil()
    lil.rows[r] = [r]
    lil.data[r] = [float(diag[r]) if diag[r] > 0.0 else 1.0]
    rhs[r] = 0.0
    return lil.tocsr()
