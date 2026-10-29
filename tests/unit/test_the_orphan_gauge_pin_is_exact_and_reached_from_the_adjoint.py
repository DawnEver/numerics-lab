"""The orphan gauge-pin is one owner, it is EXACT, and the DWR adjoint reaches it.

# The defect

``hamilton.error_estimate.dwr_indicator`` reassembles the SAME linear magnetostatic stiffness the
primal solve assembles, pins the same Dirichlet nodes, and solves it with the QoI load. The primal
paths (``fem._solve``, ``fem._newton``) additionally gauge-pin every ORPHAN dof -- a row whose
entries are all numerically zero, which a mesh that collapses at the axis produces because the node
is owned only by zero-area cells. The adjoint did not. So the adjoint's MATRIX differed from the
working primal's by exactly that pin and was exactly singular (rank deficiency 1, measured on
``case_e``: order 2294, node 0, owned by two collinear Q4 cells of area 1.0e-25).

Before ``gauss.linear``'s singular guard was reached from that call site, the adjoint returned NaN,
Dorfler marking selected nothing from a NaN indicator, the adaptive loop read that as convergence
and the CLI exited 0. Ten cases refined on noise and reported success.

# Why pinning is exact rather than a regularisation

An orphan dof appears in NO other equation, so fixing it cannot change any other unknown; and the
QoI load ``g`` is identically zero on it (asserted below), so ``z = 0`` there is the true solution
of the constrained problem, not a choice. That is what makes this ending (a) -- a dropped
constraint -- and not (b), a formulation that needs a gauge nobody had stated.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from numerics_lab.linear import LinearSolveError, gauge_pin_orphan_rows, orphan_rows, solve_linear_system


def _operator_with_orphan() -> tuple[csr_matrix, np.ndarray]:
    """A 4x4 SPD operator whose dof 3 is EXPLICITLY stored but numerically empty."""
    dense = np.array([
        [2.0, -1.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],  # the orphan: stored, all-zero
    ])
    mat = csr_matrix(dense)
    mat.data[np.abs(mat.data) == 0.0] = 0.0
    return mat, np.array([1.0, 0.0, 1.0, 0.0])


def test_orphan_rows_counts_the_numerically_empty_rows():
    mat, _ = _operator_with_orphan()
    assert orphan_rows(mat).tolist() == [3]


def test_orphan_rows_finds_none_in_a_healthy_operator():
    """The other side of the ratchet: a pin that fires on a healthy matrix would be a bug."""
    mat = csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    assert orphan_rows(mat).size == 0
    rhs = np.array([1.0, 1.0])
    assert gauge_pin_orphan_rows(mat, rhs) is not None
    assert rhs.tolist() == [1.0, 1.0], 'a healthy operator must leave the rhs untouched'


def test_the_unpinned_operator_is_refused_and_the_pinned_one_solves():
    mat, rhs = _operator_with_orphan()
    with pytest.raises(LinearSolveError):
        solve_linear_system(mat, rhs)

    pinned = gauge_pin_orphan_rows(mat, rhs)
    x = np.asarray(solve_linear_system(pinned, rhs)).reshape(-1)
    assert np.all(np.isfinite(x))
    assert x[3] == 0.0


def test_the_pin_changes_no_other_unknown():
    """EXACTNESS. The free block's solution is bit-identical to solving that block alone."""
    mat, rhs = _operator_with_orphan()
    free = np.array([0, 1, 2])
    reference = np.linalg.solve(mat.toarray()[np.ix_(free, free)], rhs[free])

    pinned = gauge_pin_orphan_rows(mat, rhs)
    x = np.asarray(solve_linear_system(pinned, rhs)).reshape(-1)
    assert x[free] == pytest.approx(reference, rel=1e-12, abs=1e-12)
