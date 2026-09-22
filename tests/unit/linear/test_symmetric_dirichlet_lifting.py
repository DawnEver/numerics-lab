"""CSR Dirichlet lifting is space-independent linear algebra with one implementation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from motronics.gauss.linear import apply_dirichlet_csr


def _system() -> tuple[sp.csr_matrix, np.ndarray]:
    matrix = sp.csr_matrix([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]])
    return matrix, np.array([1.0, 2.0, 3.0])


def test_a_nonzero_pin_lifts_the_rhs_before_its_column_is_cleared() -> None:
    matrix, rhs = _system()
    constrained, load = apply_dirichlet_csr(
        matrix,
        rhs,
        np.array([0], dtype=np.intp),
        np.array([2.0]),
    )

    assert sp.isspmatrix_csr(constrained)
    np.testing.assert_array_equal(constrained.toarray(), [[1.0, 0.0, 0.0], [0.0, 4.0, -1.0], [0.0, -1.0, 3.0]])
    np.testing.assert_array_equal(load, [2.0, 4.0, 3.0])


def test_homogeneous_pins_clear_rows_and_columns_and_write_identity() -> None:
    matrix, rhs = _system()
    constrained, load = apply_dirichlet_csr(matrix, rhs, np.array([1], dtype=np.intp))

    np.testing.assert_array_equal(constrained.toarray(), [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    np.testing.assert_array_equal(load, [1.0, 0.0, 3.0])


def test_no_pins_leave_the_system_unchanged() -> None:
    matrix, rhs = _system()
    constrained, load = apply_dirichlet_csr(matrix, rhs, np.empty(0, dtype=np.intp))

    assert constrained is matrix
    assert load is rhs
