"""CSR Dirichlet lifting is space-independent linear algebra with one implementation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear import apply_dirichlet_csr


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


def test_each_schur_rhs_keeps_its_own_nonzero_boundary_increment() -> None:
    matrix, rhs = _system()
    loads = np.column_stack((rhs, 2.0 * rhs))
    pinned = np.array([0], dtype=np.intp)
    prescribed = np.array([[2.0, -3.0]])
    constrained, lifted = apply_dirichlet_csr(matrix, loads, pinned, prescribed)
    original = matrix.toarray()
    original[0] = [1.0, 0.0, 0.0]
    row_pinned_loads = loads.copy()
    row_pinned_loads[pinned] = prescribed
    np.testing.assert_allclose(
        np.linalg.solve(constrained.toarray(), lifted),
        np.linalg.solve(original, row_pinned_loads),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_array_equal(lifted[pinned], prescribed)
    np.testing.assert_array_equal(matrix.toarray(), _system()[0].toarray())
    np.testing.assert_array_equal(loads, np.column_stack((rhs, 2.0 * rhs)))


def test_vector_boundary_values_apply_to_every_rhs_without_column_aliasing() -> None:
    matrix, rhs = _system()
    loads = np.column_stack((rhs, 2.0 * rhs, 3.0 * rhs))
    pinned = np.array([0, 2], dtype=np.intp)
    constrained, lifted = apply_dirichlet_csr(matrix, loads, pinned, np.array([2.0, -3.0]))
    np.testing.assert_array_equal(lifted[pinned], [[2.0] * 3, [-3.0] * 3])
    np.testing.assert_array_equal(constrained.toarray(), np.diag([1.0, 4.0, 1.0]))
    np.testing.assert_array_equal(lifted[1], loads[1] - 1.0)


def test_complex_boundary_values_survive_the_pure_algebraic_lifting() -> None:
    matrix, rhs = _system()
    prescribed = np.array([2.0 + 3.0j])
    constrained, lifted = apply_dirichlet_csr(matrix, rhs, np.array([0], dtype=np.intp), prescribed)
    np.testing.assert_array_equal(lifted, [2.0 + 3.0j, 4.0 + 3.0j, 3.0])
    np.testing.assert_array_equal(constrained.toarray(), [[1, 0, 0], [0, 4, -1], [0, -1, 3]])
