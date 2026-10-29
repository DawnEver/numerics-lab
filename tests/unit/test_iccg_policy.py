"""ICCG solves the same operator; automatic selection does not assume SPD."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import make_preconditioner, solve_linear_system


def test_ic0_is_exact_on_tridiagonal_spd():
    matrix = sp.diags([-np.ones(7), 3 * np.ones(8), -np.ones(7)], [-1, 0, 1]).tocsr()
    rhs = np.arange(8, dtype=float)
    preconditioner = make_preconditioner('ic0', matrix)
    np.testing.assert_allclose(matrix @ (preconditioner @ rhs), rhs, atol=1e-13)
    np.testing.assert_allclose(
        solve_linear_system(matrix, rhs, linear_solver='iccg'), np.linalg.solve(matrix.toarray(), rhs)
    )


def test_ic0_refuses_indefinite_and_nonsymmetric_operators():
    for matrix in (sp.diags([1.0, -1.0]), sp.csr_matrix([[2.0, 1.0], [0.0, 2.0]])):
        with pytest.raises(ValueError, match='IC0'):
            make_preconditioner('ic0', matrix)


def test_auto_and_explicit_iterative_tolerance():
    matrix = sp.csr_matrix([[3.0, 1.0], [0.0, 2.0]])
    np.testing.assert_allclose(solve_linear_system(matrix, np.ones(2), linear_solver='auto'), [1 / 6, 1 / 2])
    np.testing.assert_allclose(
        solve_linear_system(matrix, np.ones(2), linear_solver='gmres', rtol=1e-8), [1 / 6, 1 / 2]
    )


def test_ic0_accepts_only_relative_roundoff_asymmetry():
    for scale in [1e-20, 1.0, 1e20]:
        matrix = sp.csr_matrix(np.array([[3.0, -1.0], [-1.0 + 1e-15, 3.0]]) * scale)
        rhs = np.ones(2) * scale
        np.testing.assert_allclose(solve_linear_system(matrix, rhs, linear_solver='iccg'), [0.5, 0.5])
        matrix[1, 0] = (-1.0 + 1e-8) * scale
        with pytest.raises(ValueError, match='symmetric'):
            make_preconditioner('ic0', matrix)


def test_iccg_lifts_exact_identity_rows_without_symmetrizing_free_equations():
    matrix = sp.csr_matrix([[1.0, 0.0, 0.0], [-2.0, 4.0, -1.0], [3.0, -1.0, 5.0]])
    rhs = np.array([2.0, 1.0, 3.0])
    expected = np.linalg.solve(matrix.toarray(), rhs)
    for name in ('iccg', 'auto'):
        np.testing.assert_allclose(solve_linear_system(matrix, rhs, linear_solver=name), expected, rtol=1e-12)
    np.testing.assert_array_equal(rhs, [2.0, 1.0, 3.0])
    loose = solve_linear_system(matrix, rhs, linear_solver='iccg', atol=1e6)
    assert loose[0] == 2.0, 'prescribed values remain exact even with loose inner tolerance'
    matrix[2, 1] = -2.0
    with pytest.raises(ValueError, match='symmetric'):
        solve_linear_system(matrix, rhs, linear_solver='iccg')
