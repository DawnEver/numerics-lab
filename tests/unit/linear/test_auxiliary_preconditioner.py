"""Complex additive auxiliary-space identities, independent of any electromagnetic kernel."""

import importlib
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

from numerics_lab.linear import LinearSolveError, solve_linear_system


def test_additive_inverse_matches_the_transpose_and_complex_scale_dense_oracle():
    module = importlib.import_module('numerics_lab.linear.auxiliary')
    matrix = csr_matrix(np.array([[3.0 + 2j, 0.2], [0.2, 4.0 + 1j]]))
    transfer = csr_matrix(np.array([[1.0 + 0.3j], [-0.4 + 0.2j]]))
    scalar = csr_matrix(np.array([[2.0 + 0.7j]]))
    scale = (1j * 7.0) ** -2
    operator = module.additive_auxiliary_preconditioner(matrix, transfer, scalar, scale=scale)
    rhs = np.array([1.0 + 2j, 3.0 - 0.4j])
    expected = rhs / matrix.diagonal() + scale * transfer.toarray() @ np.linalg.solve(
        scalar.toarray(), transfer.toarray().T @ rhs
    )
    np.testing.assert_allclose(operator @ rhs, expected, rtol=1e-13, atol=1e-13)


def test_registered_gmres_preserves_complex_rhs_seed_and_explicit_preconditioner():
    matrix = csr_matrix(np.array([[3.0 + 2j, 0.2], [0.2, 4.0 + 1j]]))
    expected = np.array([1.0 + 2j, 3.0 - 0.4j])
    inverse = LinearOperator(matrix.shape, matvec=lambda value: value / matrix.diagonal(), dtype=complex)
    actual = solve_linear_system(
        matrix,
        matrix @ expected,
        linear_solver='gmres',
        preconditioner=inverse,
        x0=np.array([0.2 + 0.7j, -0.1 + 0.3j]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_explicit_preconditioner_must_match_the_actual_operator_shape():
    matrix = csr_matrix(np.eye(2))
    inverse = LinearOperator((3, 3), matvec=lambda value: value, dtype=float)
    with pytest.raises(ValueError, match='shape'):
        solve_linear_system(matrix, np.ones(2), linear_solver='gmres', preconditioner=inverse)


def test_a_complex_warm_seed_for_a_real_system_is_not_silently_projected():
    matrix = csr_matrix(np.array([[3.0, 0.2], [0.2, 4.0]]))
    expected = np.array([1.0, 2.0])
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        actual = solve_linear_system(
            matrix, matrix @ expected, linear_solver='gmres', x0=np.array([0.2 + 0.7j, -0.1 + 0.3j]), rtol=1e-12
        )
    assert not any('Casting complex values' in str(warning.message) for warning in observed)
    assert np.iscomplexobj(actual)
    np.testing.assert_allclose(matrix @ actual, matrix @ expected, rtol=1e-12, atol=1e-12)


def test_nonfinite_auxiliary_action_cannot_return_a_successful_finite_field():
    matrix = csr_matrix(np.eye(2))
    inverse = LinearOperator((2, 2), matvec=lambda _value: np.full(2, np.nan), dtype=float)
    with pytest.raises(LinearSolveError, match='converge|finite|breakdown'):
        solve_linear_system(matrix, np.ones(2), linear_solver='gmres', preconditioner=inverse)
