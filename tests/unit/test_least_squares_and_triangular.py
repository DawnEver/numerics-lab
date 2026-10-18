"""Least squares and triangular solves are numerics-lab's, and they fail with its own error type.

Each answer is checked against a numpy closed form with a RELATIVE (unit-free) tolerance, and each
refusal against a planted singular system.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.linear import LinearSolveError
from numerics_lab.linear.cholesky import jittered_cholesky
from numerics_lab.linear.dense import least_squares, solve_triangular
from numerics_lab.linear.kkt import solve_kkt

_RNG = np.random.default_rng(11)


def test_an_overdetermined_least_squares_is_the_normal_equations_solution() -> None:
    a = _RNG.standard_normal((20, 5))
    b = _RNG.standard_normal((20, 3))
    closed_form = np.linalg.solve(a.T @ a, a.T @ b)
    np.testing.assert_allclose(least_squares(a, b), closed_form, rtol=1e-10)
    np.testing.assert_allclose(least_squares(a, b[:, 0]), closed_form[:, 0], rtol=1e-10)


def test_a_rank_deficient_least_squares_is_the_minimum_norm_solution() -> None:
    a = _RNG.standard_normal((10, 4))
    a[:, 3] = a[:, 0] + a[:, 1]  # rank 3
    b = _RNG.standard_normal(10)
    np.testing.assert_allclose(least_squares(a, b), np.linalg.pinv(a) @ b, rtol=1e-9)


def test_a_triangular_solve_is_the_inverse_applied() -> None:
    lower = np.tril(_RNG.standard_normal((6, 6))) + 6.0 * np.eye(6)
    rhs = _RNG.standard_normal((6, 2))
    np.testing.assert_allclose(solve_triangular(lower, rhs), np.linalg.solve(lower, rhs), rtol=1e-12)
    np.testing.assert_allclose(solve_triangular(lower.T, rhs, lower=False), np.linalg.solve(lower.T, rhs), rtol=1e-12)


def test_the_triangle_not_named_is_never_read() -> None:
    """Planted control: garbage in the other triangle does not move the answer."""
    lower = np.tril(_RNG.standard_normal((5, 5))) + 5.0 * np.eye(5)
    rhs = _RNG.standard_normal(5)
    dirty = lower + np.triu(_RNG.standard_normal((5, 5)), k=1)
    np.testing.assert_allclose(solve_triangular(dirty, rhs), solve_triangular(lower, rhs), rtol=0.0)


def test_a_zero_pivot_triangle_is_refused() -> None:
    lower = np.tril(np.ones((3, 3)))
    lower[1, 1] = 0.0
    with pytest.raises(LinearSolveError, match='singular'):
        solve_triangular(lower, np.ones(3))


def test_non_finite_input_is_refused() -> None:
    with pytest.raises(ValueError, match='finite'):
        least_squares(np.array([[1.0], [np.nan]]), np.ones(2))
    with pytest.raises(ValueError, match='finite'):
        solve_triangular(np.eye(2), np.array([1.0, np.inf]))


def test_every_dense_refusal_is_the_seams_own_error() -> None:
    """Jittered Cholesky and the dense KKT fail with ``LinearSolveError``, as the sparse KKT already did."""
    with pytest.raises(LinearSolveError, match='jitter'):
        jittered_cholesky(np.diag([1.0, -1.0]))
    h, a = np.eye(3), np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(LinearSolveError, match='singular'):
        solve_kkt(h, a, np.ones(3), np.ones(2))
