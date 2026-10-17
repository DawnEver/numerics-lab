"""A healthy direct solve pays ONE residual, not one for the repair and another for the verdict.

Measured 2026-10-12 on a 0.02 s PMSM position march (93,007 cache misses): the repair and the
guard each took ``||A x - b||`` of the same untouched vector, 186,014 residuals for 93,007 solves.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.linear import LinearSolveError, refinement, settle_direct_solution


def test_a_healthy_solve_takes_one_residual_and_returns_the_same_bits(monkeypatch) -> None:
    calls = []
    original = refinement.direct_relative_residual

    def counted(matrix, rhs, x):
        calls.append(1)
        return original(matrix, rhs, x)

    monkeypatch.setattr(refinement, 'direct_relative_residual', counted)
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    x = np.linalg.solve(matrix, rhs)
    settled = settle_direct_solution(matrix, rhs, x, 'probe')
    assert np.array_equal(settled, x)
    assert len(calls) == 1


def test_a_finite_answer_to_a_singular_system_is_still_refused() -> None:
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(LinearSolveError, match="'probe'"):
        settle_direct_solution(matrix, np.array([1.0, 0.0]), np.array([0.5, 0.5]), 'probe')
