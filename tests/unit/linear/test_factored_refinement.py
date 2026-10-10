"""Refinement reuses the owned solve and never accepts a worse mixed-unit root."""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.gauss.linear.errors import LinearSolveError
from motronics.gauss.linear.refinement import refine_factored_solution

pytestmark = pytest.mark.unit


def test_refinement_resolves_a_small_unit_row_without_another_factor():
    matrix = sp.diags([1e20, 1e-20], format='csr')
    calls = []

    def solve(rhs):
        calls.append(rhs.copy())
        return rhs / matrix.diagonal()

    result = refine_factored_solution(matrix, np.array([1e20, 1e-20]), np.array([1.0, 0.0]), solve)
    np.testing.assert_array_equal(result, [1.0, 1.0])
    assert len(calls) == 1


def test_refinement_keeps_best_answer_when_a_correction_stagnates_or_worsens():
    initial = np.array([1 + np.finfo(float).eps])
    calls = []

    def worsening(rhs):
        calls.append(rhs)
        return np.ones(1)

    result = refine_factored_solution(sp.eye(1), np.ones(1), initial, worsening)
    np.testing.assert_array_equal(result, initial)
    assert len(calls) == 1


def test_finite_garbage_and_nonfinite_corrections_cannot_claim_physical_authority():
    with pytest.raises(LinearSolveError):
        refine_factored_solution(sp.eye(1), np.ones(1), np.zeros(1), lambda _rhs: np.zeros(1))
    with pytest.raises(LinearSolveError):
        refine_factored_solution(sp.eye(1), np.ones(1), np.zeros(1), lambda _rhs: np.full(1, np.nan))
