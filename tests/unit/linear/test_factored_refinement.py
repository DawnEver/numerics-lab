"""Refinement reuses the owned solve and never accepts a worse mixed-unit root."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import refinement as module
from numerics_lab.linear.errors import LinearSolveError
from numerics_lab.linear.refinement import refine_factored_solution

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


def test_cancellation_guard_measures_the_actual_owned_factor_not_another_factor(monkeypatch):
    matrix = sp.csr_matrix([[1e20, 1e20], [0.0, 1.0]])
    rhs, initial = np.ones(2), np.array([-1.0, 1.0])
    calls = []

    def owned(forcing):
        calls.append(forcing.copy())
        rate = forcing[1]
        return np.array([forcing[0] / 1e20 - rate, rate])

    def forbidden_factor(_matrix):
        msg = 'a factored response must retain its actual owned factor'
        raise AssertionError(msg)

    monkeypatch.setattr(module, 'splu', forbidden_factor)
    answer = refine_factored_solution(matrix, rhs, initial, owned)
    np.testing.assert_array_equal(answer, initial)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[-1], module.refinement_residual(matrix, rhs, answer))


def test_cancellation_guard_rejects_a_lying_tiny_owned_correction():
    matrix = sp.csr_matrix([[1e20, 1e20], [0.0, 1.0]])
    with pytest.raises(LinearSolveError):
        refine_factored_solution(matrix, np.ones(2), np.array([-1.0, 1.0]), lambda _rhs: np.full(2, 1e-30))


def test_singular_owned_correction_cannot_supply_a_successful_guard_measurement():
    matrix = sp.csr_matrix([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(LinearSolveError):
        refine_factored_solution(matrix, np.array([1.0, 0.0]), np.zeros(2), lambda _rhs: np.zeros(2))


def test_a_subnormal_defect_ends_refinement_instead_of_asking_for_its_correction():
    """MEASURED 2026-10-13 on test_coupled_network_checkpoint's switched network: the joint storage
    border's refinement handed its row-normalized correction a 1.3e-321 defect; divided by the row
    scale 161 it underflowed to zero, and the normalization refused the admission. A defect below the
    smallest normal float has no representable correction, so the refined value is final.
    """
    matrix = sp.diags([1.0, 1.0], format='csr')
    rhs = np.array([1.0, 5e-322])
    initial = np.array([1.0, 0.0])

    def solve(forcing):
        if np.any((forcing != 0) & (abs(forcing) < np.finfo(float).tiny)):
            msg = 'a subnormal forcing cannot be normalized'
            raise ValueError(msg)
        return forcing.copy()

    np.testing.assert_array_equal(refine_factored_solution(matrix, rhs, initial, solve), initial)
