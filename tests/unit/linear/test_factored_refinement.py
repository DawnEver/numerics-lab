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


def _jacobi(matrix):
    """An INEXACT owned solve: the diagonal alone, so each correction only contracts the error."""
    diagonal = matrix.diagonal()
    return lambda forcing: forcing / diagonal


_DIAGONALLY_DOMINANT = sp.csr_matrix([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
# Rank one and inconsistent with rhs = (1, 0): no vector has a small residual, so both criteria must
# refuse whatever a (pseudo-inverse) owned solve proposes.
_SINGULAR = sp.csr_matrix([[1.0, 1.0], [1.0, 1.0]])


def _pseudo_inverse_solve(forcing):
    return np.linalg.pinv(_SINGULAR.toarray()) @ forcing


def test_step_size_criterion_iterates_an_inexact_solve_to_the_answer():
    rhs = np.array([1.0, 2.0, 3.0])
    exact = np.linalg.solve(_DIAGONALLY_DOMINANT.toarray(), rhs)
    calls = []
    solve = _jacobi(_DIAGONALLY_DOMINANT)

    def counted(forcing):
        calls.append(forcing)
        return solve(forcing)

    answer = refine_factored_solution(
        _DIAGONALLY_DOMINANT, rhs, np.zeros(3), counted, criterion='step-size', max_steps=60, step_rtol=1e-12
    )
    np.testing.assert_allclose(answer, exact, rtol=1e-10, atol=1e-14)
    assert 3 < len(calls) < 60  # stopped on the step bar, not on the step budget


def test_step_size_criterion_refuses_a_residual_above_its_bar_after_the_step_budget():
    # Planted control for the residual leg: three Jacobi steps leave a residual near 1e-2.
    with pytest.raises(LinearSolveError, match='residual'):
        refine_factored_solution(
            _DIAGONALLY_DOMINANT,
            np.array([1.0, 2.0, 3.0]),
            np.zeros(3),
            _jacobi(_DIAGONALLY_DOMINANT),
            criterion='step-size',
            max_steps=3,
            residual_rtol=1e-8,
        )


@pytest.mark.parametrize('criterion', ['backward-error', 'step-size'])
def test_a_singular_system_is_refused_by_each_criterion(criterion):
    with pytest.raises(LinearSolveError):
        refine_factored_solution(
            _SINGULAR, np.array([1.0, 0.0]), np.zeros(2), _pseudo_inverse_solve, criterion=criterion
        )


def test_backward_error_criterion_honours_its_step_budget():
    calls = []
    solve = _jacobi(_DIAGONALLY_DOMINANT)

    def counted(forcing):
        calls.append(forcing)
        return solve(forcing)

    with pytest.raises(LinearSolveError):
        refine_factored_solution(_DIAGONALLY_DOMINANT, np.array([1.0, 2.0, 3.0]), np.zeros(3), counted, max_steps=1)
    # one refinement step, the guard's measuring correction and its verdict step, then the refusal's
    # diagnostic trajectory -- all through the SAME owned solve, never a SuperLU refactor.
    assert len(calls) == 2 + 1 + module._DIRECT_REFINEMENT_MAX_STEPS


@pytest.mark.parametrize(
    'options',
    [{'criterion': 'newest'}, {'max_steps': 0}, {'step_rtol': -1.0}, {'residual_rtol': float('nan')}],
)
def test_an_undeclared_refinement_option_is_refused(options):
    with pytest.raises(ValueError, match=next(iter(options))):
        refine_factored_solution(sp.eye(1), np.ones(1), np.ones(1), lambda f: f, **options)


def _forbid_superlu(monkeypatch):
    def refuse(*_args, **_kwargs):
        msg = 'SuperLU must not be called: the diagnostics own the caller factor'
        raise AssertionError(msg)

    monkeypatch.setattr(module, 'splu', refuse)


def test_a_refusal_words_its_diagnostics_with_the_owned_factor_never_superlu(monkeypatch):
    matrix = sp.diags([1.0, 2.0, 4.0], format='csr')
    calls = []

    def garbage(rhs):
        calls.append(rhs.copy())
        return np.full_like(rhs, 7.0)

    _forbid_superlu(monkeypatch)
    with pytest.raises(LinearSolveError, match='diagnostic refinement trajectory') as raised:
        refine_factored_solution(matrix, np.ones(3), np.zeros(3), garbage)
    assert 'SuperLU' not in str(raised.value)
    assert len(calls) > 1  # the trajectory was measured through the owned solve


def test_planted_control_the_unowned_guard_does_reach_superlu(monkeypatch):
    # Without this, the test above could pass because the patch never bit.
    _forbid_superlu(monkeypatch)
    with pytest.raises(AssertionError, match='SuperLU must not be called'):
        module.reject_unsolved_direct_system(sp.diags([1.0, 2.0], format='csr'), np.ones(2), np.full(2, 7.0), 'plant')
