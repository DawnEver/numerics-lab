"""A Picard equation need not claim source scaling or arbitrary secant solves."""

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator
from tests.unit.test_nonlinear import _LinearSystem

from numerics_lab.nonlinear import (
    BFGSStrategy,
    BroydenStrategy,
    ContinuationStrategy,
    IterationBudgetExceeded,
    LBFGSStrategy,
    NewtonKrylovStrategy,
    NewtonLineSearchStrategy,
    strategy_runs_on,
    with_time_budget,
)
from numerics_lab.nonlinear import _strategies_newton as owner


@pytest.mark.parametrize('budgeted', [False, True])
def test_continuation_without_source_scaling_solves_the_original_source(budgeted):
    system = _LinearSystem([3.0])
    wrapped = with_time_budget(system, seconds=10.0, clock=lambda: 0.0) if budgeted else system
    solved = ContinuationStrategy().solve(wrapped, max_iter=3)
    assert solved.converged
    np.testing.assert_array_equal(solved.x, [3.0])


def test_full_source_fallback_keeps_the_actual_time_budget():
    clock = iter([0.0, 20.0])
    system = with_time_budget(_LinearSystem([3.0]), seconds=10.0, clock=lambda: next(clock))
    with pytest.raises(IterationBudgetExceeded):
        ContinuationStrategy().solve(system, max_iter=3)


@pytest.mark.parametrize('strategy', [BroydenStrategy(), BFGSStrategy(), LBFGSStrategy()])
@pytest.mark.parametrize('budgeted', [False, True])
def test_quasi_newton_refuses_missing_actual_secant_capability_before_iteration(strategy, budgeted):
    system = _LinearSystem([3.0])
    wrapped = with_time_budget(system, seconds=10.0, clock=lambda: 0.0) if budgeted else system
    with pytest.raises(TypeError, match='solve_secant|secant'):
        strategy.solve(wrapped, max_iter=3)


@pytest.mark.parametrize('name', ['quasi-newton', 'bfgs', 'l-bfgs'])
@pytest.mark.parametrize('budgeted', [False, True])
def test_public_admission_uses_actual_optional_capabilities(name, budgeted):
    absent = _LinearSystem([3.0])

    class Supported(_LinearSystem):
        def solve_secant(self, _state, rhs, /):
            return np.asarray(rhs)

    present = Supported([3.0])
    if budgeted:
        absent = with_time_budget(absent, seconds=10.0, clock=lambda: 0.0)
        present = with_time_budget(present, seconds=10.0, clock=lambda: 0.0)
    assert not strategy_runs_on(name, absent)
    assert strategy_runs_on(name, present)
    assert strategy_runs_on('continuation', absent)


@pytest.mark.parametrize('strategy', [BroydenStrategy(), BFGSStrategy(), LBFGSStrategy()])
def test_supported_secant_is_called_through_its_budget_not_the_unwrapped_equation(strategy):
    class Supported(_LinearSystem):
        def residual(self, x, _state, /):
            return x - self.target

        def solve_secant(self, _state, rhs, /):
            pytest.fail('the expired secant budget must refuse before calling the equation')

    clock = iter([0.0, 1.0, 20.0])
    system = with_time_budget(Supported([3.0]), seconds=10.0, clock=lambda: next(clock))
    with pytest.raises(IterationBudgetExceeded, match='solve_secant'):
        strategy.solve(system, max_iter=3)


def test_a_negative_backtracking_budget_is_refused_before_any_solve():
    with pytest.raises(ValueError, match='max_backtracks'):
        NewtonLineSearchStrategy(max_backtracks=-1)


def test_the_canonical_directional_operator_preserves_dtype_probe_and_math(monkeypatch):
    class NoTangent(_LinearSystem):
        def residual(self, x, _state, /):
            return x - self.target

        def tangent(self, _x, _state, /):
            return None

    original = owner.aslinearoperator
    checked = []

    def compare(carrier):
        observed = []
        matvec = carrier.matvec

        def record(vector):
            observed.append(vector.copy())
            return matvec(vector)

        carrier.matvec = record
        actual = original(carrier)
        expected = LinearOperator(carrier.shape, matvec=carrier.matvec)
        assert len(observed) == 2
        assert observed[0].dtype == observed[1].dtype == np.int8
        np.testing.assert_array_equal(observed[0], observed[1])
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual.matvec(np.array([0.375])), expected.matvec(np.array([0.375])))
        checked.append(True)
        return actual

    monkeypatch.setattr(owner, 'aslinearoperator', compare)
    result = NewtonKrylovStrategy().solve(NoTangent([3.0]), max_iter=3)
    assert result.converged
    np.testing.assert_allclose(result.x, [3.0], rtol=0.0, atol=1e-8)
    assert checked
