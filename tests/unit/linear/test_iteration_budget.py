"""Caller Krylov budgets preserve the shared default and reject false convergence."""

import numpy as np
import pytest
from scipy.sparse import diags

from numerics_lab import linear
from numerics_lab.linear import solve_linear_system


@pytest.mark.parametrize('budget', [0, -1, True, np.bool_(True), 1.5])
def test_krylov_budget_requires_a_positive_nonboolean_integer(budget):
    with pytest.raises(ValueError, match='positive.*integer|iteration'):
        solve_linear_system(diags([1.0, 2.0]), np.ones(2), linear_solver='gmres', max_iterations=budget)


def test_a_restrictive_cg_budget_refuses_the_unsolved_original_rhs():
    with pytest.raises(RuntimeError, match='converge'):
        solve_linear_system(diags(np.arange(1.0, 41.0)), np.ones(40), linear_solver='cg', max_iterations=1)


def test_an_explicit_sufficient_budget_reproduces_the_shared_default():
    matrix, rhs = diags(np.arange(1.0, 6.0)), np.ones(5)
    actual = solve_linear_system(matrix, rhs, linear_solver='cg', max_iterations=10)
    np.testing.assert_allclose(actual, solve_linear_system(matrix, rhs, linear_solver='cg'), rtol=1e-12)


def test_gmres_cap_counts_inner_iterations_not_a_whole_restart_cycle():
    matrix, rhs = diags(np.arange(1.0, 6.0)), np.ones(5)
    with pytest.raises(RuntimeError, match='converge'):
        solve_linear_system(matrix, rhs, linear_solver='gmres', max_iterations=1)
    actual = solve_linear_system(matrix, rhs, linear_solver='gmres', max_iterations=10)
    np.testing.assert_allclose(actual, solve_linear_system(matrix, rhs), rtol=1e-12)


def test_gmres_explicit_cap_reaches_scipys_documented_inner_iteration_callback(monkeypatch):
    original = linear.gmres
    callbacks = []

    def counted(matrix, rhs, **kwargs):
        callback = kwargs.get('callback')
        assert callback is not None
        assert kwargs.get('callback_type') == 'legacy'

        def observe(residual):
            callbacks.append(residual)
            if callback is not None:
                callback(residual)

        kwargs['callback'] = observe
        return original(matrix, rhs, **kwargs)

    # Build the wrapper against the spy so numerical work and callback semantics
    # come from actual SciPy, not a fabricated solver response.
    monkeypatch.setattr(linear, 'gmres', counted)
    monkeypatch.setitem(linear.LINEAR_SOLVER_REGISTRY, 'gmres', linear._krylov(counted))
    with pytest.raises(RuntimeError, match='converge'):
        linear.solve_linear_system(diags(np.arange(1.0, 6.0)), np.ones(5), linear_solver='gmres', max_iterations=2)
    assert len(callbacks) == 2
