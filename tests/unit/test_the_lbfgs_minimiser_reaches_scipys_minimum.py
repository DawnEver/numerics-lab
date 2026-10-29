"""``minimize_lbfgs`` -- the one unconstrained minimiser -- against scipy.optimize L-BFGS-B, unbounded (tests only).

Tolerances: positions are ABSOLUTE in the variables' own units (dimensionless here); values RELATIVE.
"""

import numpy as np
import pytest
from scipy.optimize import minimize, rosen, rosen_der

from numerics_lab.minimize import MinimizeResult, minimize_lbfgs


def _rosen(x):
    return float(rosen(x)), rosen_der(x)


def _quadratic(x):
    a = np.diag(np.arange(1.0, x.size + 1))
    return float(0.5 * x @ a @ x - x.sum()), a @ x - 1


@pytest.mark.parametrize(
    ('fun', 'x0'), [(_rosen, np.array([-1.2, 1.0])), (_rosen, np.full(5, -0.5)), (_quadratic, np.ones(6))]
)
def test_the_minimum_is_scipys(fun, x0) -> None:
    ours = minimize_lbfgs(fun, x0, rtol=0.0, atol=1e-10)
    theirs = minimize(fun, x0, jac=True, method='L-BFGS-B', options={'gtol': 1e-12, 'ftol': 1e-15, 'maxiter': 10_000})
    assert ours.converged, ours.reason
    assert np.max(np.abs(ours.x - theirs.x)) <= 1e-6
    assert ours.fun <= theirs.fun + 1e-12 * max(1.0, abs(theirs.fun))


def test_the_stop_is_rtol_of_the_first_gradient_or_atol() -> None:
    x0 = np.array([-1.2, 1.0])
    g0 = np.max(np.abs(rosen_der(x0)))
    relative = minimize_lbfgs(_rosen, x0, rtol=1e-3, atol=0.0)
    assert np.max(np.abs(relative.grad)) <= 1e-3 * g0
    assert relative.reason == 'rtol'
    absolute = minimize_lbfgs(_rosen, x0, rtol=0.0, atol=1e-6)
    assert np.max(np.abs(absolute.grad)) <= 1e-6
    assert absolute.reason == 'atol'
    assert absolute.iterations > relative.iterations


def test_a_wrong_gradient_is_never_reported_converged() -> None:
    """Planted control: with the gradient's sign flipped no step decreases f, and the result says so."""

    def lying(x):
        value, grad = _rosen(x)
        return value, -grad

    result = minimize_lbfgs(lying, np.array([-1.2, 1.0]), rtol=1e-8, atol=0.0)
    assert not result.converged
    assert result.reason == 'line search failed'


def test_the_iteration_budget_is_named() -> None:
    result = minimize_lbfgs(_rosen, np.array([-1.2, 1.0]), max_iter=3)
    assert isinstance(result, MinimizeResult)
    assert not result.converged
    assert result.reason == 'max_iter'
    assert result.iterations == 3


def test_settings_are_checked() -> None:
    with pytest.raises(ValueError, match='memory'):
        minimize_lbfgs(_rosen, np.zeros(2), memory=0)
    with pytest.raises(ValueError, match='rtol'):
        minimize_lbfgs(_rosen, np.zeros(2), rtol=0.0, atol=0.0)
