"""Multiple-shooting and collocation defects vanish on a trajectory and converge at their orders."""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.time_integrators.collocation import hermite_simpson_defects, trapezoidal_defects
from numerics_lab.time_integrators.shooting import shooting_defects

_RATE = np.array([[0.0, 1.0], [-4.0, -0.2]])


def _exact(t):
    """A trajectory of x' = _RATE x, from x(0) = (1, 0), by the matrix exponential."""
    values, vectors = np.linalg.eig(_RATE)
    coefficients = np.linalg.solve(vectors, [1.0, 0.0])
    return np.real(np.stack([vectors @ (coefficients * np.exp(values * s)) for s in np.atleast_1d(t)]))


def _propagate(x, t0, t1):
    values, vectors = np.linalg.eig(_RATE)
    return np.real(vectors @ (np.exp(values * (t1 - t0)) * np.linalg.solve(vectors, x)))


def test_shooting_defects_vanish_on_a_trajectory_and_see_a_displaced_node() -> None:
    times = np.linspace(0.0, 2.0, 5)
    states = _exact(times)
    np.testing.assert_allclose(shooting_defects(_propagate, states, times), 0.0, atol=1e-12)
    states[2, 0] += 1e-3  # planted control
    defects = shooting_defects(_propagate, states, times)
    assert np.abs(defects[1]).max() > 1e-4
    assert np.abs(defects[2]).max() > 1e-4
    assert np.abs(defects[[0, 3]]).max() < 1e-12


@pytest.mark.parametrize(('defects', 'order'), [(trapezoidal_defects, 3), (hermite_simpson_defects, 5)])
def test_the_local_defect_converges_at_its_order(defects, order) -> None:
    f = lambda _t, x: _RATE @ x  # noqa: E731
    errors = []
    for h in (0.1, 0.05):
        times = np.array([0.3, 0.3 + h])
        errors.append(np.abs(defects(f, _exact(times), times)).max())
    observed = np.log2(errors[0] / errors[1])
    assert abs(observed - order) < 0.3


def test_controls_enter_the_dynamics() -> None:
    f = lambda _t, x, u: -x + u  # noqa: E731
    times = np.array([0.0, 0.5])
    states = np.array([[0.0], [1.0 - np.exp(-0.5)]])  # x' = -x + 1 from 0
    with_ones = hermite_simpson_defects(f, states, times, controls=np.ones((2, 1)))
    assert abs(with_ones).max() < 1e-4
    assert abs(hermite_simpson_defects(f, states, times, controls=np.zeros((2, 1)))).max() > 0.1


def test_a_malformed_grid_is_refused() -> None:
    with pytest.raises(ValueError, match='increase'):
        trapezoidal_defects(lambda _t, x: x, np.zeros((2, 1)), np.array([1.0, 0.0]))
