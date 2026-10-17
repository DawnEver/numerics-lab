"""Shooting solves the cycle map, not an endpoint-only convergence heuristic."""

import numpy as np
import pytest

from numerics_lab.time_integrators.shooting import periodic_shooting


def test_affine_cycle_map_returns_its_fixed_point():
    result = periodic_shooting(lambda u: 0.5 * u + np.array([1.0, 2.0]), np.zeros(2))
    np.testing.assert_allclose(result.state, [2.0, 4.0], rtol=1e-7, atol=1e-9)
    assert result.residual_norm < 1e-7


def test_noncontractive_cycle_map_is_still_solved():
    result = periodic_shooting(lambda u: 2 * u + 3, np.zeros(1))
    np.testing.assert_allclose(result.state, [-3.0], atol=1e-7)


def test_no_periodic_state_does_not_report_success():
    with pytest.raises(RuntimeError, match='shooting'):
        periodic_shooting(lambda u: u + 1, np.zeros(1), max_iterations=3)


def test_invalid_cycle_map_is_refused():
    with pytest.raises(ValueError, match='cycle map'):
        periodic_shooting(lambda _u: np.array([np.nan]), np.zeros(1))
