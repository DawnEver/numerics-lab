"""Owned affine coordinates retain exact storage without choosing a nonlinear state."""

import numpy as np
import pytest

from motronics.gauss.nonlinear.initial_constraints import nonlinear_dae_initial_state


@pytest.mark.parametrize('source', [0.1, -0.1, 0.3, 1e-12])
@pytest.mark.parametrize('scale', [1.0, 1e-30])
def test_affine_storage_and_tiny_independent_constraint_remain_original_equations(source, scale):
    mass = np.diag([1.0, 0.0, 0.0])
    initial = np.array([0.0, 0.0, source])

    def residual(state):
        charge, current, voltage = state
        return np.array([-current, scale * (current - charge), voltage + voltage**3 - source])

    def jacobian(state):
        return np.array([[0.0, -1.0, 0.0], [-scale, scale, 0.0], [0.0, 0.0, 1 + 3 * state[2] ** 2]])

    state = nonlinear_dae_initial_state(
        residual,
        jacobian,
        lambda _state: mass,
        initial,
        extra_rate_columns=np.empty((3, 0)),
        preferred_indices=[1],
        max_iter=30,
        stored_state=lambda value: mass @ value,
        affine_constraints=(np.array([[1.0, 0.0, 0.0], [-scale, scale, 0.0]]), np.zeros(2)),
    )
    assert state[0] == state[1] == 0.0
    np.testing.assert_array_equal(mass @ state, mass @ initial)
    assert abs(residual(state)[2]) <= 1e-10 * abs(source)


def test_affine_certificate_cannot_override_original_finite_storage():
    with pytest.raises(ValueError, match='stored initial|initial constraints'):
        nonlinear_dae_initial_state(
            lambda value: np.array([0.0, value[1]]),
            lambda _value: np.diag([0.0, 1.0]),
            lambda _value: np.diag([1.0, 0.0]),
            np.zeros(2),
            extra_rate_columns=np.empty((2, 0)),
            preferred_indices=[1],
            max_iter=3,
            stored_state=lambda value: np.array([value[0], 0.0]),
            affine_constraints=(np.array([[1.0, 0.0]]), np.array([1.0])),
        )
