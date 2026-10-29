"""Actual nonlinear constraints preserve stored coordinates and preferred authority."""

import numpy as np
import pytest

from numerics_lab.nonlinear.initial_constraints import nonlinear_dae_initial_state


def residual(state):
    return np.array([0.0, state[1] - state[0], state[2] ** 3 - 8.0])


def jacobian(state):
    return np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 3 * state[2] ** 2]])


def test_shared_newton_solves_actual_nonlinear_ic_without_replacing_stored_state():
    state = nonlinear_dae_initial_state(
        residual,
        jacobian,
        lambda _current: np.diag([1e-12, 0, 0]),
        [0.3, 0, 1],
        extra_rate_columns=np.zeros((3, 0)),
        preferred_indices=(1,),
        max_iter=20,
    )
    np.testing.assert_allclose(state, [0.3, 0.3, 2.0], atol=1e-10)


def test_nonlinear_ic_does_not_freeze_a_rotating_storage_coordinate():
    with pytest.raises(ValueError, match='invariant storage/rate space'):
        nonlinear_dae_initial_state(
            residual,
            jacobian,
            lambda current: np.array([[1, current[2], 0], [0, 0, 0], [0, 0, 0]]),
            [0.3, 0, 1],
            extra_rate_columns=np.zeros((3, 0)),
            preferred_indices=(1,),
            max_iter=20,
        )


def test_a_seed_already_at_the_root_is_accepted_not_driven_below_roundoff():
    """A seed already on its roundoff floor is the answer, not a bar of roundoff squared.

    The Newton stop used to be relative to the SEED residual, so a seed on its floor demanded
    ~1e-10 of a residual no step can reduce and never converged (twelve_pulse_thyristor's
    held-current state, 2026-10-12). It is judged on the source scale acceptance already uses.
    """

    def on_its_floor(state):
        # 2e-15 is under the residual's float granularity at x = 2 (12 * 4.4e-16): no step moves it.
        return residual(state) + np.array([0.0, 0.0, 2e-15])

    state = nonlinear_dae_initial_state(
        on_its_floor,
        jacobian,
        lambda _current: np.diag([1e-12, 0, 0]),
        [0.3, 0.3, 2.0],
        extra_rate_columns=np.zeros((3, 0)),
        preferred_indices=(1,),
        max_iter=20,
    )
    np.testing.assert_allclose(state, [0.3, 0.3, 2.0], atol=1e-12)
