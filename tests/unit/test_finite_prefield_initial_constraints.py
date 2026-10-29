"""An explicit finite primitive, not local mass times state, owns admission."""

import numpy as np
import pytest

from numerics_lab.nonlinear.initial_constraints import nonlinear_dae_initial_state


def test_finite_storage_allows_proven_storage_row_rotation_without_changing_algebraic_rate_space():
    def mass(state):
        return np.array([[1.0, 2 * state[1]], [0.0, 0.0]])

    def residual(state):
        return np.array([0.0, state[1] - 2.0])

    def storage(state):
        return np.array([state[0] + state[1] ** 2, 0.0])

    initial = np.array([1.0, 1.0])
    state = nonlinear_dae_initial_state(
        residual,
        lambda _s: np.array([[0.0, 0.0], [0.0, 1.0]]),
        mass,
        initial,
        extra_rate_columns=np.zeros((2, 0)),
        preferred_indices=(),
        max_iter=20,
        stored_state=storage,
    )
    np.testing.assert_allclose(state, [-2.0, 2.0], atol=1e-10)
    np.testing.assert_allclose(storage(state), storage(initial), atol=1e-10)
    # A frozen initial mass would preserve x+2*y, yielding x=-1, which does not
    # preserve the declared storage. The undeclared nonlinear route must refuse.
    with pytest.raises(ValueError, match='invariant storage'):
        nonlinear_dae_initial_state(
            residual,
            lambda _s: np.array([[0.0, 0.0], [0.0, 1.0]]),
            mass,
            initial,
            extra_rate_columns=np.zeros((2, 0)),
            preferred_indices=(),
            max_iter=20,
        )
