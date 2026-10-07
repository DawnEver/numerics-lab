"""Preferred states change only when the actual algebraic constraints determine them."""

import numpy as np
import pytest

from motronics.gauss.linear.initial_constraints import linear_dae_initial_state


def test_storage_forces_a_unique_preferred_state_without_changing_the_stored_value():
    # x=(stored, preferred), stored_dot is present only in the first row;
    # the second row requires preferred=stored.
    result = linear_dae_initial_state(
        [[0.0, 0.0], [-1.0, 1.0]],
        [0.0, -0.3],
        [[1e-12, 0.0], [0.0, 0.0]],
        [0.3, 0.0],
        extra_rate_columns=np.zeros((2, 0)),
        preferred_indices=(1,),
    )
    np.testing.assert_allclose(result, [0.3, 0.3], atol=1e-14)


def test_an_aggregate_constraint_is_not_permission_to_guess_equal_initial_states():
    # x=(stored, preferred0, preferred1), only their sum is determined.
    with pytest.raises(ValueError, match='do not determine their split'):
        linear_dae_initial_state(
            [[0, 0, 0], [-1, 1, 1], [0, 0, 0]],
            [0, -0.3, 0],
            np.diag([1e-12, 0, 0]),
            [0.3, 0, 0],
            extra_rate_columns=np.zeros((3, 0)),
            preferred_indices=(1, 2),
        )


def test_row_unit_rescaling_does_not_change_the_left_algebraic_nullspace():
    # The dynamic rows store only x0-x1; their sum enforces x2=3.
    mass = np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    jacobian = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    residual = np.array([-3.0, -3.0, 0.0])
    for scale in (np.ones(3), np.array([1e-12, 1e8, 1.0])):
        result = linear_dae_initial_state(
            scale[:, None] * jacobian,
            scale * residual,
            scale[:, None] * mass,
            np.zeros(3),
            extra_rate_columns=np.zeros((3, 0)),
            preferred_indices=(2,),
        )
        assert result[2] == pytest.approx(3.0, abs=1e-12)
