"""Preferred states change only when the actual algebraic constraints determine them."""

import numpy as np
import pytest

from numerics_lab.linear.initial_constraints import linear_dae_initial_pair, linear_dae_initial_state


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


def test_series_stored_states_determine_initial_algebraic_voltage_and_rates():
    # x=(current0, port_voltage, current1), L0=.5, L1=.25,
    # V=3, R=2. The algebraic series-current constraint must also
    # hold for rates, rather than choosing separate arbitrary di/dt.
    pair = linear_dae_initial_pair(
        [[0, 1, 0], [2, -1, 0], [1, 0, -1]],
        [-3, 0.6, 0],
        [[0, 0, 0.25], [0.5, 0, 0], [0, 0, 0]],
        [0.3, 0, 0.3],
        partial_time=[0, 0, 0],
    )
    np.testing.assert_allclose(pair.state, [0.3, 2.2, 0.3], atol=1e-13)
    np.testing.assert_allclose(pair.rate[[0, 2]], [3.2, 3.2], atol=1e-13)
    np.testing.assert_array_equal(pair.rate_is_unique, [True, False, True])


def test_source_derivative_reaches_initial_storage_rate_without_a_zero_slope_default():
    # A stored x is constrained to a declared time-dependent source.
    pair = linear_dae_initial_pair([[0, 1], [1, 0]], [0, 0], [[1e-12, 0], [0, 0]], [3, 0], partial_time=[0, -4])
    np.testing.assert_allclose(pair.state, [3, -4e-12], atol=1e-14)
    assert pair.rate[0] == pytest.approx(4.0)
    np.testing.assert_array_equal(pair.rate_is_unique, [True, False])
    assert not pair.rate_is_unique.flags.writeable
    assert np.isnan(pair.physical_rate[1])
    assert pair.physical_rate[0] == pytest.approx(4.0)
