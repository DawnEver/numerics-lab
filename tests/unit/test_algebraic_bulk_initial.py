"""A sparse algebraic bulk does not invent unique rates in a rank-deficient border."""

import numpy as np
import pytest

from numerics_lab.linear.algebraic_bulk_initial import algebraic_bulk_initial_pair


@pytest.mark.parametrize('inductance', [1e-9, 1.0, 1e9])
@pytest.mark.parametrize('state_scale', [1.0, 1e3])
def test_shared_rank_one_storage_preserves_state_and_marks_only_free_branch_rates(inductance, state_scale):
    jacobian = np.array(
        [
            [1 / inductance, -1, 0.5, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 5, 0, 0],
            [-1, 0, 0, 1, 0],
            [0.5, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    mass = np.zeros((5, 5))
    mass[1, 3] = mass[2, 4] = 1
    initial = state_scale * np.array([0.04 * inductance, 0.05, 0.02, 0.04 * inductance, -0.02 * inductance])
    pair = algebraic_bulk_initial_pair(
        jacobian,
        np.zeros(5),
        mass,
        initial,
        bulk_size=1,
        partial_time=np.zeros(5),
        stored_increment=np.zeros(2),
        tolerance=1e-10,
    )
    assert np.array_equal(pair.state, initial)
    assert pair.rate_is_unique.tolist() == [True, False, False, True, True]
    assert np.isnan(pair.physical_rate[1:3]).all()
    assert np.array_equal(pair.physical_rate[[0, 3, 4]], np.zeros(3))
    fixed = algebraic_bulk_initial_pair(
        jacobian,
        np.zeros(5),
        mass,
        initial,
        bulk_size=1,
        partial_time=np.zeros(5),
        stored_increment=np.zeros(2),
        tolerance=1e-10,
        fixed_state=True,
    )
    assert np.array_equal(fixed.state, initial)
    assert np.array_equal(fixed.rate_is_unique, pair.rate_is_unique)
    cancelled = np.zeros(5)
    cancelled[2] = np.spacing(0.1)
    roundoff = algebraic_bulk_initial_pair(
        jacobian,
        cancelled,
        mass,
        initial,
        bulk_size=1,
        partial_time=np.zeros(5),
        stored_increment=np.zeros(2),
        tolerance=1e-10,
        fixed_state=True,
    )
    assert np.array_equal(roundoff.state, initial)
    assert np.array_equal(roundoff.physical_rate[[0, 3, 4]], np.zeros(3))
    cancelled[2] = 0.001
    with pytest.raises(ValueError, match='contradicts'):
        algebraic_bulk_initial_pair(
            jacobian,
            cancelled,
            mass,
            initial,
            bulk_size=1,
            partial_time=np.zeros(5),
            stored_increment=np.zeros(2),
            tolerance=1e-10,
            fixed_state=True,
        )
    impossible_partial = np.zeros(5)
    impossible_partial[4] = 0.001
    with pytest.raises(ValueError, match='contradicts'):
        algebraic_bulk_initial_pair(
            jacobian,
            np.zeros(5),
            mass,
            initial,
            bulk_size=1,
            partial_time=impossible_partial,
            stored_increment=np.zeros(2),
            tolerance=1e-10,
            fixed_state=True,
        )
    # The stored coordinate is a FLUX (~ inductance x current), so a contradiction is a flux increment
    # that forces a CURRENT change: 0.01 x inductance moves the bulk current by 0.01 at every scale.
    # A bare 0.01 moved it by 1e-11 at inductance 1e9 -- under the 1e-10 bar, so not a contradiction.
    inconsistent = np.array([0.01 * inductance, 0.0])
    with pytest.raises(ValueError, match='contradicts'):
        algebraic_bulk_initial_pair(
            jacobian,
            np.zeros(5),
            mass,
            initial,
            bulk_size=1,
            partial_time=np.zeros(5),
            stored_increment=inconsistent,
            tolerance=1e-10,
        )
    mass[0, 0] = 0.01
    with pytest.raises(ValueError, match='storage-free'):
        algebraic_bulk_initial_pair(
            jacobian,
            np.zeros(5),
            mass,
            initial,
            bulk_size=1,
            partial_time=np.zeros(5),
            stored_increment=np.zeros(2),
            tolerance=1e-10,
        )
