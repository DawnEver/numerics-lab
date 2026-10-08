"""A sparse algebraic bulk does not invent unique rates in a rank-deficient border."""

import numpy as np
import pytest

from motronics.gauss.linear.algebraic_bulk_initial import algebraic_bulk_initial_pair


@pytest.mark.parametrize('inductance', [1e-9, 1.0, 1e9])
def test_shared_rank_one_storage_preserves_state_and_marks_only_free_branch_rates(inductance):
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
    initial = np.array([0.04 * inductance, 0.05, 0.02, 0.04 * inductance, -0.02 * inductance])
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
    inconsistent = np.array([0.01, 0.0])
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
