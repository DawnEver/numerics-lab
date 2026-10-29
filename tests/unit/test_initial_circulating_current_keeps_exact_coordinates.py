"""A normalized numerical nullspace cannot create exact physical pin equations."""

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear.storage_border import storage_border_columns, storage_border_rows
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair


def test_conducting_storage_and_a_field_owned_zero_inductance_branch_admit_the_actual_pair():
    # Coordinates: field, voltage, source current, winding current, linkage.
    jacobian = sp.csr_matrix([
        [3.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 2.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 1.0],
    ])
    mass = sp.csr_matrix(([2.0, 1.0], ([0, 3], [0, 4])), shape=(5, 5))
    initial = np.array([0.4, 0.0, 0.0, 0.0, 0.4])
    bulk, border = np.array([0]), np.arange(1, 5)
    pair = finite_storage_actual_rate_initial_pair(
        initial,
        residual_at=lambda state, rate: jacobian @ state + mass @ rate,
        jacobian_at=lambda _state, _rate: jacobian,
        mass_at=lambda _state, _rate: mass,
        storage_at=lambda state: mass @ state,
        partial_time_at=lambda _state, _rate: np.zeros(5),
        storage_rows_at=lambda _state, _rate: storage_border_rows(mass, bulk_indices=bulk, border_rows=border),
        rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
        stored_columns_at=lambda _state, _rate, storage: storage_border_columns(
            mass, bulk_indices=bulk, stored_rows=storage.stored_rows
        ),
        sparse_state_indices=bulk,
    )
    np.testing.assert_allclose(pair.state, [0.4, 0.0, -0.24, 0.24, 0.4], rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(pair.rate[[0, 4]], [-0.48, -0.48], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(jacobian @ pair.state + mass @ pair.rate, 0.0, atol=5e-16)
    np.testing.assert_array_equal(mass @ pair.state, mass @ initial)
