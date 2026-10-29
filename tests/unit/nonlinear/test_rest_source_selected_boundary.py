"""A declared zero charge does not imply zero source-selected algebraic currents."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.storage_border import StorageRows
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair


@pytest.mark.parametrize('voltage', [0.1, -0.3, 1e-12])
@pytest.mark.parametrize(('stiffness', 'linkage'), [(3.0, 1.0), (1e6, 0.05)])
def test_rest_charge_and_nonadmitted_source_voltage_have_a_boundary_particular(voltage, stiffness, linkage):
    # Coordinates A,v,i0,i1,lambda0,lambda1. The two linkage witnesses
    # freeze A=0, but only the source-selected voltage equations fix currents.
    jacobian = sp.csr_matrix([
        [stiffness, 0, -1, 0.5, 0, 0],
        [0, 1, -2, 0, 0, 0],
        [0, 1, 0, -5, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [-linkage, 0, 0, 0, 1, 0],
        [0.5 * linkage, 0, 0, 0, 0, 1],
    ])
    mass = sp.csr_matrix(([1.0, 1.0], ([1, 2], [4, 5])), shape=(6, 6))
    forcing = np.array([0, 0, 0, voltage, 0, 0])
    algebraic = sp.eye(6, format='csr')[[0, 3, 4, 5]]
    pair = finite_storage_actual_rate_initial_pair(
        np.zeros(6),
        residual_at=lambda state, rate: jacobian @ state + mass @ rate - forcing,
        jacobian_at=lambda _state, _rate: jacobian,
        mass_at=lambda _state, _rate: mass,
        storage_at=lambda state: mass @ state,
        partial_time_at=lambda _state, _rate: np.zeros(6),
        storage_rows_at=lambda _state, _rate: StorageRows(np.array([1, 2]), algebraic),
        stored_columns_at=lambda _state, _rate, _storage: np.array([4, 5]),
        sparse_state_indices=np.array([0]),
        rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
    )
    np.testing.assert_array_equal(mass @ pair.state, np.zeros(6))
    np.testing.assert_allclose(pair.state, [0, voltage, 3 * voltage / 22, 3 * voltage / 11, 0, 0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(pair.rate[[4, 5]], [-8 * voltage / 11, 4 * voltage / 11], rtol=1e-12, atol=0)
    original = jacobian @ pair.state + mass @ pair.rate - forcing
    scale = abs(jacobian) @ abs(pair.state) + abs(mass) @ abs(pair.rate) + abs(forcing)
    assert np.all(abs(original) <= 1e-10 * scale)
