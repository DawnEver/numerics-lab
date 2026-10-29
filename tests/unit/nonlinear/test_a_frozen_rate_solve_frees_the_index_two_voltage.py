"""The frozen-state rate projection frees a solid body's index-two voltage instead of pivoting on rounding.

Planted DAE (`test_an_index_two_solid_voltage_is_state_authority`'s, at non-binary coefficients):

    0.1 a1' + 3 a1 - 0.1 u = 1
    0.2 a2' + 5 a2 - 0.2 u = 1
    0.3 u - (0.1 a1' + 0.2 a2') = 0        (zero net current)

``Y = (1, 1, 1)`` annihilates the mass and, exactly, the ``u`` column -- but in floating point
``-0.1 - 0.2 + 0.3`` is 5.6e-17, not zero. That residue made the state search look determined, so
the admission took the frozen-state branch and pivoted the voltage's RATE on it: ``u'`` = 2.3e16
with ``u`` left at the seed's 0 (case_a at its magnetostatic rest: 1.2e21 on its six magnet bodies).
The certified hidden rows drop the residue and the voltage is re-solved as state:
``u = -5/32``, ``a' = (75/32, -45/32)``, its rate free.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.storage_border import StorageRows
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair

pytestmark = pytest.mark.unit

_M = sp.csr_matrix([[0.1, 0, 0], [0, 0.2, 0], [-0.1, -0.2, 0]])
_J = sp.csr_matrix([[3.0, 0, -0.1], [0, 5.0, -0.2], [0, 0, 0.3]])
_F = np.array([1.0, 1.0, 0.0])


@pytest.mark.parametrize('stored_columns', [None, np.array([0, 1])])
def test_the_voltage_is_state_and_its_rate_is_never_claimed(stored_columns):
    assert (sp.csr_matrix([[1.0, 1.0, 1.0]]) @ _J)[0, 2] != 0.0  # the residue this guards against
    pair = finite_storage_actual_rate_initial_pair(
        np.array([0.25, 0.25, 0.0]),
        residual_at=lambda state, rate: _J @ state + _M @ rate - _F,
        jacobian_at=lambda _state, _rate: _J,
        mass_at=lambda _state, _rate: _M,
        partial_time_at=lambda _state, _rate: np.zeros(3),
        storage_at=lambda state: _M @ state,
        storage_rows_at=lambda _state, _rate: StorageRows(np.array([0, 1]), sp.csr_matrix([[1.0, 1.0, 1.0]])),
        rate_constraint_jacobian_at=lambda _state, _rate: _J,
        stored_columns_at=None if stored_columns is None else (lambda _state, _rate, _storage: stored_columns),
    )
    np.testing.assert_allclose(pair.state, [0.25, 0.25, -5 / 32], rtol=1e-14)
    np.testing.assert_allclose(pair.rate[:2], [75 / 32, -45 / 32], rtol=1e-14)
    assert pair.rate_is_unique.tolist() == [True, True, False]
    np.testing.assert_allclose(_J @ pair.state + _M @ pair.rate - _F, 0.0, atol=1e-15)
