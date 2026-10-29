"""The state search judges a boundary column on the CERTIFIED hidden rows, never on their rounding.

Planted DAE (`test_a_frozen_rate_solve_frees_the_index_two_voltage`'s), now with the two linkages
declared as the sparse field coordinates and the solid body's voltage ``u`` as a BOUNDARY column:

    0.1 a1' + 3 a1 - 0.1 u = 1
    0.2 a2' + 5 a2 - 0.2 u = 1
    0.3 u - (0.1 a1' + 0.2 a2') = 0        (zero net current)

``Y = (1, 1, 1)`` annihilates ``u``'s column exactly; in floating point ``Y J[:, u]`` is 5.6e-17.
Judged on that product, the state search called ``u`` independent and owned it as STATE authority.
The incident: case_a `voltage_driven`, REST at its magnetostatic root (2026-10-16). Four of six
solid-body voltage columns sat at 3.6e-15..2.1e-14, so only two boundary directions were exported, and
the march refused at t=0 with "joint stored-coordinate equations require additional physical state
authority". Certified, the column is empty and ``u`` goes to the joint state/rate path.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import numerics_lab.nonlinear.finite_storage as runtime
from numerics_lab.linear.storage_border import StorageRows
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair

pytestmark = pytest.mark.unit

_M = sp.csr_matrix([[0.1, 0, 0], [0, 0.2, 0], [-0.1, -0.2, 0]])
_J = sp.csr_matrix([[3.0, 0, -0.1], [0, 5.0, -0.2], [0, 0, 0.3]])
_F = np.array([1.0, 1.0, 0.0])
_Y = sp.csr_matrix([[1.0, 1.0, 1.0]])
_SEED = np.array([0.25, 0.25, 0.0])


def test_a_rounding_residue_never_makes_a_boundary_voltage_state_determined():
    assert (_Y @ _J)[0, 2] != 0.0  # the residue this guards against
    with pytest.raises(runtime.StorageRateAuthorityRequired) as authority:
        runtime._actual_finite_storage_state(
            _SEED,
            _M @ _SEED,
            lambda state, rate: _J @ state + _M @ rate - _F,
            lambda _state, _rate: _J,
            lambda _state, _rate: _M,
            lambda state: _M @ state,
            lambda _state, _rate: StorageRows(np.array([0, 1]), _Y),
            20,
            1e-10,
            None,
            np.array([0, 1]),
        )
    # The one boundary column, exported as a homogeneous state direction for the joint path.
    assert authority.value.directions is not None
    np.testing.assert_array_equal(
        np.asarray(sp.csr_matrix(authority.value.directions).todense()), [[0.0], [0.0], [1.0]]
    )


def test_the_joint_path_solves_the_index_two_voltage_from_its_rates():
    """The net-current row's Schur entry (-5.6e-17 against its 2.0e-16 bound) is a possible zero, so it anchors
    nothing (`border_rank.homogeneous_zero_coordinates`, 2026-10-17) and the driven row 8 u = -1.25 solves u.
    """
    pair = finite_storage_actual_rate_initial_pair(
        _SEED,
        residual_at=lambda state, rate: _J @ state + _M @ rate - _F,
        jacobian_at=lambda _state, _rate: _J,
        mass_at=lambda _state, _rate: _M,
        partial_time_at=lambda _state, _rate: np.zeros(3),
        storage_at=lambda state: _M @ state,
        storage_rows_at=lambda _state, _rate: StorageRows(np.array([0, 1]), _Y),
        rate_constraint_jacobian_at=lambda _state, _rate: _J,
        sparse_state_indices=np.array([0, 1]),
    )
    np.testing.assert_allclose(pair.state, [0.25, 0.25, -5 / 32], rtol=1e-14)
    np.testing.assert_allclose(pair.rate[:2], [75 / 32, -45 / 32], rtol=1e-14)
