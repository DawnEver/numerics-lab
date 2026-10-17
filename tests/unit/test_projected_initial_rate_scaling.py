"""Differentiated physical constraints do not depend on arbitrary null-basis units."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.initial_sparse import projected_rate_consistent
from numerics_lab.linear.storage_border import StorageRows
from numerics_lab.nonlinear.finite_storage import finite_storage_initial_pair


@pytest.mark.parametrize('scale', [1e-9, 1.0, 1e9])
def test_same_actual_rate_constraints_admit_identically_after_null_basis_rescaling(scale):
    mass = sp.csr_matrix([[1.0, 0], [0, 0]])
    jacobian = sp.csr_matrix([[0, 0], [-3.0, 7.0]])
    pair = finite_storage_initial_pair(
        np.zeros(2),
        residual_at=lambda state, rate: mass @ rate + jacobian @ state + np.array([-0.2, -0.1]),
        jacobian_at=lambda _state: jacobian,
        mass_at=lambda _state: mass,
        storage_at=lambda state: mass @ state,
        partial_time_at=lambda _state: np.zeros(2),
        storage_rows=StorageRows(np.array([0]), sp.csr_matrix([[0, scale]])),
    )
    np.testing.assert_allclose(pair.state, [0, 0.1 / 7], rtol=1e-14, atol=0)
    np.testing.assert_allclose(pair.rate, [0.2, 0.6 / 7], rtol=1e-14, atol=0)


@pytest.mark.parametrize('scale', [1e-9, 1.0, 1e9])
def test_inconsistent_actual_storage_rate_is_not_admitted_by_null_basis_rescaling(scale):
    mass = sp.csr_matrix([[1.0, 0], [0, 0]])
    jacobian = sp.csr_matrix([[0, 0], [1.0, 0]])
    with pytest.raises(ValueError, match='contradicts|physical authority|constraint subsystem'):
        finite_storage_initial_pair(
            np.zeros(2),
            residual_at=lambda state, rate: mass @ rate + jacobian @ state + np.array([1.0, 0]),
            jacobian_at=lambda _state: jacobian,
            mass_at=lambda _state: mass,
            storage_at=lambda state: mass @ state,
            partial_time_at=lambda _state: np.zeros(2),
            storage_rows=StorageRows(np.array([0]), sp.csr_matrix([[0, scale]])),
        )


@pytest.mark.parametrize('scale', [1e-9, 1.0, 1e9])
def test_actual_differentiated_rate_contradiction_is_refused_at_every_null_basis_scale(scale):
    assert not projected_rate_consistent(
        sp.csr_matrix([[0, 0], [-3.0, 7.0]]),
        np.array([0.2, 0.6 / 7 + 0.1]),
        np.zeros(2),
        algebraic_rows=sp.csr_matrix([[0, scale]]),
        tolerance=1e-10,
    )


def test_a_large_other_row_cannot_hide_a_small_but_contradictory_differentiated_constraint():
    assert not projected_rate_consistent(
        sp.eye(2, format='csr'),
        np.array([1.0, 1.0]),
        np.array([-1.0, 0.0]),
        algebraic_rows=sp.diags([1e9, 1e-9], format='csr'),
        tolerance=1e-10,
    )
