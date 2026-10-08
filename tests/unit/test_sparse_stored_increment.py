"""A finite stored-coordinate correction is not a zero-storage perturbation."""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.gauss.linear.initial_sparse import sparse_initial_pair


@pytest.mark.parametrize('capacitance', [1.0, 1e-9])
def test_nonzero_stored_increment_preserves_all_original_dependent_charge_rows(capacitance):
    mass = sp.csr_matrix([[capacitance, 0], [-capacitance, 0]])
    pair = sparse_initial_pair(
        sp.eye(2, format='csr'),
        np.zeros(2),
        mass,
        np.zeros(2),
        algebraic_rows=sp.csr_matrix([[1.0, 1.0]]),
        stored_rows=np.array([0]),
        partial_time=np.zeros(2),
        stored_increment=np.array([2 * capacitance]),
    )
    np.testing.assert_allclose(pair.state, [2, -2], rtol=1e-14, atol=0)
    np.testing.assert_allclose(mass @ pair.state, [2 * capacitance, -2 * capacitance], rtol=1e-14, atol=0)
    np.testing.assert_allclose(pair.state + mass @ pair.rate, 0, atol=1e-14)
