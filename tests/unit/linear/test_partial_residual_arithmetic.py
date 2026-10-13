"""Sparse row arithmetic cannot borrow certainty or scale from another row."""

import numpy as np
import scipy.sparse as sp

from motronics.contracts.runtime.ports import ResidualAssemblyBound
from motronics.gauss.linear.arithmetic import matrix_vector_roundoff, transport_residual_bound


def test_actual_mapping_preserves_partial_known_authority_and_counts_its_own_arithmetic():
    matrix = sp.csr_matrix([[0, 2], [1, 1]])
    source = ResidualAssemblyBound(np.array([0.0, 1e-17]), np.array([False, True]))
    result = transport_residual_bound(matrix, np.array([1.0, -1.0]), source)
    np.testing.assert_array_equal(result.known, [True, False])
    assert result.values[0] > 2e-17


def test_non_cancelling_tiny_row_has_no_floor_from_unrelated_large_row():
    result = matrix_vector_roundoff(sp.diags([1e9, 1e-30]), np.ones(2), np.zeros(2))
    assert result.known.all()
    assert result.values[1] < 1e-44


def test_underflowed_nonzero_product_is_not_a_zero_certification():
    result = matrix_vector_roundoff(sp.csr_matrix([[1e-308]]), np.array([1e-308]), np.zeros(1))
    assert not result.known[0]
