"""Sparse row arithmetic cannot borrow certainty or scale from another row."""

from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from numerics_lab.formulation import ResidualAssemblyBound
from numerics_lab.linear.arithmetic import matrix_vector_roundoff, rounded_defect, transport_residual_bound


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


def test_rounded_defect_sees_the_last_ulp_a_working_precision_residual_rounds_away():
    # 3 - (0.9999999999999999 + 2) is 0.0 in doubles; the exact defect is 2**-53, and it is the
    # only signal an iterative refinement has that the answer is one ulp off.
    matrix = sp.csr_matrix([[1.0, 1.0], [1e18, 0.0]])
    value = np.array([np.nextafter(1.0, 0.0), 2.0])
    rhs = np.array([3.0, 1e18])
    assert (rhs - matrix @ value)[0] == 0.0
    np.testing.assert_array_equal(rounded_defect(matrix, value, rhs), [2.0**-53, 1e18 * 2.0**-53])


def test_rounded_defect_carries_each_products_own_rounding():
    # 0.1 * 3 rounds; the defect of the rounded product against itself is that rounding exactly.
    product = 0.1 * 3.0
    defect = rounded_defect(sp.csr_matrix([[0.1]]), np.array([3.0]), np.array([product]))
    assert defect[0] != 0.0
    assert Fraction(float(defect[0])) == Fraction(product) - Fraction(0.1) * 3
