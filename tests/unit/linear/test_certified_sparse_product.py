"""A product coefficient inside its own rounding bound carries no dependence.

Planted from the Cummins dual-star accepted projection (2026-10-12): a solid
conductor's voltage column lies in the range of the eddy mass, so every
left-null storage row annihilates it EXACTLY; the float product left 7e-15
against a 67 magnitude sum, and the rate LU pivoted on it (pivots 1e-22,
rates 1e26).
"""

import numpy as np
import scipy.sparse as sp

from motronics.gauss.linear.arithmetic import certified_sparse_product


def test_a_cancellation_residue_is_not_a_coefficient():
    left = sp.csr_matrix(np.array([[0.1, 0.2, -0.3]]))
    right = sp.csr_matrix(np.ones((3, 1)))
    assert (left @ right).nnz == 1  # the float product keeps the residue
    assert certified_sparse_product(left, right).nnz == 0


def test_a_small_coefficient_above_its_own_bound_survives():
    left = sp.csr_matrix(np.array([[1e-14, 2e-14], [3.0, 0.0]]))
    right = sp.csr_matrix(np.array([[1.0, 0.5], [1.0, 0.0]]))
    product = certified_sparse_product(left, right)
    assert product.toarray().tolist() == (left @ right).toarray().tolist()


def test_certifying_a_large_sparse_product_never_forms_its_dense_pattern():
    """MEASURED 2026-10-13: `abs(product) <= bound` between two sparse matrices is True where both
    are absent, so scipy built an all-True dense matrix of the product's shape -- 140 GiB on the
    transformer_3ph_concentric network march's accepted projection. Certification reads only the
    entries the product has.
    """
    n = 10**6
    left = sp.identity(n, format='csr')
    right = sp.csr_matrix((np.array([0.1, 2.0]), (np.array([0, 5]), np.array([0, 7]))), shape=(n, n))
    product = certified_sparse_product(left, right)
    assert product.shape == (n, n)
    assert product.nnz == 2
    assert product[5, 7] == 2.0
