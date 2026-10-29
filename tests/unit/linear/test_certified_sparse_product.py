"""A product coefficient inside its own rounding bound carries no dependence.

Planted from the case_a dual-star accepted projection (2026-10-12): a solid
conductor's voltage column lies in the range of the eddy mass, so every
left-null storage row annihilates it EXACTLY; the float product left 7e-15
against a 67 magnitude sum, and the rate LU pivoted on it (pivots 1e-22,
rates 1e26).
"""

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear.arithmetic import certified_sparse_product


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


def test_a_huge_sparse_product_is_judged_on_its_own_entries_only():
    # MEASURED 2026-10-12 on case_a voltage-driven: `abs(product) <= bound` on two sparse
    # matrices is TRUE at every implicit zero, so the comparison materialised n^2 entries (76 GiB).
    n = 2_000_000
    left = sp.identity(n, format='csr') * 2.0
    product = certified_sparse_product(left, left)
    assert product.nnz == n
    assert product[n - 1, n - 1] == 4.0
