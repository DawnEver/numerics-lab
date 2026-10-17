"""Singular field-size systems cannot enter an unbounded rational fallback: the sparse least-squares retry refuses them."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import initial_rate


def test_large_singular_rate_requires_owned_sparse_subsystem_without_rationalizing(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('field-size singular fallback must not rationalize or densify')

    n = 513
    matrix = sp.eye(n, format='lil')
    matrix[-1, -1] = 2
    matrix[-1, 0] = 1
    matrix[-1, 1] = 1
    matrix[0, -1] = 1
    matrix[1, -1] = 1
    monkeypatch.setattr(initial_rate, 'Fraction', forbidden)
    monkeypatch.setattr(sp.csr_matrix, 'toarray', forbidden)
    with pytest.raises(ValueError, match='lack full column rank'):
        initial_rate.sparse_initial_rate(matrix.tocsr(), np.zeros(n))
