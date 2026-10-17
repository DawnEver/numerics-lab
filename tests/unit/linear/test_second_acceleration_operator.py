"""First and second differentiated equations retain distinct owned operators."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.initial_sparse import SparseInitialPair, sparse_complete_initial_rate


def test_second_acceleration_operator_does_not_replace_first_derivative_operator():
    first = sp.csr_matrix([[0.0, -1.0], [1.0, 0.0]])
    second = sp.csr_matrix([[0.0, -1.0], [2.0, 0.0]])
    pair = SparseInitialPair(np.zeros(2), np.array([1.0, 0.0]), np.array([True, False]))
    completed = sparse_complete_initial_rate(
        first,
        np.array([-1.0, 0.0]),
        sp.diags([1.0, 0.0], format='csr'),
        pair,
        algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
        partial_time=np.array([0.0, -1.0]),
        acceleration_remainder=np.array([0.0, -3.0]),
        acceleration_jacobian=second,
    )
    np.testing.assert_array_equal(completed.rate, [1.0, 1.5])
    assert completed.rate_is_unique.all()


@pytest.mark.parametrize(
    'second',
    [np.eye(3), np.eye(2, dtype=bool), np.eye(2, dtype=complex), np.diag([np.nan, 1.0]), np.diag([np.inf, 1.0])],
)
def test_second_acceleration_operator_rejects_unknown_numeric_authority(second):
    pair = SparseInitialPair(np.zeros(2), np.array([1.0, 0.0]), np.array([True, False]))
    with pytest.raises(ValueError, match='second acceleration operator'):
        sparse_complete_initial_rate(
            sp.eye(2, format='csr'),
            np.zeros(2),
            sp.diags([1.0, 0.0], format='csr'),
            pair,
            algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
            partial_time=np.zeros(2),
            acceleration_remainder=np.zeros(2),
            acceleration_jacobian=second,
        )
