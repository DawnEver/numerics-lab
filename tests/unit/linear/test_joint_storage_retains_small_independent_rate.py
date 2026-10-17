"""A small dependent storage rate is not reconstructed by subtracting large forcing."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.initial_rate import normalize_sparse_rows, sparse_initial_rate
from numerics_lab.linear.initial_storage_border import _frozen_rate_forcing, stored_initial_correction


@pytest.mark.parametrize('coefficient', [1e-300, 1e300])
def test_extreme_finite_rate_row_scale_retains_its_actual_equation(coefficient):
    rate, unique = sparse_initial_rate(sp.csr_matrix([[coefficient]]), np.array([2 * coefficient]))
    np.testing.assert_allclose(rate, [2.0], rtol=1e-14)
    assert unique.all()


@pytest.mark.parametrize(
    ('row', 'rhs'),
    [([1e-308], [2.0]), ([1e308, 1e-308], [1.0]), ([1e308], [1e-308])],
)
def test_unrepresentable_row_normalization_refuses_to_drop_finite_equations(row, rhs):
    with np.errstate(over='ignore', under='ignore'), pytest.raises(ValueError, match='representable'):
        normalize_sparse_rows(sp.csr_matrix([row]), np.array(rhs))


def test_actual_rhs_bound_is_row_local_and_does_not_hide_tiny_independent_forcing():
    jacobian = sp.csr_matrix([[1.0, 0.0], [0.0, 0.0]])
    forcing, bound = _frozen_rate_forcing(
        jacobian, np.array([-1.0, 1e-30]), np.array([1.0, 0.0]), sp.eye(2, format='csr'), np.zeros(2), 1e-10
    )
    assert bound[0] > 0
    assert bound[1] == 0
    with pytest.raises(ValueError, match='contradict'):
        sparse_initial_rate(sp.csr_matrix([[1.0], [0.0]]), forcing, residual_allowance=bound)


@pytest.mark.parametrize(('coefficient', 'correction'), [(1e-300, 1e-100), (1e-300, 1e-10), (1e308, 2.0)])
def test_actual_forcing_refuses_underflowed_or_overflowed_products(coefficient, correction):
    with np.errstate(over='ignore', under='ignore'), pytest.raises(ValueError, match='finite products'):
        _frozen_rate_forcing(
            sp.csr_matrix([[coefficient]]),
            np.zeros(1),
            np.array([correction]),
            sp.eye(1, format='csr'),
            np.zeros(1),
            1e-10,
        )


def test_actual_forcing_refuses_an_underflowed_arithmetic_bound():
    with pytest.raises(ValueError, match='underflowed arithmetic bound'):
        _frozen_rate_forcing(
            sp.csr_matrix([[np.finfo(float).tiny]]),
            np.zeros(1),
            np.ones(1),
            sp.eye(1, format='csr'),
            np.zeros(1),
            1e-10,
        )


def test_a_real_forcing_contradiction_above_its_construction_bound_is_rejected():
    forcing, bound = _frozen_rate_forcing(
        sp.csr_matrix([[1.0]]), np.array([-1.0 + 1e-12]), np.ones(1), sp.eye(1, format='csr'), np.zeros(1), 1e-10
    )
    assert abs(forcing[0]) > 100 * bound[0]
    with pytest.raises(ValueError, match='contradict'):
        sparse_initial_rate(
            sp.csr_matrix([[1.0], [1.0]]),
            np.concatenate(([0.0], forcing)),
            residual_allowance=np.concatenate(([0.0], bound)),
        )


def test_rate_kernel_does_not_mutate_the_owned_sparse_operator():
    matrix = sp.csr_matrix(([1.0, 0.0, 1.0], ([0, 0, 1], [0, 1, 1])), shape=(2, 2))
    matrix.data.flags.writeable = False
    rate, unique = sparse_initial_rate(matrix, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(rate, [1.0, 2.0])
    assert unique.all()
    assert matrix.nnz == 3


def test_joint_storage_retains_the_original_small_differentiated_constraint():
    coupling = 1e-5
    jacobian = sp.csr_matrix([[0.0, -coupling, 0.0], [0.0, 1.0, 0.0], [-coupling, 0.0, 1.0]])
    mass = sp.csr_matrix([[2.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    algebraic = sp.csr_matrix([[0.0, 0.0, 1.0]])
    correction, unique = stored_initial_correction(
        jacobian,
        np.array([0.0, -1.0, 0.0]),
        mass,
        algebraic,
        np.array([0, 1]),
        np.array([0, 2]),
        np.zeros(2),
        np.zeros(3),
        tolerance=1e-10,
        state=np.zeros(3),
        rate_tangent=jacobian,
    )
    np.testing.assert_array_equal(correction[[0, 2]], 0.0)
    expected_current = 1 / (1 + coupling**2 / 2)
    np.testing.assert_allclose(correction[1], expected_current, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(correction[5], coupling**2 * expected_current / 2, rtol=1e-10, atol=0.0)
    assert unique[[0, 2]].all()
