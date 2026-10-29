"""Completion retains an admitted Ft's performed arithmetic, not a new physical floor."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.formulation import ResidualAssemblyBound
from numerics_lab.linear.arithmetic import matrix_vector_roundoff
from numerics_lab.linear.initial_sparse import SparseInitialPair, sparse_complete_initial_rate


def _performed_partial():
    operands = sp.csr_matrix([[0.0, 0.0], [0.1, -1.0]])
    values = np.array([0.2, 0.02])
    partial = operands @ values
    bound = matrix_vector_roundoff(operands, values, np.zeros(2))
    assert partial[1] != 0.0
    assert abs(partial[1]) <= bound.values[1]
    return partial, bound


def _complete(*, bound=None, residual=None, second=None, curvature=None):
    partial, _actual_bound = _performed_partial()
    pair = SparseInitialPair(np.zeros(2), np.zeros(2), np.array([True, False]))
    options = {} if bound is None else {'partial_time_bound': bound}
    return sparse_complete_initial_rate(
        sp.csr_matrix([[0.0, -1.0], [1.0, 0.0]]),
        np.zeros(2) if residual is None else residual,
        sp.diags([1.0, 0.0], format='csr'),
        pair,
        algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
        partial_time=partial,
        acceleration_remainder=np.array([0.0, -3.0]) if curvature is None else curvature,
        acceleration_jacobian=second,
        stored_rows=np.array([0]),
        stored_columns=np.array([0]),
        **options,
    )


def test_without_actual_partial_authority_completion_retains_its_original_refusal():
    with pytest.raises(ValueError, match='curvature|forcing|contradict'):
        _complete()


def test_actual_performed_partial_bound_survives_both_differentiated_equations():
    _partial, bound = _performed_partial()
    completed = _complete(bound=bound)
    np.testing.assert_array_equal(completed.state, np.zeros(2))
    np.testing.assert_array_equal(completed.rate, [0.0, 3.0])
    assert completed.rate_is_unique.all()


def test_unknown_partial_arithmetic_cannot_be_promoted_to_a_zero_equation():
    _partial, bound = _performed_partial()
    unknown = ResidualAssemblyBound(bound.values, np.array([True, False]))
    with pytest.raises(ValueError, match='curvature|forcing|contradict'):
        _complete(bound=unknown)


def test_partial_bound_cannot_be_borrowed_from_a_different_physical_row():
    _partial, bound = _performed_partial()
    other_row = ResidualAssemblyBound(bound.values[::-1], np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match='curvature|forcing|contradict'):
        _complete(bound=other_row)


def test_partial_certificate_never_relaxes_original_f_rows():
    partial, bound = _performed_partial()
    with pytest.raises(ValueError, match='curvature|forcing|contradict'):
        _complete(bound=bound, residual=np.array([0.0, partial[1]]))


def test_partial_certificate_never_relaxes_independent_curvature_rows():
    partial, bound = _performed_partial()
    with pytest.raises(ValueError, match='curvature|forcing|contradict'):
        _complete(bound=bound, second=sp.csr_matrix((2, 2)), curvature=np.array([0.0, partial[1] / 2]))


def test_partial_certificate_requires_the_actual_equation_dimensions():
    with pytest.raises(ValueError, match='partial'):
        _complete(bound=ResidualAssemblyBound(np.zeros(3), np.ones(3, dtype=bool)))
