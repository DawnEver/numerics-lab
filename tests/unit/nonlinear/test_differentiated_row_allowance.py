"""Only the independently owned hidden-row allowance reaches rate elimination."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.formulation import ResidualAssemblyBound
from numerics_lab.linear.arithmetic import affine_residual_bound, transport_residual_bound
from numerics_lab.linear.index_two import index_two_frozen_rate
from numerics_lab.linear.initial_sparse import projected_rate_consistent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('certified', [False, True])
def test_original_hidden_row_replay_keeps_its_independent_assembly_allowance(certified):
    defect = 2.0**-52
    jacobian = sp.csr_matrix([[0.0, 0.0], [1.0, 0.0]])
    result = projected_rate_consistent(
        jacobian,
        np.zeros(2),
        np.array([0.0, defect]),
        algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
        tolerance=1e-10,
        differentiated_evaluation=SimpleNamespace(
            residual=np.array([defect]),
            bound=ResidualAssemblyBound(np.array([defect]), np.array([certified])),
        ),
    )
    assert result is certified


def test_original_rate_replay_uses_the_same_performed_direct_value_as_its_arithmetic_owner():
    size = 128
    jacobian = sp.csr_matrix(
        (np.tile([3 * 0.1, -0.1], size), (np.repeat(np.arange(size), 2), np.tile([0, 1], size))),
        shape=(size, size),
    )
    rate = np.zeros(size)
    rate[:2] = [1.0, 3.0]
    partial = np.zeros(size)
    algebraic = sp.csr_matrix(np.ones((1, size)))
    physical = jacobian @ rate + partial
    direct = algebraic @ physical
    bound = affine_residual_bound(
        jacobian, rate, partial, ResidualAssemblyBound(np.zeros(size), np.ones(size, dtype=bool))
    )
    bound = transport_residual_bound(algebraic, physical, bound)
    reassociated = (algebraic @ jacobian) @ rate + algebraic @ partial
    np.testing.assert_array_equal(direct, [0.0])
    assert bound.known.all() and bound.values[0] > 0
    assert abs(reassociated[0]) > bound.values[0], "the alternate arithmetic is not this certificate's owner"
    assert projected_rate_consistent(
        jacobian,
        rate,
        partial,
        algebraic_rows=algebraic,
        tolerance=1e-20,
        differentiated_evaluation=SimpleNamespace(residual=direct, bound=bound),
    )


@pytest.mark.parametrize('defect', [2.0**-52, 1e-3])
def test_frozen_rate_solve_retains_hidden_operand_error_but_not_a_physical_contradiction(defect):
    mass = sp.csr_matrix([[1.0, 0.0], [0.0, 0.0]])
    jacobian = sp.csr_matrix([[0.0, 0.0], [1.0, 0.0]])
    algebraic = sp.csr_matrix([[0.0, 1.0]])
    arguments = (
        algebraic,
        algebraic @ jacobian,
        mass,
        jacobian,
        np.array([defect]),
        np.zeros(2),
    )
    options = {
        'tolerance': 1e-10,
        'allowance': np.zeros(2),
        'differentiated_allowance': np.array([2.0**-52]),
    }
    if defect > 2.0**-52:
        with pytest.raises(ValueError, match='contradict|inconsistent|authority|state'):
            index_two_frozen_rate(*arguments, **options)
    else:
        columns, state_step, rate, unique = index_two_frozen_rate(*arguments, **options)
        assert len(columns) == len(state_step) == 0
        assert abs(rate[0]) <= 2.0**-52
        assert unique[0] and not unique[1]
