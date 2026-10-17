"""A solid body's voltage is index two: a frozen-state rate solve refuses it, the hidden constraint fixes it.

Planted DAE, two field coordinates ``a`` and one isolated body voltage ``u``
(``W = M 1``, ``G = 1^T M 1``):

    m1 a1' + k1 a1 - m1 u = f1
    m2 a2' + k2 a2 - m2 u = f2
    G u - (m1 a1' + m2 a2') = 0        (zero net current)

``Y = (1, 1, 1)`` annihilates the mass AND the ``u`` column, so only the
differentiated hidden row ``k1 a1' + k2 a2' = 0`` fixes ``u``. At the
consistent ``a = (1/4, 1/4)`` the exact answer is binary:
``u = -1/64``, ``a' = (15/64, -9/64)``. A discrete ``u = 0`` (what a step
leaves behind) contradicts that hidden row under a frozen state.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.index_two import index_two_directions, index_two_frozen_rate
from numerics_lab.linear.initial_rate import sparse_initial_rate

pytestmark = pytest.mark.unit

_M = sp.csr_matrix([[1.0, 0, 0], [0, 2.0, 0], [-1.0, -2.0, 0]])
_J = sp.csr_matrix([[3.0, 0, -1.0], [0, 5.0, -2.0], [0, 0, 3.0]])
_Y = sp.csr_matrix([[1.0, 1.0, 1.0]])
_F = np.array([1.0, 1.0, 0.0])


def _system(u):
    state = np.array([0.25, 0.25, u])
    hidden = (_Y @ _J).tocsr()
    return hidden, -(_J @ state - _F)


def test_the_body_voltage_is_the_one_index_two_column():
    columns, basis = index_two_directions(_M, _J, _Y)
    assert columns.tolist() == [2]
    assert basis.tolist() == [[1.0]]


def test_the_frozen_discrete_voltage_is_refused():
    hidden, forcing = _system(0.0)
    with pytest.raises(ValueError, match='contradict'):
        sparse_initial_rate(sp.vstack((hidden, _M)), np.concatenate(([0.0], forcing)))


def test_the_voltage_is_re_solved_with_the_rates_and_every_row_holds():
    hidden, forcing = _system(0.0)
    columns, step, rate, unique = index_two_frozen_rate(
        _Y, hidden, _M, _J, np.zeros(1), forcing, tolerance=1e-10, allowance=np.zeros(3)
    )
    assert columns.tolist() == [2]
    assert step.tolist() == [-1 / 64]
    np.testing.assert_array_equal(rate[:2], [15 / 64, -9 / 64])
    assert not unique[2], 'the voltage rate is free, never claimed'
    state = np.array([0.25, 0.25, step[0]])
    np.testing.assert_array_equal(_J @ state + _M @ rate - _F, 0.0)
    assert (hidden @ rate)[0] == 0.0


def test_a_consistent_voltage_is_the_plain_frozen_solve():
    hidden, forcing = _system(-1 / 64)
    columns, step, rate, _unique = index_two_frozen_rate(
        _Y, hidden, _M, _J, np.zeros(1), forcing, tolerance=1e-10, allowance=np.zeros(3)
    )
    assert len(columns) == 0 and len(step) == 0
    np.testing.assert_array_equal(rate[:2], [15 / 64, -9 / 64])
