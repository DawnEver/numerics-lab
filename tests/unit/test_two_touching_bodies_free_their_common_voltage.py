"""Two conducting bodies that share a field node free their COMMON voltage, not each voltage.

case_n (t0 admission, `case/04-49-30`): of its 9 solid bodies, two pairs share field nodes
(8 and 1 nodes). Each voltage column of a pair is seen by the same two certified hidden rows with
equal and opposite weight (+-4.25, +-0.35), so neither column lies in the range of the mass, and
the column test froze both voltages; the frozen-state rate solve then refused. Their SUM lies in
the range of the mass -- a uniform field over both bodies -- so that one direction is index two,
while the difference is fixed by the hidden rows as a rate.

Planted DAE: body 1 on nodes (a1, a2), body 2 on (a2, a3), lumped mass 1 and 2 per node:

    a1' + 3 a1 - u1 = 1
    3 a2' + 5 a2 - u1 - 2 u2 = 1
    2 a3' + 7 a3 - 2 u2 = 1
    2 u1 - (a1' + a2') = 0
    4 u2 - (2 a2' + 2 a3') = 0
"""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.arithmetic import certified_sparse_product, sparse_product_bound
from numerics_lab.linear.index_two import certified_null_directions, index_two_directions, index_two_frozen_rate

pytestmark = pytest.mark.unit

_M = sp.csr_matrix([
    [1.0, 0, 0, 0, 0],
    [0, 3.0, 0, 0, 0],
    [0, 0, 2.0, 0, 0],
    [-1.0, -1.0, 0, 0, 0],
    [0, -2.0, -2.0, 0, 0],
])
_J = sp.csr_matrix([
    [3.0, 0, 0, -1.0, 0],
    [0, 5.0, 0, -1.0, -2.0],
    [0, 0, 7.0, 0, -2.0],
    [0, 0, 0, 2.0, 0],
    [0, 0, 0, 0, 4.0],
])
_Y = sp.csr_matrix([[3.0, 1.0, 0, 3.0, 0], [0, 2.0, 3.0, 0, 3.0]])  # left null of the mass
_F = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
# Meets the undifferentiated constraints Y (J x - F) = 0, which fix u1 - u2 = 9/16 and a3; the
# common voltage u1 + u2 is left at a discrete value the hidden rows contradict.
_STATE = np.array([0.25, 0.125, 13 / 56, 9 / 16, 0.0])


def test_neither_voltage_column_is_index_two_but_their_sum_is():
    assert abs(_Y @ _M).max() == 0.0
    assert (certified_sparse_product(_Y, _J[:, [3, 4]]) != 0).toarray().all()  # each column is seen
    columns, basis = index_two_directions(_M, _J, _Y)
    assert columns.tolist() == [3, 4]
    assert basis.shape == (2, 1)
    np.testing.assert_allclose(basis[:, 0] / basis[0, 0], [1.0, 1.0], rtol=1e-15)


@pytest.mark.parametrize('copies', [1, 260])
def test_the_common_voltage_is_re_solved_and_every_row_holds(copies):
    mass = sp.block_diag([_M] * copies, format='csr')
    jacobian = sp.block_diag([_J] * copies, format='csr')
    algebraic = sp.block_diag([_Y] * copies, format='csr')
    initial, source = np.tile(_STATE, copies), np.tile(_F, copies)
    hidden = certified_sparse_product(algebraic, jacobian)
    np.testing.assert_allclose(algebraic @ (jacobian @ initial - source), 0.0, atol=1e-15)
    forcing = -(jacobian @ initial - source)
    columns, step, rate, unique = index_two_frozen_rate(
        algebraic,
        hidden,
        mass,
        jacobian,
        np.zeros(2 * copies),
        forcing,
        tolerance=1e-12,
        allowance=np.zeros(5 * copies),
    )
    np.testing.assert_array_equal(columns, np.concatenate([np.array([3, 4]) + 5 * i for i in range(copies)]))
    np.testing.assert_allclose(step[::2], step[1::2], rtol=1e-14)
    state = initial.copy()
    state[columns] += step
    np.testing.assert_allclose(jacobian @ state + mass @ rate - source, 0.0, atol=1e-14)
    np.testing.assert_allclose(hidden @ rate, 0.0, atol=1e-14)
    assert unique.reshape(copies, 5)[:, :3].all()


def test_a_pair_that_cancels_only_to_rounding_is_one_null_direction():
    # The boundary-state search (case_o) read its freedom as EXACT rational dependencies, so
    # a body column that cancels to residue against the left-null rows, and a touching pair that
    # cancels only as a pair, were both missed. Here Y J[:, 0] = 0.1 + 0.2 - 0.3 = 5.6e-17 and the
    # pair (1, 2) is seen as (0.1 + 0.2, -0.3) and (1, -1): only its sum is null, and only to rounding.
    left = sp.csr_matrix([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    right = sp.csr_matrix([[0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [-0.3, 0.0, -0.3], [0.0, 1.0, -1.0]])
    product = (left @ right).toarray()
    assert product[0, 0] != 0.0 and product[0, 1] + product[0, 2] != 0.0  # residue, not exact zeros
    basis = certified_null_directions(left, right)
    assert basis.shape == (3, 2)
    np.testing.assert_allclose(abs(left @ right @ basis), 0.0, atol=1e-15)


def test_a_null_direction_keeps_its_small_component_at_the_original_row_certificate():
    # Normwise SVD rounds the 1e-32 component away; the second actual row still
    # requires it. Candidate accuracy is not physical state-rank authority.
    left = sp.eye(2, format='csr')
    right = sp.csr_matrix([[1.0, 1e16, 0.0], [0.0, 1e-16, 1.0]])
    basis = certified_null_directions(left, right)
    assert basis.shape == (3, 1)
    np.testing.assert_allclose(basis[:, 0] / basis[0, 0], [1.0, -1e-16, 1e-32], rtol=1e-15, atol=0)
    product, bound = sparse_product_bound(left, right)
    assert np.all(abs(product @ basis) <= bound @ abs(basis))
