"""A declared state coordinate is an exact anchor of the joint correction, never an unknown.

A declared initial value (a winding's REST current) is a stated initial condition: the joint
correction fixes its state column at exactly zero change, solves the rest, and every original
row -- the anchored coordinate's own stored-charge row included -- is still checked against its
bar. A declaration the rows contradict is refused, never rounded onto.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.initial_storage_border import joint_coordinate_sparse_correction

MATRIX = sp.csr_matrix(
    np.array([
        [4.0, 0.0, 1.0, 10.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 3.0, 0.0, 0.0, 2.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0, 0.0, 5.0, 0.0],
    ])
)


def _solve(rhs, fixed):
    return joint_coordinate_sparse_correction(
        MATRIX,
        rhs,
        mass=sp.diags([2.0, 0.0, 1.0], format='csr'),
        algebraic_rows=sp.csr_matrix([[0.0, 1.0, 0.0]]),
        stored_rows=np.array([0, 2]),
        indices=np.array([0, 1]),
        fixed_state=fixed,
    )


def test_a_declared_column_changes_by_exactly_zero():
    answer, unique = _solve(MATRIX @ np.array([1.0, 2.0, 0.0, 3.0, 4.0, 5.0]), np.array([2]))
    assert answer[2] == 0.0
    np.testing.assert_array_equal(answer, [1.0, 2.0, 0.0, 3.0, 4.0, 5.0])
    assert unique.all()


def test_a_declaration_the_rows_contradict_is_refused():
    with pytest.raises(ValueError, match='contradicts'):
        _solve(MATRIX @ np.array([1.0, 2.0, 1.0, 3.0, 4.0, 5.0]), np.array([2]))


def test_a_declared_column_must_name_a_state_coordinate():
    with pytest.raises(ValueError, match='declared fixed state'):
        _solve(MATRIX @ np.array([1.0, 2.0, 0.0, 3.0, 4.0, 5.0]), np.array([3]))
