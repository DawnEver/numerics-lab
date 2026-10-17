"""``gauss.linear.apply_dirichlet_triplets`` -- Dirichlet pinning as pure COO algebra.

The tests move WITH the code (V2 step 2b): this function used to be
``hamilton.discretizations.fem._solve._apply_dirichlet`` and was only ever exercised through FEA
and time-harmonic solves. Pinning is not an FEA operation, so it is now a ``gauss`` citizen and
gets its own tests at that level -- including the DTYPE property the harmonic path silently
depended on and nothing checked directly.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

from numerics_lab.linear import apply_dirichlet_triplets


def _dense(row, col, val, n):
    return coo_matrix((val, (row, col)), shape=(n, n)).toarray()


def test_pinned_row_becomes_the_identity_row_and_rhs_takes_the_value():
    row = np.array([0, 0, 1, 1, 2])
    col = np.array([0, 1, 0, 1, 2])
    val = np.array([2.0, -1.0, -1.0, 2.0, 3.0])
    rhs = np.zeros(3)

    r, c, v = apply_dirichlet_triplets(row, col, val, rhs, {5.0: [1]}, 0.0)

    dense = _dense(r, c, v, 3)
    assert np.array_equal(dense[1], [0.0, 1.0, 0.0]), 'pinned row is not the identity row'
    assert rhs[1] == 5.0
    # Unpinned rows are untouched -- the COLUMN of a pinned index is deliberately NOT eliminated
    # (this is row-pinning, not static condensation), so row 0 keeps its -1 coupling.
    assert np.array_equal(dense[0], [2.0, -1.0, 0.0])


def test_bias_is_added_to_every_prescribed_value():
    rhs = np.zeros(3)
    apply_dirichlet_triplets(np.array([0]), np.array([0]), np.array([1.0]), rhs, {2.0: [0, 2]}, 0.5)
    assert rhs[0] == 2.5
    assert rhs[2] == 2.5


def test_a_complex_rhs_keeps_its_phase():
    """The time-harmonic path pins a complex phasor; a real-only implementation would truncate it.

    This is the property ``studies/harmonic`` relies on and the reason the function had to be
    dtype-agnostic. It was never asserted anywhere before this file.
    """
    rhs = np.zeros(2, dtype=complex)
    row, col, val = (np.array([0]), np.array([0]), np.array([1.0 + 0.0j]))
    _r, _c, v = apply_dirichlet_triplets(row, col, val, rhs, {1.0 + 2.0j: [1]}, 0.0)
    assert rhs[1] == 1.0 + 2.0j
    assert np.iscomplexobj(v), 'the appended unit diagonal demoted a complex value array to real'


def test_an_empty_index_group_is_a_no_op():
    row, col, val = (np.array([0]), np.array([0]), np.array([1.0]))
    rhs = np.array([7.0])
    r, c, v = apply_dirichlet_triplets(row, col, val, rhs, {0.0: []}, 0.0)
    assert rhs[0] == 7.0
    assert (len(r), len(c), len(v)) == (1, 1, 1)


def test_duplicate_indices_pin_once():
    """``np.unique`` on the group: a repeated index must not append two diagonal entries.

    Two unit entries on the same (i, i) SUM to 2.0 in COO assembly, which would scale the pinned
    row by two and halve the recovered value -- a wrong answer, not an error.
    """
    row, col, val = (np.array([0]), np.array([0]), np.array([9.0]))
    rhs = np.zeros(1)
    r, c, v = apply_dirichlet_triplets(row, col, val, rhs, {3.0: [0, 0, 0]}, 0.0)
    assert _dense(r, c, v, 1)[0, 0] == 1.0
