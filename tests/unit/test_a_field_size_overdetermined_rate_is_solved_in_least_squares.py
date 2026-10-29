"""A consistent, full-column-rank, overdetermined field-size rate system is solved, not refused.

The case_a dual-star PWM accepted projection (31650 rows x 31644 rate columns, six dependent
net-current mass rows) was MEASURED consistent (out-of-range residual 4.8e-10 of 5.8e6) and of full
column rank, yet the square matching picked a dependent row set and refused it
(a downstream memory entry).
Planted here at > 512 unknowns: the matching's priority prefers the two leading, mutually
dependent rows, so the square subsystem is singular while the whole system has one exact answer.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.initial_rate import sparse_initial_rate

_N = 600


def _system(*, contradict=False):
    rows = [[1.0, 1.0], [2.0, 2.0], [1.0, -1.0]]
    head = sp.lil_matrix((3, _N))
    for row, (a, b) in enumerate(rows):
        head[row, 0], head[row, 1] = a, b
    tail = sp.hstack((sp.csr_matrix((_N - 2, 2)), sp.diags(np.arange(1.0, _N - 1))))
    matrix = sp.vstack((head.tocsr(), tail), format='csr')
    expected = np.concatenate(([0.75, 0.25], np.linspace(-1.0, 1.0, _N - 2)))
    rhs = matrix @ expected
    if contradict:
        rhs[1] += 1.0
    return matrix, rhs, expected


def test_the_dependent_square_choice_no_longer_refuses_a_consistent_system() -> None:
    matrix, rhs, expected = _system()
    rate, unique = sparse_initial_rate(matrix, rhs, tolerance=1e-10)
    np.testing.assert_allclose(rate, expected, rtol=0, atol=1e-12)
    assert unique.all()


def test_a_contradicting_field_size_system_still_refuses() -> None:
    matrix, rhs, _ = _system(contradict=True)
    with pytest.raises(ValueError, match='contradict'):
        sparse_initial_rate(matrix, rhs, tolerance=1e-10)
