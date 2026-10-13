"""A dependent border row set places its inconsistency on the rows whose bars can hold it.

Cummins dual_star_pwm admission 80 stalled 30 iterations on a solid body's net-current row
(.claude/memory/2026/10/12/cummins-border-a-dependent-row-set-is-weighted-by-its-bars.md). In the
border the six body rows are DEPENDENT on the flux-definition and stored-flux rows: the integrator's
stored target satisfies the body's algebraic net-current constraint only to its rounding (about 2e-15
of that constraint's operands), and the elimination left that whole inconsistency on the body row,
whose own operands are 3e5 times smaller, every iteration. No refinement moves it: it is a property of
the data, not of the solve. A weighted least squares over the border rows, each weighted by its own
original bar, keeps every row under the unchanged 1e-10 bar.

PLANTED here: a retained row duplicated at 1e-6 of its size, inconsistent by 1.4e-10 of its own
operands. Left on the small row alone it is over that row's bar; shared by both rows in proportion to
their bars, each keeps 0.7e-10.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.gauss.linear.initial_storage_border import _sparse_joint_border, _weighted_by_row_bars

pytestmark = pytest.mark.unit


def _planted(count=3):
    n = count + 2
    jacobian = sp.lil_matrix((n, n))
    jacobian[:count, :count] = sp.eye(count)
    jacobian[0, count] = -1
    jacobian[count, count] = 5e3
    jacobian[count + 1, 0] = 1
    jacobian[count + 1, count + 1] = -1
    mass = sp.lil_matrix((n, n))
    mass[0, 0] = 2
    mass[count, count + 1] = 1
    jacobian, mass = jacobian.tocsr(), mass.tocsr()
    algebraic = sp.eye(n, format='csr')[np.r_[np.arange(1, count), count + 1]]
    stored_rows = np.array([0, count])
    residual = np.zeros(n)
    residual[count] = -7e3
    stored = mass[stored_rows]
    matrix = sp.bmat(
        [
            [jacobian, mass],
            [stored, sp.csr_matrix(stored.shape)],
            [sp.csr_matrix(algebraic.shape), algebraic @ jacobian],
        ],
        format='lil',
    )
    rhs = np.concatenate((-residual, np.zeros(len(stored_rows)), np.zeros(algebraic.shape[0])))
    # PLANTED dependent row: the retained row ``count`` at 1e-6 of its size, inconsistent by 2e-12.
    small = matrix[count].toarray().ravel() * 1e-6
    matrix = sp.vstack((matrix.tocsr(), sp.csr_matrix(small)), format='csr')
    rhs = np.append(rhs, rhs[count] * 1e-6 + 2e-12)
    allowance = np.zeros(len(rhs))
    allowance[-1] = 1e-8  # the elimination admits it, as Cummins's driven image did
    row_scale = np.zeros(len(rhs))
    return n, count, jacobian, mass, algebraic, stored_rows, matrix, rhs, allowance, row_scale


def _bars(matrix, rhs, correction):
    return 1e-10 * (abs(matrix) @ abs(correction) + abs(rhs))


def test_without_its_bar_the_inconsistency_lands_on_the_small_row():
    _n, count, _j, mass, algebraic, stored_rows, matrix, rhs, allowance, _scale = _planted()
    correction, _directions, _admitted = _sparse_joint_border(
        matrix, rhs, mass, algebraic, stored_rows, np.arange(count), residual_allowance=allowance
    )
    defect = abs(matrix @ correction - rhs)
    assert defect[-1] > _bars(matrix, rhs, correction)[-1], 'the planted stall: the small row is over its bar'


def test_each_dependent_row_keeps_its_own_bar():
    _n, count, _j, mass, algebraic, stored_rows, matrix, rhs, allowance, row_scale = _planted()
    correction, _directions, _admitted = _sparse_joint_border(
        matrix, rhs, mass, algebraic, stored_rows, np.arange(count), residual_allowance=allowance, row_scale=row_scale
    )
    defect = abs(matrix @ correction - rhs)
    assert np.all(defect <= _bars(matrix, rhs, correction))


def test_an_eliminated_rows_slack_stays_on_it_and_is_not_exported():
    """A field row already inside its own certified bound is not re-solved through elimination.

    Cummins admission 163: the eliminated field rows' remainders, each inside its own allowance,
    were solved and so summed onto a body net-current row (1.3x its bar).
    """
    _n, count, _j, mass, algebraic, stored_rows, matrix, rhs, allowance, row_scale = _planted()
    rhs[0], allowance[0] = 1e-12, 1e-11  # PLANTED slack on an eliminated row coupled to the border
    correction, _directions, _admitted = _sparse_joint_border(
        matrix, rhs, mass, algebraic, stored_rows, np.arange(count), residual_allowance=allowance, row_scale=row_scale
    )
    defect = abs(matrix @ correction - rhs)
    assert defect[0] <= allowance[0], 'the slack stays on its own row, inside its own bound'
    assert np.all(defect[1:] <= _bars(matrix, rhs, correction)[1:])


@pytest.mark.filterwarnings('error')
def test_a_zero_bar_row_is_an_exact_constraint_never_a_division():
    """A homogeneous witness has bar 0 (or a subnormal one); it is met exactly, not weighted by 1/0.

    PLANTED after the switched-network checkpoint regression: dividing by a zero bar overflowed and
    the weighted least squares raised ``SVD did not converge``.
    """
    schur = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    forcing = np.array([0.0, 2.0, 2.5, 0.0])
    bars = np.array([0.0, 1e-320, 1.0, 0.0])  # rows 0 and 1 exact, row 2 soft, row 3 empty
    moved = _weighted_by_row_bars(schur, forcing, np.zeros(2), bars)
    assert np.all(np.isfinite(moved))
    assert moved[0] == 0.0 and moved[1] == 2.0, 'the exact rows hold exactly; the soft row keeps the 0.5'
