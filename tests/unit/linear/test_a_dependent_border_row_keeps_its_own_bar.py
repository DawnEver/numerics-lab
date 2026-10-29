"""A dependent border row set places its inconsistency on the rows whose bars can hold it.

case_a dual-star PWM admission 80 stalled 30 iterations on a solid body's net-current row
(a downstream memory entry). In the
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

from numerics_lab.linear.initial_storage_border import _sparse_joint_border, _weighted_by_row_bars

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
    allowance[-1] = 1e-8  # the elimination admits it, as case_a's driven image did
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

    case_a admission 163: the eliminated field rows' remainders, each inside its own allowance,
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


def test_an_answer_inside_every_bar_is_not_moved_so_an_exact_zero_stays_zero():
    """An answer that already meets every row's bar is returned as it is.

    MEASURED 2026-10-13 on the switched-network checkpoint: the border answered the at-rest boundary
    exactly (field coordinates 0), every row inside its bar, and the unconditional weighted least
    squares still moved it -- 1.4e-58 on the winding coordinates, 1e-63 on every field rate -- noise the
    boundary Newton then judged against itself and amplified until it diverged.
    """
    schur = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    forcing = np.array([2e-12, 0.0, 0.0])
    bars = np.array([2e-4, 2e-4, 0.0])  # rows 0, 1 soft; row 2 exact
    answer = np.zeros(3)
    moved = _weighted_by_row_bars(schur, forcing, answer, bars)
    assert np.array_equal(moved, answer), 'a met answer is exact authority, not a least-squares seed'


def test_a_joint_correction_that_already_holds_is_not_rate_re_solved(monkeypatch):
    """Every original row at the bare bar: the rate re-solve is skipped, not re-run.

    case_a admission 338: the border met body 5's net-current row exactly, and the frozen-state
    rate re-solve put it back at 2.1x its bar (the dependent row's rounding image), every iteration.
    """
    from numerics_lab.linear import initial_storage_border as owner  # noqa: PLC0415

    def planted_re_solve(*_args, **_kwargs):
        msg = 'PLANTED: the rate re-solve must not run'
        raise AssertionError(msg)

    monkeypatch.setattr(owner, 'index_two_frozen_rate', planted_re_solve)
    count = 3
    n = count + 2
    jacobian = sp.lil_matrix((n, n))
    jacobian[:count, :count] = sp.eye(count)
    jacobian[0, count] = -1
    jacobian[count, count] = 5
    jacobian[count + 1, 0] = 1
    jacobian[count + 1, count + 1] = -1
    mass = sp.lil_matrix((n, n))
    mass[0, 0] = 2
    mass[count, count + 1] = 1
    jacobian, mass = jacobian.tocsr(), mass.tocsr()
    algebraic = sp.eye(n, format='csr')[np.r_[np.arange(1, count), count + 1]]
    residual = np.zeros(n)
    residual[count] = -7
    correction, _unique = owner.stored_initial_correction(
        jacobian,
        residual,
        mass,
        algebraic,
        np.array([0, count]),
        np.array([0, count + 1]),
        np.zeros(2),
        np.zeros(n),
        tolerance=1e-10,
        state=np.zeros(n),
        rate_tangent=jacobian,
        sparse_state_indices=np.arange(count),
    )
    assert correction[count] == pytest.approx(7 / 5.5, abs=1e-14)


def test_a_weak_retained_row_takes_its_residual_from_the_eliminated_rows_by_their_bars():
    """The adjoint share: ``y . shift = residual`` at least weighted size; a zero-bar row takes nothing.

    case_a admission 338: body 5's Schur row is 2e-5 of its neighbours, so no retained row can
    hold its 2.6x-bar residual; the eliminated rows (bars far larger) can.
    """
    from numerics_lab.linear.initial_storage_border import _eliminated_share  # noqa: PLC0415

    block = sp.csr_matrix(np.array([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]]))
    left = sp.csr_matrix(np.array([[1.0, 1.0, 1.0]]))
    bars = np.array([1.0, 3.0, 0.0])  # row 2 exact
    shift = _eliminated_share(block, left, bars, np.array([5e-9]))
    adjoint = np.array([0.5, 0.25, 1.0])  # B^T y = left^T
    assert adjoint @ shift == pytest.approx(5e-9, rel=1e-12)
    assert shift[2] == 0.0
    assert shift[1] / shift[0] == pytest.approx((9 * 0.25) / (1 * 0.5), rel=1e-12), 'weighted by bar squared'


def test_the_exact_row_null_space_is_never_formed_as_a_square_basis():
    """Memory is O(rows x live), never O(live^2): a wide border with an exact row stays small.

    The transformer march (2026-10-14, Ada OOM at 62.6 GB) reached this with a live set spanning the
    field: ``svd(full_matrices=True)`` and ``np.eye(live)`` each formed a live x live dense square.
    PLANTED: 3 rows x 6000 live columns (a square would be 288 MB); the bound is 64x the input.
    """
    import tracemalloc  # noqa: PLC0415

    rng = np.random.default_rng(0)
    schur = rng.standard_normal((3, 6000))
    forcing = rng.standard_normal(3)
    bars = np.array([0.0, 1.0, 1.0])  # row 0 exact, rows 1-2 soft
    tracemalloc.start()
    try:
        moved = _weighted_by_row_bars(schur, forcing, np.zeros(schur.shape[1]), bars)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 64 * schur.nbytes, f'peak {peak / 2**20:.0f} MiB: a live x live square was formed'
    assert schur[0] @ moved == pytest.approx(forcing[0], rel=1e-12), 'the exact row holds'
    assert np.allclose(schur[1:] @ moved, forcing[1:], rtol=1e-10), 'an underdetermined soft set is met'
