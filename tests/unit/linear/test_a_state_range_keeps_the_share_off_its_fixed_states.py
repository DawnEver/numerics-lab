"""With a state range, the adjoint share never moves a state the range holds fixed.

JMAG cond03 mesh, closed-loop Laplace x FEA re-admission at t = 156 us (Ada 8002602; operands
captured by 8008211): a winding row over its bar was shared onto the ELIMINATED rows, and the share
took up to 1.3e-10 on the air field rows and 1.5e-12 on the stored-charge rows. Those rows solve for
STATE coordinates, so 46110 field states moved by up to 1.7e-10. The supplied state range holds
them fixed, and the correction resets them to zero; the eliminated rows then lost the shift they
had absorbed. Field rows were left up to 250x their bars ("contradicts actual original rows after
exact homogeneous anchors"). Before that reset, every original row held its bar.

Under a state range the share moves RATES only: the rows that solve for states (air state rows and
stored-charge rows) take a zero bar, which `_eliminated_share` reads as exact.

PLANTED: the two-bodies case of `test_a_shared_body_row_is_judged_at_its_trial` (every field
coordinate stored: its state row solves for its RATE, its stored-charge row for its STATE). Under
a range holding the field states, the body-row plant is shared through the rate rows only, the
field states stay exactly zero and every original row holds its bar. The stored-charge rows carry
the caller's row scale, so on the parent (2ea8e4c) the share saw them at bar 1.0. The behavioural
red at full scale is the captured cond03 replay. CONTROL: with no range, the same share still uses
the stored-charge rows.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from tests.unit.linear.test_a_shared_body_row_is_judged_at_its_trial import (
    TOLERANCE,
    _two_bodies_sharing_a_field_row,
)

from numerics_lab.linear import initial_storage_border
from numerics_lab.linear.initial_storage_border import _sparse_joint_border, joint_coordinate_sparse_correction

pytestmark = pytest.mark.unit

STORED = slice(5, 10)  # the block's stored-charge rows: [state rows; stored rows; differentiated rows]


def _spy_on_the_share(monkeypatch):
    seen = []
    share = initial_storage_border._eliminated_share

    def spy(block, left, bars, residual):
        seen.append(bars.copy())
        return share(block, left, bars, residual)

    monkeypatch.setattr(initial_storage_border, '_eliminated_share', spy)
    return seen


def _with_state_range(delta):
    matrix, mass, algebraic, stored_rows, _rhs, m = _two_bodies_sharing_a_field_row(0.0)
    n = mass.shape[0]
    # An exact answer whose field states do not move (the range holds them): bodies 1, phase 2,
    # field rates 1..m. The planted inconsistency sits on body 1's row alone.
    exact = np.r_[np.zeros(m), 1.0, 1.0, 2.0, np.arange(1.0, m + 1), np.zeros(n - m)]
    rhs = matrix @ exact
    rhs[m] += delta
    directions = sp.csr_matrix((np.ones(n - m), (np.arange(m, n), np.arange(n - m))), shape=(n, n - m))
    return joint_coordinate_sparse_correction(
        matrix,
        rhs,
        mass=mass,
        algebraic_rows=algebraic,
        stored_rows=stored_rows,
        indices=np.arange(m),
        state_directions=directions,
        row_scale=_state_operand_scale(matrix, n, m),
        tolerance=TOLERANCE,
    )


def _state_operand_scale(matrix, n, m):
    # As the caller supplies it: each state/stored row's operands at the CURRENT state (field 1,
    # bodies 1, phase 2), so a stored-charge row has a bar of its own; the voltage rows have none.
    state = np.r_[np.ones(m), 1.0, 1.0, 2.0]
    return np.r_[abs(matrix[: n + m, :n]) @ state, np.zeros(matrix.shape[0] - n - m)]


def test_under_a_state_range_the_share_never_uses_a_row_that_solves_for_a_state(monkeypatch):
    seen = _spy_on_the_share(monkeypatch)
    correction, _unique = _with_state_range(1e-4)
    np.testing.assert_array_equal(correction[:5], 0.0)
    assert seen, 'the planted body residual reaches the share'
    for bars in seen:
        np.testing.assert_array_equal(bars[STORED], 0.0)
        assert np.all(bars[:5] > 0), 'the rate rows still share'


def test_without_a_state_range_the_share_still_moves_stored_states(monkeypatch):
    seen = _spy_on_the_share(monkeypatch)
    matrix, mass, algebraic, stored_rows, rhs, m = _two_bodies_sharing_a_field_row(1e-4)
    _sparse_joint_border(
        matrix, rhs, mass, algebraic, stored_rows, np.arange(m), row_scale=np.zeros(len(rhs)), tolerance=TOLERANCE
    )
    assert seen
    assert all(np.all(bars[STORED] > 0) for bars in seen)
