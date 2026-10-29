"""A retained row that is exactly a sum of eliminated rows inherits their evaluation floors.

case_a voltage_driven at the 0.75 mm floor (a downstream repo memory 2026-10-25,
`every-native-binding-reads-its-magnet-directors-from-one-source.md`): a solid bar's net-current row
is, after elimination, exactly the sum of its ~100 nodes' field rows -- its own pivot (G = 202.452)
cancels to 0.0. Each node's seed residual sits at 0.06-0.09 of its own evaluation floor
``gamma_k |J||u|``, but a differential row's allowance is zero, so the 1e-11 sum reached the bar row
with a 1e-12 bar and the admission refused at t0. The bar inherits the nodes' floors through the
elimination's adjoint; a residual ABOVE a node's floor, or an offset on the bar row itself, still
exceeds that inheritance and is still refused.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.initial_storage_border import (
    _cancelled_pivot_rows,
    _driven_allowance_image,
    _evaluation_floor,
)

pytestmark = pytest.mark.unit

_MASS = np.array([1.0, 2.0, 1.5, 0.5])
_SCALE = np.array([3e4, 4e4, 3.5e4, 2e4])  # |J||u| of each node row


def _bar():
    """Four bar-node field rows (eliminated, block = their masses) and the bar row (left = the masses)."""
    block = sp.diags(_MASS, format='csr')
    left = sp.csr_matrix(_MASS[None, :])
    rows = sp.csr_matrix(np.hstack((np.diag(_MASS), np.ones((4, 9)))))  # 10 terms per node row
    floor = _evaluation_floor(rows, np.arange(4), _SCALE)
    return block, left, floor


def _bar_forcing(residuals, offset=0.0):
    """The bar row's eliminated forcing: its own offset minus ``left B^-1`` of the node residuals."""
    return offset - float(np.sum(residuals))


def test_the_floor_is_gamma_k_times_the_row_operands():
    _block, _left, floor = _bar()
    gamma = 11 * np.finfo(float).eps / (1 - 11 * np.finfo(float).eps)
    np.testing.assert_allclose(floor, gamma * _SCALE, rtol=0)


def test_a_bar_whose_nodes_sit_at_their_floors_is_admitted():
    block, left, floor = _bar()
    residuals = 0.09 * floor * np.array([1, -1, 1, 1])  # each node at 9 % of its own floor
    image = _driven_allowance_image(block, left, floor, np.array([True]))
    assert abs(_bar_forcing(residuals)) <= image[0]


def test_a_node_above_its_own_floor_is_still_refused():
    """PLANTED non-convergence: one node 50x its floor; the bar inherits floors, not that residual."""
    block, left, floor = _bar()
    residuals = 0.09 * floor
    residuals[2] = 50 * floor[2]
    image = _driven_allowance_image(block, left, floor, np.array([True]))
    assert abs(_bar_forcing(residuals)) > image[0]


def test_a_bar_level_offset_above_the_inherited_floor_is_still_refused():
    """PLANTED: every node under its floor, but the bar row carries its own contradiction."""
    block, left, floor = _bar()
    residuals = 0.09 * floor * np.array([1, -1, 1, -1])
    image = _driven_allowance_image(block, left, floor, np.array([True]))
    assert abs(_bar_forcing(residuals, offset=2 * image[0])) > image[0]


def test_only_a_row_whose_direct_pivot_cancels_is_a_summed_row():
    direct = sp.csr_matrix(np.array([[202.452, -1.0], [202.452, -1.0]]))
    schur = sp.csr_matrix(np.array([[0.0, -1.0], [101.2, -1.0]]))  # row 0 cancelled to 0.0
    errors = np.full((2, 2), 4.4e-12)
    np.testing.assert_array_equal(_cancelled_pivot_rows(direct, schur, errors), [True, False])
    # A noise remainder below its own bound is a cancellation too (the 63 bars that passed on 1e-13).
    noisy = sp.csr_matrix(np.array([[-8.5e-14, -1.0], [101.2, -1.0]]))
    np.testing.assert_array_equal(_cancelled_pivot_rows(direct, noisy, errors), [True, False])
