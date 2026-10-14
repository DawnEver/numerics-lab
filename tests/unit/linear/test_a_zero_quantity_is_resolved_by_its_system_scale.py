"""A value's resolution is its SYSTEM's scale, never its own row's -- one owner, three sites.

MEASURED 2026-10-13/14. Each shape below refused a correct solve because a physically zero
quantity was judged against a bar made of its own rounding:

* transformer_3ph_concentric `network_three_phase`, t = 2.05e-7: 89122 nodal A values at 2.7e-40
  Wb/m changing 2% per Newton iteration while the 300 V circuit rows sat still and the residual
  stayed 800x under its floor -- the step was refused 50 times and halved to 4.9 ns.
* the moving-band networks: joint-border soft rows with forcing 0, residual 1.1e-44, bar 1e-56.
* the same networks: free rates of 2.7e-51 on ``r53 + r54 = 0`` read as a 100% defect.

Each planted control is a defect AT SYSTEM SCALE, which must still be refused.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.gauss.linear.initial_sparse import projected_rate_consistent
from motronics.gauss.linear.initial_storage_border import _row_bars
from motronics.gauss.linear.resolution import system_resolution
from motronics.gauss.time_integrators._forms import change_below_residual_resolution

pytestmark = pytest.mark.unit


def test_the_system_resolution_is_the_rounding_of_its_largest_row():
    assert system_resolution(np.array([1e-40, 300.0]), np.array([2.0])) == pytest.approx(64 * 2.220446e-16 * 300)
    assert system_resolution(np.zeros(3)) == 0.0


# 1. the transient step's local check -------------------------------------------------------------

#: Unknowns (A, i, v): the A row is driven by the winding current i, which is zero at rest, while the
#: 300 V terminal row carries the system's scale.
_TANGENT = sp.csr_matrix(np.array([[1.0, 1e-3, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]))


def test_a_homogeneous_zero_field_driven_by_300_volts_is_resolved():
    before = np.array([2.74e-40, 0.0, 300.0])
    after = np.array([2.80e-40, 0.0, 300.0])  # the field's noise moved 2%, the circuit did not
    assert change_below_residual_resolution(_TANGENT, before, after)


def test_planted_control_a_field_change_at_system_scale_is_not_resolved():
    assert not change_below_residual_resolution(_TANGENT, np.array([0.0, 0.0, 300.0]), np.array([1e-3, 0.0, 300.0]))


# 2. the joint border's row bars ------------------------------------------------------------------


def _border():
    matrix = sp.csr_matrix(np.array([[300.0, 0.0], [0.0, 1e-10]]))
    trial = np.array([1.0, 1e-45])  # row 1's operands are rounding
    return matrix, trial


def test_a_noise_row_s_bar_is_its_system_s_resolution():
    matrix, trial = _border()
    bars = _row_bars(matrix, np.array([0, 1]), trial, np.zeros(2), np.zeros(2), 1e-10)
    assert bars[1] >= system_resolution(np.array([300.0]))
    assert bars[1] >= 1.1e-44, 'the measured 1.1e-44 rounding residual is inside it'


def test_planted_control_a_system_scale_row_keeps_its_own_relative_bar():
    matrix, trial = _border()
    bars = _row_bars(matrix, np.array([0, 1]), trial, np.zeros(2), np.zeros(2), 1e-10)
    assert bars[0] == pytest.approx(1e-10 * 300.0)


def test_a_row_with_no_operands_stays_exact():
    matrix = sp.csr_matrix(np.array([[300.0, 0.0], [0.0, 0.0]]))
    bars = _row_bars(matrix, np.array([0, 1]), np.array([1.0, 0.0]), np.zeros(2), np.zeros(2), 1e-10)
    assert bars[1] == 0.0


# 3. the projected-rate consistency ---------------------------------------------------------------

_JACOBIAN = sp.csr_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
_CONSTRAINT = sp.csr_matrix(np.array([[0.0, 1.0, 1.0]]))  # d/dt: r1 + r2 = 0


def test_noise_rates_on_a_differentiated_constraint_are_consistent():
    rate = np.array([1.0e3, 2.6727647e-51, 2.6727647e-51])
    assert projected_rate_consistent(_JACOBIAN, rate, np.zeros(3), algebraic_rows=_CONSTRAINT, tolerance=1e-10)


def test_planted_control_a_contradicted_constraint_at_system_scale_is_refused():
    rate = np.array([1.0e3, 1.0, 0.0])
    assert not projected_rate_consistent(_JACOBIAN, rate, np.zeros(3), algebraic_rows=_CONSTRAINT, tolerance=1e-10)
