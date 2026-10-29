"""A homogeneous row holding only the solve's rounding is at the ROUNDOFF FLOOR of its system.

The norm-verdict floor (``_at_roundoff_floor``) judged every row by its own ``eps |J| |u|`` (plus, since
3ba7e65, its source and rate). A homogeneous row -- a field coordinate that is physically zero beside a
coupled 1e6-scale circuit -- has operands that are themselves rounding, so its own floor sits ~1e-14 below
its own residual for every iterate. MEASURED 2026-10-24 on a downstream repo's
``test_switched_network_checkpoint_preserves_actual_trips_and_full_storage[bdf2]``: the event probe at
t = 0.00147075 left rows 1, 2, 3, 34, 42, 46 at 2e-69..4e-68 against floors of 3e-83..5e-82, every circuit
row inside its floor, and Newton exhausted 50 iterations, so the event could not be located.
``RowAcceptance`` (80bc916) and ``change_below_residual_resolution`` already judge such a row at the
system's resolution; the floor now asks the same owner (`numerics_lab.linear.resolution.row_bar`).
The planted control: a row with a DECLARED tiny source keeps its own strict floor.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._step import _solve_step

_SCALE = 1e6
_NOISE = 1e-69


def _form():
    return SimpleNamespace(
        t_eval=0.0,
        weights=DerivativeWeights(1.0, 0.0, 0.0),
        u_arg=lambda state: state,
        u_dot=np.zeros_like,
        u_ddot=np.zeros_like,
    )


class _Coupled:
    """``u0 = 1e6`` and ``u1 = source``: the second row's only source is the declared one."""

    def __init__(self, source) -> None:
        self.source = source

    def residual(self, state, _rate, _acceleration, _time):
        return np.array([state[0] - _SCALE, state[1] - self.source])

    def jacobian(self, _state, _rate, _acceleration, _time, weights):
        return weights.alpha0 * np.eye(2)


def _rounding_solve(matrix, rhs):
    """An exact solve leaving the coupled system's rounding: 1e-10 on the 1e6 row, 1e-69 on the zero one."""
    return np.linalg.solve(matrix, rhs) + np.array([1e-10, _NOISE])


@pytest.mark.parametrize('source', [0.0, 1e-60])
def test_only_a_homogeneous_row_is_at_its_systems_roundoff_floor(source):
    _root, converged, *_ = _solve_step(
        _Coupled(source),
        _form(),
        np.array([0.0, 0.0]),
        solve=_rounding_solve,
        max_newton=4,
        tol=1e-12,
    )
    # The 1e6 row is inside its own floor in both cases; only a homogeneous zero row is noise.
    assert bool(converged) is (source == 0.0)
