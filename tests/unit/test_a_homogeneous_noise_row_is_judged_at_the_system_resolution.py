"""A homogeneous row holding only the solve's rounding is judged at the system's resolution.

``u1 = 0`` beside a 1e3-scale row: the coupled solve leaves ``u1`` at rounding size, and the row's own bar
``ADMISSION_BAR |u1|`` is below its own residual ``|u1|`` for EVERY nonzero iterate, so Newton could never
accept it (measured on a downstream repo's native PM conducting stack, 2026-10-23: ``r = 6.0e-51``, bar 0). The
row's source is read as its affine intercept at the ACTUAL state, ``F(u, 0) - F_u u``, so no zero state is
ever queried. The planted control: a row with a DECLARED tiny source keeps its own strict bar.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._step import _solve_step

_NOISE = 6e-51


def _form():
    return SimpleNamespace(
        t_eval=0.0,
        weights=DerivativeWeights(1.0, 0.0, 0.0),
        u_arg=lambda state: state,
        u_dot=np.zeros_like,
        u_ddot=np.zeros_like,
    )


class _Coupled:
    """``u0 = 1e3`` and ``u1 = source``: the second row's only source is the declared one."""

    def __init__(self, source) -> None:
        self.source = source

    def residual(self, state, _rate, _acceleration, _time):
        return np.array([state[0] - 1e3, state[1] - self.source])

    def jacobian(self, _state, _rate, _acceleration, _time, weights):
        return weights.alpha0 * np.eye(2)


def _rounding_solve(matrix, rhs):
    """An exact solve whose second component carries the coupled system's rounding."""
    return np.linalg.solve(matrix, rhs) + np.array([0.0, _NOISE])


@pytest.mark.parametrize('source', [0.0, 1e-50])
def test_only_a_homogeneous_row_is_judged_at_the_system_resolution(source):
    _root, converged, *_ = _solve_step(
        _Coupled(source),
        _form(),
        np.array([0.0, 0.0]),
        solve=_rounding_solve,
        max_newton=4,
        tol=1e-50,
        row_acceptance=True,
    )
    # A homogeneous row at rounding size is noise; a declared 1e-50 source missed by 6e-51 is a real miss.
    assert bool(converged) is (source == 0.0)


@pytest.mark.parametrize('source', [0.0, 1e-50])
def test_the_roundoff_floor_judges_the_same_noise_row_the_same_way(source):
    """The floor path (no row acceptance) reads the one noise rule too: two answers would be a drift."""
    _root, converged, _iterations, _residual, on_floor = _solve_step(
        _Coupled(source),
        _form(),
        np.array([0.0, 0.0]),
        solve=_rounding_solve,
        max_newton=4,
        tol=1e-300,
    )
    assert bool(converged) is bool(on_floor) is (source == 0.0)
