"""A rate built from an unknown that is itself a CANCELLING SUM cannot resolve below that sum's rounding.

MEASURED 2026-10-24 (a downstream repo, the case_a native field, base 2 / 0.6 / 0.5 mm): every refinement refused its
first step on the same circuit row, ``R i - V_N + dlambda/dt``, stalled at 1.1x / 6.5x / 24x its bar. That bar
was the construction floor ``FLOOR_SAFETY eps |F_u'| |alpha1 lambda|`` -- but ``lambda = C^T A`` is a sum over
the winding's field nodes whose terms are ~1e3x ``|lambda|`` and cancel (the magnet flux enters and leaves),
so ``lambda`` is resolved to ``eps |C|^T |A|``, never to ``eps |lambda|``, and ``alpha1 = 1e6`` carries that
resolution into the row. The problem knows the sum; it declares ``state_resolution(u)`` and the floor reads it.
"""

from types import SimpleNamespace

import numpy as np

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._step import _solve_step

ALPHA1 = 1e6
LAMBDA = 1e-12  # the linkage: its step change makes a rate of 1e-6, as at the case_a first step
SUM = 1e-3  # the magnitude of the two field terms it cancels from
EPS = float(np.finfo(float).eps)
MISS = 10 * EPS * SUM  # the linkage's rounding as the sum leaves it; alpha1 x MISS ~ 2e-12 on the rate row


class _Problem:
    """``u = [A1, A2, lambda, g]``: two field terms, the linkage as their sum, and the rate row ``lambda' - g``."""

    def __init__(self, declares: bool) -> None:
        if declares:
            self.state_resolution = lambda u: np.array([0.0, 0.0, abs(u[0]) + abs(u[1]), 0.0])

    def residual(self, state, rate, _acceleration, _time):
        a1, a2, lam, g = state
        return np.array([a1 - SUM, a2 + SUM - LAMBDA, lam - (a1 + a2), rate[2] - g])

    def jacobian(self, _state, _rate, _acceleration, _time, weights):
        a0, a1 = weights.alpha0, weights.alpha1
        return np.array([
            [a0, 0.0, 0.0, 0.0],
            [0.0, a0, 0.0, 0.0],
            [-a0, -a0, a0, 0.0],
            [0.0, 0.0, a1, -a0],
        ])


_ORIGIN = np.zeros(4)
_FORM = SimpleNamespace(
    t_eval=0.0,
    weights=DerivativeWeights(1.0, ALPHA1, 0.0),
    u_arg=lambda state: state,
    u_dot=lambda state: ALPHA1 * (state - _ORIGIN),
    u_ddot=np.zeros_like,
)


def _rounded(matrix, rhs):
    """An exact solve that leaves the linkage off by the sum's rounding, as the field sum does."""
    out = np.linalg.solve(matrix, rhs)
    out[2] += MISS
    return out


def _converged(declares: bool) -> bool:
    _u, converged, *_ = _solve_step(
        _Problem(declares),
        _FORM,
        _ORIGIN.copy(),
        solve=_rounded,
        max_newton=4,
        tol=1e-50,
        row_acceptance=True,
    )
    return bool(converged)


def test_a_declared_resolution_floors_the_rate_row_at_the_sums_rounding():
    assert _converged(declares=True)


def test_without_the_declaration_the_same_rounding_is_refused():
    """PLANTED CONTROL: the floor at eps |lambda| cannot see the sum, and refuses its rounding."""
    assert not _converged(declares=False)
