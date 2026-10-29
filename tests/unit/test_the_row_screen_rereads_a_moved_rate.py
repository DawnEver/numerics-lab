"""The row screen may reuse bars only while the ROW SCALE cannot have moved -- and the scale reads the RATE.

MEASURED 2026-10-24 (a downstream repo, the case_a native field at 0.5 / 0.6 mm): the first iterate of a step from
rest was judged with rest-sized bars (a circuit row whose scale is ``|dlambda/dt|``, zero at rest). Every later
iterate moved the state by ~1e-6 against a field max of ~0.1 -- "near" in the state's max norm -- so the screen
refused it UNREAD against those stale bars, 49 times, and the step failed naming a residual of the FIRST iterate.
But ``u' = alpha1 (u - origin)``: with ``alpha1 = 1e6`` a state change that is near is a rate change that is not,
and the scale ``|F_u'| |u'|`` moved from zero to the row's whole size.
"""

from types import SimpleNamespace

import numpy as np

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._step import _solve_step

ALPHA1 = 1e6
K = 4e-6
_ORIGIN = np.array([1e3, 0.0, 0.0])


class _Problem:
    """Row 0: a large settled unknown. Row 1: ``u1' - u2`` (no forcing of its own). Row 2: ``u2 - K``."""

    def residual(self, state, rate, _acceleration, _time):
        return np.array([state[0] - 1e3, rate[1] - state[2], state[2] - K])

    def jacobian(self, _state, _rate, _acceleration, _time, weights):
        a0, a1 = weights.alpha0, weights.alpha1
        return np.array([[a0, 0.0, 0.0], [0.0, a1, -a0], [0.0, 0.0, a0]])


_FORM = SimpleNamespace(
    t_eval=0.0,
    weights=DerivativeWeights(1.0, ALPHA1, 0.0),
    u_arg=lambda state: state,
    u_dot=lambda state: ALPHA1 * (state - _ORIGIN),  # BDF1 centred on the previous state, as the march builds it
    u_ddot=np.zeros_like,
)


def _slightly_off(matrix, rhs):
    """An exact solve but for a persistent 1e-24 miss on the rate unknown: a residual of 1e-18 on row 1, far under
    its real bar (1e-10 of its 4e-6 scale) and over the rest bar (zero) every time.
    """
    out = np.linalg.solve(matrix, rhs)
    out[1] += 1e-24
    return out


def test_an_iterate_whose_rate_moved_is_reread_not_screened_against_rest_bars():
    _u, converged, iterations, _residual, _floor = _solve_step(
        _Problem(),
        _FORM,
        _ORIGIN.copy(),
        solve=_slightly_off,
        max_newton=4,
        tol=1e-50,
        row_acceptance=True,
    )
    assert converged
    assert iterations == 2
