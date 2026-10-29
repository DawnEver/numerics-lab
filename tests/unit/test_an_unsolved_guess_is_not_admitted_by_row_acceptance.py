"""Row acceptance admits only an iterate a Newton step produced, never the un-solved guess.

``RowAcceptance`` judges each row at ``ADMISSION_BAR`` of ``|F_u| |u| + |F_u'| |u'| + |F(u, 0)|``. On a
system settling onto a nonzero equilibrium that scale is the STATE's, not the remaining transient's, so
the previous step's state -- the guess -- passes as soon as its distance to the equilibrium drops below
1e-10 of the state, and the march stops decaying there. MEASURED 2026-10-24 on a downstream repo's
voltage-driven bordered march: 30 of 170 step checks admitted the guess with no solve, and the field froze
1.63e-9 short of its settled value against a 1e-11 bar; refusing the guess brought the gap to 2.6e-13.
"""

from types import SimpleNamespace

import numpy as np

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._step import _solve_step

_TAU = 1e-3
_EQUILIBRIUM = 1.0


def _backward_euler(previous, h):
    return SimpleNamespace(
        t_eval=h,
        weights=DerivativeWeights(1.0, 1.0 / h, 0.0),
        u_arg=lambda state: state,
        u_dot=lambda state: (state - previous) / h,
        u_ddot=np.zeros_like,
    )


class _Decay:
    """``u' + (u - 1) / tau = 0``: a linear relaxation onto a nonzero equilibrium."""

    def residual(self, state, rate, _acceleration, _time):
        return rate + (state - _EQUILIBRIUM) / _TAU

    def jacobian(self, _state, _rate, _acceleration, _time, weights):
        return np.atleast_2d(weights.alpha0 / _TAU + weights.alpha1)


def test_a_settling_decay_is_stepped_not_frozen_at_its_guess():
    gap = 1e-11  # 1e-11 of the state: under ADMISSION_BAR of |F_u| |u|, far over the step's own error
    previous = np.array([_EQUILIBRIUM + gap])
    u, converged, *_ = _solve_step(
        _Decay(),
        _backward_euler(previous, _TAU),
        previous.copy(),
        solve=np.linalg.solve,
        max_newton=4,
        tol=1e-12,
        row_acceptance=True,
    )
    assert converged
    # One exact backward-Euler step at h = tau halves the gap; the frozen guess keeps all of it.
    assert abs(u[0] - _EQUILIBRIUM) <= 0.5 * gap * (1 + 1e-3)
