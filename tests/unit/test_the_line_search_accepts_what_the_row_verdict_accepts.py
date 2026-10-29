"""Under ``row_acceptance`` the line search takes a trial the row verdict accepts, whatever its NORM.

A planted coupled step: a 4e7-scale circuit row sitting at its rounding (it reshuffles from 1e-8 to 3e-8 as the
linkage closes, far under its own 4e-3 bar) and a small linkage row that one Newton step closes. The norm
is the circuit row's noise, so the exact step RAISES it: a norm merit halves it to the cap and the linkage
row creeps (a downstream repo's hysteresis chopper, 2026-10-25: per-step stage exhausted at a trip probe with its
linkage row 1.45e-10 over a 6.0e-11 bar). The row verdict accepts the full step, so the search must too.
The control: without ``row_acceptance`` the norm march is unchanged; its own arithmetic-floor escape
already takes that step.
"""

from __future__ import annotations

import numpy as np

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient

_C = 2.0
_START = np.array([1.0, _C + 1e-5])


class _NoisyCircuitAndLinkage:
    def residual(self, u, u_dot, u_ddot, t):
        # The circuit row's rounding, uncorrelated with its Jacobian: it reshuffles from 1e-8 to 3e-8 as the
        # linkage closes, so every trial toward the root RAISES the norm the linkage row hides under.
        noise = 3e-8 - 2e-8 * min(1.0, abs(u[1] - _C) / 1e-5)
        return np.array([4e7 * (u[0] - 1.0) + noise, 1e-4 * (u[1] - _C)])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.diag([4e7, 1e-4]) * weights.alpha0.real


_STUDY = TimeWindow(t0=0.0, t1=1.0, dt=1.0, integrator=IntegratorKind.BDF1)


def _march(**bars):
    return integrate_transient(
        _NoisyCircuitAndLinkage(), _STUDY, _START.copy(), max_newton=20, backtrack_cap=8, backtrack_band=None, **bars
    )


def test_a_row_accepted_trial_is_taken_whatever_its_norm() -> None:
    solution = _march(row_acceptance=True)
    assert solution.converged
    assert abs(solution.states[-1][1] - _C) < 1e-12


def test_control_the_norm_march_keeps_its_own_floor_escape() -> None:
    # Without the row verdict nothing changes: the norm march already takes a trial at its arithmetic floor.
    solution = _march(tol_rel=1e-14)
    assert solution.converged
    assert abs(solution.states[-1][1] - _C) < 1e-12
