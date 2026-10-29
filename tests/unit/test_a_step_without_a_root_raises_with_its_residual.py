"""Solver honesty: a step whose Newton cannot converge never joins a returned trajectory.

The planted case is the srdab shape reduced to one unknown: a source whose GATE is decided by the
state it drives, ``u_dot + u = E * gate(u)`` with ``gate = u < 0.5``. Past the threshold there is no
root at all -- each mode's root lies in the other mode -- so Newton chatters between them. The march
must RAISE with that residual and the last converged state; the planted control is the same circuit
with the gate frozen, which converges and returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import StepNotConverged, integrate_transient


class _GateCoupledSource:
    def __init__(self, frozen: bool = False) -> None:
        self.frozen = frozen

    def residual(self, u, u_dot, u_ddot, t):
        gate = 1.0 if self.frozen or u[0] < 0.5 else 0.0
        return u_dot + u - 1.0 * gate

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.atleast_2d(weights.alpha0.real + weights.alpha1.real)


_STUDY = TimeWindow(t0=0.0, t1=2.0, dt=0.1, integrator=IntegratorKind.BDF2)


def test_a_rootless_gate_coupled_step_raises_with_its_residual_and_the_state_reached() -> None:
    with pytest.raises(StepNotConverged) as info:
        integrate_transient(_GateCoupledSource(), _STUDY, np.zeros(1), max_newton=20)
    err = info.value
    assert err.residual > 1e-3
    assert err.t_reached < err.t
    assert float(err.u_reached[0]) < 0.5


def test_planted_control_the_frozen_gate_converges_and_returns() -> None:
    sol = integrate_transient(_GateCoupledSource(frozen=True), _STUDY, np.zeros(1), max_newton=20)
    assert sol.converged
    assert sol.times[-1] == pytest.approx(2.0)


def test_label_mode_records_instead_of_raising() -> None:
    sol = integrate_transient(_GateCoupledSource(), _STUDY, np.zeros(1), max_newton=20, on_nonconverged='label')
    assert not sol.converged
    assert sol.nonconverged_times
