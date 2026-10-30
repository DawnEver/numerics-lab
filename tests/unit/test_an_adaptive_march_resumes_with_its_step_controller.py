"""A local-error-controlled march resumes through the ONE checkpoint contract, its controller included.

``step_control`` sizes each step from the predictor through the last ``order + 1`` accepted points and
proposes the next step ``h``. Both are history: a resume without them restarts the controller (step
doubling, a fresh ``h``) and lands other instants. Straight ``[0, T]`` against killed + resumed from the
sealed file, BDF1 / BDF2 / TR-BDF2: ``assert_array_equal`` on times and states. Planted controls: the
record with its controller state stripped is refused (the march declares step control), a record whose
next step is nudged resumes onto other instants, and a resume under another tolerance refuses.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import StepTolerance, integrate_transient
from numerics_lab.time_integrators.checkpoint import (
    TransientCheckpoint,
    read_checkpoint_file,
    write_checkpoint_file,
)

_KEY = 'adaptive-resume'
_TOL = StepTolerance(rtol=1e-6, atol=1e-9)
#: The window: long enough that the controller moves the step and step 40 is mid-march for every
#: integrator, and no longer -- the claim is bit-for-bit resume, not reach; BDF1 at [0, 6] took 22 s.
_T1 = 2.0


class _Duffing:
    """``x' = v``, ``v' = -0.3 v - x - x**3 + 2 cos(1.2 t)``: nonlinear and driven, so the step size moves."""

    def residual(self, u, u_dot, _u_ddot, t):
        x, v = u
        return np.array([u_dot[0] - v, u_dot[1] + 0.3 * v + x + x**3 - 2.0 * np.cos(1.2 * t)])

    def jacobian(self, u, _u_dot, _u_ddot, _t, weights):
        a0, a1 = weights.alpha0.real, weights.alpha1.real
        x = u[0]
        return np.array([[a1, -a0], [a0 * (1.0 + 3.0 * x**2), a1 + 0.3 * a0]])


class _Killed(RuntimeError):
    pass


def _march(t0, integrator, *, tol=_TOL, **checkpoint):
    return integrate_transient(
        _Duffing(),
        TimeWindow(t0=t0, t1=_T1, dt=0.2, integrator=integrator),
        np.array([0.5, 0.0]),
        max_newton=30,
        tol=1e-12,
        step_control=tol,
        checkpoint_key=_KEY,
        **checkpoint,
    )


def _killed(integrator, steps, tmp_path):
    records = []

    def on_checkpoint(record):
        records.append(record)
        if len(records) == steps:
            raise _Killed

    with pytest.raises(_Killed):
        _march(0.0, integrator, on_checkpoint=on_checkpoint)
    return write_checkpoint_file(tmp_path / 'checkpoint.json', records[-1])


_INTEGRATORS = [IntegratorKind.BDF1, IntegratorKind.BDF2, IntegratorKind.TR_BDF2]


@pytest.mark.parametrize('integrator', _INTEGRATORS)
def test_straight_equals_killed_plus_resumed_bit_for_bit(tmp_path, integrator):
    straight = _march(0.0, integrator)
    assert len(set(np.round(np.diff(straight.times), 12))) > 3, 'FLOOR: the controller never moved the step'
    saved = read_checkpoint_file(_killed(integrator, 40, tmp_path))
    assert saved.step_control_state is not None
    resumed = _march(saved.time, integrator, restart=saved)
    start = straight.times.index(saved.time)
    np.testing.assert_array_equal(np.asarray(resumed.times), np.asarray(straight.times[start:]))
    np.testing.assert_array_equal(np.asarray(resumed.states), np.asarray(straight.states[start:]))
    assert resumed.checkpoint.to_mapping() == straight.checkpoint.to_mapping()


def test_planted_controls(tmp_path):
    integrator = IntegratorKind.BDF2
    saved = read_checkpoint_file(_killed(integrator, 40, tmp_path))
    straight = _march(0.0, integrator)
    mapping = saved.to_mapping()
    stripped = {key: value for key, value in mapping.items() if key != 'step_control_state'}
    with pytest.raises(ValueError, match='step control'):
        _march(saved.time, integrator, restart=TransientCheckpoint.from_mapping(stripped))
    nudged = {**mapping, 'step_control_state': {**mapping['step_control_state']}}
    nudged['step_control_state']['next_step'] *= 1.0 + 1e-9
    other = _march(saved.time, integrator, restart=TransientCheckpoint.from_mapping(nudged))
    start = straight.times.index(saved.time)
    assert other.times != straight.times[start:]
    with pytest.raises(ValueError, match='step control'):
        _march(saved.time, integrator, tol=StepTolerance(rtol=1e-5, atol=1e-9), restart=saved)
