"""Both sampled marches resume through the ONE TransientCheckpoint contract, bit for bit.

Straight ``[0, T]`` against ``[0, kill]`` + resume ``[kill, T]`` from the sealed FILE: the resumed
trajectory and the sampled task's own state agree with ``assert_array_equal``. A kill lands mid-segment
and on a segment end (the next hit's tasks not yet fired). Planted controls: an edited task history is
refused at read, a foreign key refuses at resume, and resuming WITHOUT the task history (the PI
integral restarted from zero) is visibly NOT the straight march -- the equality above can fail.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import Hold, SampledTask, integrate_explicit_sampled, integrate_sampled
from numerics_lab.time_integrators.checkpoint import read_checkpoint_file, write_checkpoint_file
from numerics_lab.time_integrators.waveform_relaxation import waveform_relaxation
from numerics_lab.time_integrators.zoh import march_zoh, zoh_discretize

_KEY = 'sampled-resume'
_T1, _DT, _TS = 1.0, 0.01, 0.07


class _Pi:
    """A sampled PI on ``1 - u``: its integral IS state the plant cannot reconstruct."""

    def __init__(self) -> None:
        self.out = Hold(0.0)
        self.integral = 0.0

    def update(self, _t, u) -> None:
        error = 1.0 - float(u[0])
        self.integral += 0.3 * error
        self.out.value = self.integral + 0.8 * error

    def checkpoint_history(self) -> dict:
        return {'integral': self.integral, 'out': self.out.value}

    def restore_checkpoint_history(self, data: dict) -> None:
        self.integral, self.out.value = float(data['integral']), float(data['out'])


class _Plant:
    """``0.2 u' + u**3 = hold``: nonlinear, so a resumed Newton history that drifted would show."""

    def __init__(self, hold: Hold) -> None:
        self.hold = hold

    def residual(self, u, u_dot, _u_ddot, _t):
        return 0.2 * u_dot + u**3 - self.hold.value

    def jacobian(self, u, _u_dot, _u_ddot, _t, weights):
        return np.atleast_2d(weights.alpha0.real * 3.0 * u**2 + 0.2 * weights.alpha1.real)


def _explicit(pi, t0, u0, **checkpoint):
    return integrate_explicit_sampled(
        lambda u, _t: (pi.out.value - u**3) / 0.2,
        t0,
        _T1,
        _DT,
        u0,
        [SampledTask(_TS, pi.update)],
        task_history=pi,
        **checkpoint,
    )


def _implicit(pi, t0, u0, **checkpoint):
    return integrate_sampled(
        _Plant(pi.out),
        TimeWindow(t0=t0, t1=_T1, dt=_DT, integrator=IntegratorKind.BDF2),
        u0,
        [SampledTask(_TS, pi.update)],
        task_history=pi,
        max_newton=20,
        tol=1e-13,
        **checkpoint,
    )


_MARCHES = {'explicit_rk4': _explicit, 'implicit_bdf2': _implicit}


class _Killed(RuntimeError):
    pass


def _kill_at(steps, records):
    def on_checkpoint(record):
        records.append(record)
        if len(records) == steps:
            raise _Killed

    return on_checkpoint


def _killed(march, steps, tmp_path):
    records = []
    with pytest.raises(_Killed):
        march(_Pi(), 0.0, np.zeros(1), checkpoint_key=_KEY, on_checkpoint=_kill_at(steps, records))
    return write_checkpoint_file(tmp_path / 'checkpoint.json', records[-1])


@pytest.mark.parametrize('name', sorted(_MARCHES))
@pytest.mark.parametrize('steps', [33, 7])  # 33: mid-segment; 7: t = 0.07, the end of the first segment
def test_straight_equals_killed_plus_resumed_bit_for_bit(tmp_path, name, steps):
    march = _MARCHES[name]
    straight_pi = _Pi()
    straight = march(straight_pi, 0.0, np.zeros(1), checkpoint_key=_KEY)
    saved = read_checkpoint_file(_killed(march, steps, tmp_path))
    assert saved.schedule_state is not None
    resumed_pi = _Pi()
    resumed = march(resumed_pi, saved.time, np.zeros(1), checkpoint_key=_KEY, restart=saved)
    start = straight.times.index(saved.time)
    np.testing.assert_array_equal(np.asarray(resumed.times), np.asarray(straight.times[start:]))
    np.testing.assert_array_equal(np.asarray(resumed.states), np.asarray(straight.states[start:]))
    assert resumed_pi.checkpoint_history() == straight_pi.checkpoint_history()
    assert resumed.checkpoint.to_mapping() == straight.checkpoint.to_mapping()


@pytest.mark.parametrize('name', sorted(_MARCHES))
def test_planted_controls(tmp_path, name):
    march = _MARCHES[name]
    path = _killed(march, 33, tmp_path)
    saved = read_checkpoint_file(path)
    straight = march(_Pi(), 0.0, np.zeros(1), checkpoint_key=_KEY)
    # Resuming with the task history ZEROED: the comparison above is able to fail.
    forgetful = _Pi()
    blank = type(saved).from_mapping({
        **saved.to_mapping(),
        'schedule_state': {**saved.schedule_state.to_mapping(), 'tasks': forgetful.checkpoint_history()},
    })
    wrong = march(forgetful, saved.time, np.zeros(1), checkpoint_key=_KEY, restart=blank)
    assert not np.array_equal(np.asarray(wrong.states[-1]), np.asarray(straight.states[-1]))
    with pytest.raises(ValueError, match='checkpoint'):
        march(_Pi(), saved.time, np.zeros(1), checkpoint_key='another model', restart=saved)
    sealed = json.loads(path.read_text(encoding='utf-8'))
    sealed['checkpoint']['schedule_state']['tasks']['integral'] += 1e-15
    path.write_text(json.dumps(sealed), encoding='utf-8')
    with pytest.raises(ValueError, match='sha256'):
        read_checkpoint_file(path)


@pytest.mark.parametrize('name', sorted(_MARCHES))
def test_a_checkpointed_march_with_tasks_but_no_task_history_refuses(name):
    pi = _Pi()
    march = _MARCHES[name]
    with pytest.raises(ValueError, match='task_history'):
        if name == 'explicit_rk4':
            integrate_explicit_sampled(
                lambda u, _t: -u, 0.0, _T1, _DT, np.zeros(1), [SampledTask(_TS, pi.update)], checkpoint_key=_KEY
            )
        else:
            integrate_sampled(
                _Plant(pi.out),
                TimeWindow(t0=0.0, t1=_T1, dt=_DT, integrator=IntegratorKind.BDF1),
                np.zeros(1),
                [SampledTask(_TS, pi.update)],
                checkpoint_key=_KEY,
            )
    assert march is not None


# ---------------------------------------------------------------- the exact zero-order-hold march

_AD, _COLUMNS = zoh_discretize(np.array([[0.0, 1.0], [-4.0, -0.4]]), (np.array([0.0, 1.0]),), 0.01)


class _Feedback:
    """A sampled integral law on the position: its integral is state the plant cannot rebuild."""

    def __init__(self) -> None:
        self.integral = 0.0

    def __call__(self, _t, x) -> tuple[float]:
        self.integral += 0.05 * (1.0 - x[0])
        return (2.0 * (1.0 - x[0]) + self.integral,)

    def checkpoint_history(self) -> dict:
        return {'integral': self.integral}

    def restore_checkpoint_history(self, data: dict) -> None:
        self.integral = float(data['integral'])


def _zoh(law, t0, n_steps, **checkpoint):
    return march_zoh(
        _AD, _COLUMNS, np.zeros(2), t0=t0, dt=0.01, n_steps=n_steps, control=law, control_history=law, **checkpoint
    )


def test_the_zoh_march_resumes_bit_for_bit_and_refuses_a_foreign_record(tmp_path):
    straight = _zoh(_Feedback(), 0.3, 100, checkpoint_key=_KEY)
    records = []
    with pytest.raises(_Killed):
        _zoh(_Feedback(), 0.3, 100, checkpoint_key=_KEY, on_checkpoint=_kill_at(37, records))
    saved = read_checkpoint_file(write_checkpoint_file(tmp_path / 'checkpoint.json', records[-1]))
    times, states = _zoh(_Feedback(), saved.time, 100 - 37, checkpoint_key=_KEY, restart=saved)
    np.testing.assert_array_equal(times, straight[0][37:])
    np.testing.assert_array_equal(states, straight[1][37:])
    # Planted: the law's integral restarted from zero is visibly another trajectory.
    blank = type(saved).from_mapping({
        **saved.to_mapping(),
        'schedule_state': {**saved.schedule_state.to_mapping(), 'tasks': {'integral': 0.0}},
    })
    _times, wrong = _zoh(_Feedback(), saved.time, 100 - 37, checkpoint_key=_KEY, restart=blank)
    assert not np.array_equal(wrong[-1], straight[1][-1])
    with pytest.raises(ValueError, match='checkpoint'):
        _zoh(_Feedback(), saved.time, 100 - 37, checkpoint_key='another model', restart=saved)


# ---------------------------------------------------------------- a fixed point OVER a march


def _contraction(x):
    return 0.5 * np.cos(np.asarray(x, dtype=float)) + np.linspace(0.0, 1.0, 5), None


def test_waveform_relaxation_resumes_from_its_iterate_not_mid_window():
    """WR is a fixed point over whole windows: its resumable state is the relaxed iterate, handed back as the seed.

    Straight to the bar against six sweeps + a resume seeded with their state: the same final state bit for
    bit, in the same total sweep count. Planted: a resume that forgot its iterate (seeded from the start again)
    saves no sweep, so the count identity is falsifiable.
    """
    seed = np.zeros(5)
    straight = waveform_relaxation(_contraction, seed, relaxation=0.7, tol=1e-13, max_iter=200)
    partial = waveform_relaxation(_contraction, seed, relaxation=0.7, tol=1e-300, max_iter=6)
    resumed = waveform_relaxation(_contraction, partial.state, relaxation=0.7, tol=1e-13, max_iter=200)
    np.testing.assert_array_equal(resumed.state, straight.state)
    assert resumed.iterations + 6 == straight.iterations
    forgot = waveform_relaxation(_contraction, seed, relaxation=0.7, tol=1e-13, max_iter=200)
    assert forgot.iterations + 6 != straight.iterations
