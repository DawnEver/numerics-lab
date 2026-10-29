"""Explicit time tables and periodic convergence use accepted trajectories."""

import numpy as np
import pytest

from numerics_lab.formulation import TimeWindow
from numerics_lab.time_integrators import _step, integrate_transient
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint
from numerics_lab.time_integrators.periodic import periodic_converged


class Ramp:
    def residual(self, u, u_dot, u_ddot, t):
        return u_dot - 2 * t

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * weights.alpha1.real


def test_step_table_uses_variable_bdf_history():
    times = (0.0, 0.1, 0.25, 0.4, 0.6)
    solution = integrate_transient(Ramp(), TimeWindow(0, 0.6, 0.1), np.array([0.0]), step_times=times)
    assert solution.times == list(times)
    # First backward Euler step introduces .01 error; variable BDF2 is exact for quadratics.
    assert abs(solution.states[-1][0] - 0.36) < 0.025


def test_explicit_step_table_lands_the_representable_instant_after_start():
    corner = float(np.nextafter(0.5, 1.0))
    times = (0.5, corner, 0.75, 1.0)
    checkpoints = []
    solution = integrate_transient(
        Ramp(),
        TimeWindow(0.5, 1.0, 0.25),
        np.zeros(1),
        step_times=times,
        checkpoint_key='actual-source-corner',
        on_checkpoint=checkpoints.append,
    )
    assert solution.converged
    assert solution.times == list(times)
    assert any(checkpoint.time == corner for checkpoint in checkpoints)


def test_invalid_step_table_is_refused():
    with pytest.raises(ValueError, match='step_times'):
        integrate_transient(Ramp(), TimeWindow(0, 1, 0.1), np.zeros(1), step_times=(0.0, 0.2, 0.1, 1.0))


class Constant:
    def residual(self, u, u_dot, u_ddot, t):
        return u - 1

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1)


def test_periodic_stop_needs_two_complete_matching_cycles():
    solution = integrate_transient(Constant(), TimeWindow(0, 10, 0.1), np.ones(1), steady_period=1.0, steady_atol=1e-10)
    assert solution.times[-1] == pytest.approx(2.0)


def test_matching_endpoints_do_not_hide_changing_period_shape():
    times = np.linspace(0.0, 2.0, 9)
    states = [(1.0 + t) * np.sin(2.0 * np.pi * t) * np.ones(1) for t in times]
    assert not periodic_converged(times, states, 1.0, 1e-10, 1e-3)


def test_adaptive_line_search_is_forwarded(monkeypatch):
    original = _step._solve_step
    modes = []

    def record(*args, **kwargs):
        modes.append(kwargs.get('line_search'))
        return original(*args, **kwargs)

    monkeypatch.setattr(_step, '_solve_step', record)
    integrate_transient(Constant(), TimeWindow(0, 0.2, 0.1), np.ones(1), adaptive=True, line_search='energy')
    assert modes and set(modes) == {'energy'}


class SlowDecay:
    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + 0.01 * u

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * (0.01 * weights.alpha0.real + weights.alpha1.real)


def test_eec_removes_slow_cycle_mean_transient():
    study = TimeWindow(0, 2.0, 0.1)
    plain = integrate_transient(SlowDecay(), study, np.ones(1))
    corrected = integrate_transient(SlowDecay(), study, np.ones(1), eec_period=1.0, eec_affine_first_order=True)
    assert abs(corrected.states[-1][0]) < abs(plain.states[-1][0]) / 100
    # The first endpoint remains the physical solve, not the EEC jump.
    assert corrected.states[10][0] == pytest.approx(plain.states[10][0])
    assert corrected.corrected_times == [1.0, 2.0]


def test_eec_requires_matched_period_steps():
    with pytest.raises(ValueError, match='correction period'):
        integrate_transient(
            SlowDecay(), TimeWindow(0, 2.0, 0.3), np.ones(1), eec_period=1.0, eec_affine_first_order=True
        )


def test_eec_does_not_infer_an_affine_operator_from_a_generic_problem():
    with pytest.raises(ValueError, match='explicit assertion'):
        integrate_transient(SlowDecay(), TimeWindow(0, 2.0, 0.1), np.ones(1), eec_period=1.0)


def test_eec_restart_refuses_missing_cycle_origin_history():
    checkpoint = TransientCheckpoint('decay', 'bdf2', 1.0, 0.1, ((0.9,),))
    with pytest.raises(ValueError, match='cycle-origin history'):
        integrate_transient(
            SlowDecay(),
            TimeWindow(1.0, 3.0, 0.1),
            np.ones(1),
            restart=checkpoint,
            checkpoint_key='decay',
            eec_period=1.0,
            eec_affine_first_order=True,
        )


def test_an_explicit_table_records_the_parent_lattice_it_was_cut_from():
    """ONE MEANING (2026-10-10): `step_times` is the grid; `step_origin` is only its PROVENANCE --
    the parent lattice ``origin + k dt`` the table was derived from -- carried on the checkpoint so a
    restart can re-derive the same table. It never moves a step.
    """
    times = (0.0, 0.1, 0.25, 0.4, 0.6)
    captured = []
    solution = integrate_transient(
        Ramp(),
        TimeWindow(0, 0.6, 0.1),
        np.array([0.0]),
        step_times=times,
        step_origin=-0.2,
        checkpoint_key='table-provenance',
        on_checkpoint=captured.append,
    )
    assert solution.times == list(times)
    assert captured and all(checkpoint.step_origin == -0.2 for checkpoint in captured)


def test_a_table_provenance_after_t0_is_refused():
    with pytest.raises(ValueError, match='step_origin'):
        integrate_transient(Ramp(), TimeWindow(0, 0.6, 0.1), np.zeros(1), step_times=(0.0, 0.3, 0.6), step_origin=0.1)
