"""Explicit time tables and periodic convergence use accepted trajectories."""

import numpy as np
import pytest

import motronics.gauss.time_integrators as runtime
from motronics.contracts.spec.study_spec import TransientStudy
from motronics.gauss.time_integrators import integrate_transient
from motronics.gauss.time_integrators.checkpoint import TransientCheckpoint
from motronics.gauss.time_integrators.periodic import periodic_converged


class Ramp:
    def residual(self, u, u_dot, u_ddot, t):
        return u_dot - 2 * t

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * weights.alpha1.real


def test_step_table_uses_variable_bdf_history():
    times = (0.0, 0.1, 0.25, 0.4, 0.6)
    solution = integrate_transient(Ramp(), TransientStudy(0, 0.6, 0.1), np.array([0.0]), step_times=times)
    assert solution.times == list(times)
    # First backward Euler step introduces .01 error; variable BDF2 is exact for quadratics.
    assert abs(solution.states[-1][0] - 0.36) < 0.025


def test_invalid_step_table_is_refused():
    with pytest.raises(ValueError, match='step_times'):
        integrate_transient(Ramp(), TransientStudy(0, 1, 0.1), np.zeros(1), step_times=(0.0, 0.2, 0.1, 1.0))


class Constant:
    def residual(self, u, u_dot, u_ddot, t):
        return u - 1

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1)


def test_periodic_stop_needs_two_complete_matching_cycles():
    solution = integrate_transient(
        Constant(), TransientStudy(0, 10, 0.1), np.ones(1), steady_period=1.0, steady_atol=1e-10
    )
    assert solution.times[-1] == pytest.approx(2.0)


def test_matching_endpoints_do_not_hide_changing_period_shape():
    times = np.linspace(0.0, 2.0, 9)
    states = [(1.0 + t) * np.sin(2.0 * np.pi * t) * np.ones(1) for t in times]
    assert not periodic_converged(times, states, 1.0, 1e-10, 1e-3)


def test_adaptive_line_search_is_forwarded(monkeypatch):
    original = runtime._solve_step
    modes = []

    def record(*args, **kwargs):
        modes.append(kwargs.get('line_search'))
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, '_solve_step', record)
    integrate_transient(Constant(), TransientStudy(0, 0.2, 0.1), np.ones(1), adaptive=True, line_search='energy')
    assert modes and set(modes) == {'energy'}


class SlowDecay:
    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + 0.01 * u

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * (0.01 * weights.alpha0.real + weights.alpha1.real)


def test_eec_removes_slow_cycle_mean_transient():
    study = TransientStudy(0, 2.0, 0.1)
    plain = integrate_transient(SlowDecay(), study, np.ones(1))
    corrected = integrate_transient(SlowDecay(), study, np.ones(1), eec_period=1.0, eec_affine_first_order=True)
    assert abs(corrected.states[-1][0]) < abs(plain.states[-1][0]) / 100
    # The first endpoint remains the physical solve, not the EEC jump.
    assert corrected.states[10][0] == pytest.approx(plain.states[10][0])
    assert corrected.corrected_times == [1.0, 2.0]


def test_eec_requires_matched_period_steps():
    with pytest.raises(ValueError, match='correction period'):
        integrate_transient(
            SlowDecay(), TransientStudy(0, 2.0, 0.3), np.ones(1), eec_period=1.0, eec_affine_first_order=True
        )


def test_eec_does_not_infer_an_affine_operator_from_a_generic_problem():
    with pytest.raises(ValueError, match='explicit assertion'):
        integrate_transient(SlowDecay(), TransientStudy(0, 2.0, 0.1), np.ones(1), eec_period=1.0)


def test_eec_restart_refuses_missing_cycle_origin_history():
    checkpoint = TransientCheckpoint('decay', 'bdf2', 1.0, 0.1, ((0.9,),))
    with pytest.raises(ValueError, match='cycle-origin history'):
        integrate_transient(
            SlowDecay(),
            TransientStudy(1.0, 3.0, 0.1),
            np.ones(1),
            restart=checkpoint,
            checkpoint_key='decay',
            eec_period=1.0,
            eec_affine_first_order=True,
        )
