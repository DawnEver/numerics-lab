"""The predictor discrepancy measures curvature, not the BDF2 local error."""

import numpy as np
import pytest

import numerics_lab.time_integrators as runtime
from numerics_lab.formulation import TimeWindow
from numerics_lab.time_integrators import _step


class QuadraticTrajectory:
    """u'=2t has u=t², which variable-step BDF2 reproduces exactly."""

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot - 2.0 * t

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * weights.alpha1.real


def test_adaptive_growth_uses_the_quadratic_predictor_discrepancy(monkeypatch):
    original = _step._solve_step
    trial_times = []

    def record(problem, form, *args, **kwargs):
        trial_times.append(form.t_eval)
        return original(problem, form, *args, **kwargs)

    monkeypatch.setattr(_step, '_solve_step', record)
    dt, atol = 0.1, 0.04
    solution = runtime.integrate_transient(
        QuadraticTrajectory(),
        TimeWindow(0.0, 0.6, dt),
        np.zeros(1),
        initial_history=(np.array([dt * dt]), np.zeros(1)),
        curvature_atol=atol,
        h_min=1e-6,
        h_max=0.5,
    )
    # At equal spacings, u(t+h)-u_extrap = h(h+h_prev) = 2h².
    # The next proposal therefore uses order two, not the BDF2 LTE order three.
    expected_growth = 1.2 * np.sqrt(atol / (2.0 * dt * dt))
    assert trial_times[1] - trial_times[0] == pytest.approx(dt * expected_growth)
    # This proposal exceeds the curvature bar and is retried, not recorded as accepted.
    assert trial_times[1] not in solution.times
    assert solution.converged


def test_quantized_adaptive_retries_keep_absolute_ticks_and_label_unmet_floor(monkeypatch):
    original = _step._solve_step
    trials = []

    def record(problem, form, *args, **kwargs):
        trials.append(form.t_eval)
        return original(problem, form, *args, **kwargs)

    monkeypatch.setattr(_step, '_solve_step', record)
    origin, quantum = 0.37, 0.1
    solution = runtime.integrate_transient(
        QuadraticTrajectory(),
        TimeWindow(origin, origin + 0.8, 0.2),
        np.array([origin**2]),
        initial_history=(np.array([(origin - 0.2) ** 2]), np.array([origin**2])),
        curvature_atol=0.001,
        timestep_quantum=quantum,
        timestep_origin=origin,
    )
    np.testing.assert_allclose(
        (np.asarray(trials) - origin) / quantum, np.rint((np.asarray(trials) - origin) / quantum), atol=1e-12
    )
    assert trials[0] > trials[1]  # the two-tick proposal was rejected and retried
    assert solution.indicator_unmet_times == solution.times[1:]
    assert not solution.indicator_satisfied
    assert solution.converged  # independent Newton residual verdict
    for time, state in zip(solution.times, solution.states, strict=True):
        assert state[0] == pytest.approx(time**2, abs=1e-12)


def test_quantized_floor_uses_ticks_not_subtraction_roundoff(monkeypatch):
    original = _step._solve_step
    trials = []

    def record(problem, form, *args, **kwargs):
        trials.append(form.t_eval)
        assert len(trials) <= 4, 'a floor tick was rejected and retried without clock progress'
        return original(problem, form, *args, **kwargs)

    monkeypatch.setattr(_step, '_solve_step', record)
    origin, quantum = 0.37, 0.001
    assert (origin + quantum) - origin > quantum
    solution = runtime.integrate_transient(
        QuadraticTrajectory(),
        TimeWindow(origin, origin + 4 * quantum, quantum),
        np.array([origin**2]),
        initial_history=(np.array([(origin - quantum) ** 2]), np.array([origin**2])),
        curvature_atol=1e-30,
        timestep_quantum=quantum,
        timestep_origin=origin,
    )
    np.testing.assert_allclose(solution.times, origin + np.arange(5) * quantum, rtol=0, atol=0)
    assert len(trials) == 4
    assert solution.indicator_unmet_times == solution.times[1:]
    assert not solution.indicator_satisfied
    assert solution.converged


def test_quantized_adaptive_rejects_a_fractional_final_tick():
    with pytest.raises(ValueError, match='lattice'):
        runtime.integrate_transient(
            QuadraticTrajectory(),
            TimeWindow(0.37, 0.72, 0.1),
            np.array([0.37**2]),
            curvature_atol=1e-6,
            timestep_quantum=0.1,
            timestep_origin=0.37,
        )


def test_quantized_adaptive_restart_retains_original_clock_and_history():
    origin, quantum = 0.37, 0.1
    options = {
        'curvature_atol': 0.03,
        'timestep_quantum': quantum,
        'timestep_origin': origin,
        'checkpoint_key': 'quadratic-clock',
    }
    first = runtime.integrate_transient(
        QuadraticTrajectory(),
        TimeWindow(origin, origin + 0.4, quantum),
        np.array([origin**2]),
        initial_history=(np.array([(origin - quantum) ** 2]), np.array([origin**2])),
        **options,
    )
    resumed = runtime.integrate_transient(
        QuadraticTrajectory(),
        TimeWindow(origin + 0.4, origin + 0.8, quantum),
        first.states[-1],
        restart=first.checkpoint,
        **options,
    )
    ticks = (np.asarray(resumed.times) - origin) / quantum
    np.testing.assert_allclose(ticks, np.rint(ticks), atol=1e-12)
    for time, state in zip(resumed.times, resumed.states, strict=True):
        assert state[0] == pytest.approx(time**2, abs=1e-12)
    assert resumed.times[-1] == pytest.approx(origin + 0.8)


@pytest.mark.parametrize('dt', [0.1, 0.05])
def test_accepted_quadratic_steps_obey_the_curvature_bar_without_an_lte_claim(dt):
    atol = 0.015
    solution = runtime.integrate_transient(
        QuadraticTrajectory(),
        TimeWindow(0.0, 0.6, dt),
        np.zeros(1),
        initial_history=(np.array([dt * dt]), np.zeros(1)),
        curvature_atol=atol,
        h_min=1e-6,
        h_max=0.5,
    )
    previous_dt = dt
    for previous_time, time, state in zip(
        solution.times[:-1],
        solution.times[1:],
        solution.states[1:],
        strict=True,
    ):
        h = time - previous_time
        # Independent exact predictor discrepancy, including unequal accepted spacings.
        assert h * (h + previous_dt) <= atol * (1.0 + 1e-12)
        assert state[0] == pytest.approx(time * time, abs=1e-12)
        previous_dt = h
    assert solution.times[-1] == pytest.approx(0.6)
    assert solution.converged
