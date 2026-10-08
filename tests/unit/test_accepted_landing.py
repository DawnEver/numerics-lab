"""Accepted lattice commands restart BDF history without off-grid observations."""

import json

import numpy as np
import pytest

from motronics.contracts.spec.study_spec import IntegratorKind, TransientStudy
from motronics.gauss.time_integrators import integrate_transient
from motronics.gauss.time_integrators.checkpoint import TransientCheckpoint
from motronics.gauss.time_integrators.events import integrate_transient_with_events

pytestmark = pytest.mark.unit


class _Forced:
    forcing = 0.0

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot - self.forcing

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * weights.alpha1.real


def test_landing_dispatches_only_committed_states_and_restarts_history():
    problem = _Forced()
    observed = []

    def landing(time, state):
        observed.append((time, state.copy()))
        assert not state.flags.writeable
        if time == pytest.approx(0.2):
            problem.forcing = 3.0
            return (7,)
        return ()

    solution = integrate_transient(
        problem,
        TransientStudy(t0=0.0, t1=0.4, dt=0.1, integrator=IntegratorKind.BDF2),
        np.zeros(1),
        on_landing=landing,
    )
    assert [time for time, _ in observed] == solution.times[1:]
    assert solution.landed_events == [(7, pytest.approx(0.2))]
    # BE restart gives .3; retaining pre-command BDF2 history would incorrectly give .2.
    assert np.array(solution.states).ravel() == pytest.approx([0.0, 0.0, 0.0, 0.3, 0.6])
    assert np.array(solution.state_rates).ravel() == pytest.approx([0.0, 0.0, 3.0, 3.0])


@pytest.mark.parametrize('indices', [(True,), (-1,), (0, 0), (1.5,)])
def test_invalid_landing_identifiers_are_not_published(indices):
    with pytest.raises(ValueError, match='event identifiers'):
        integrate_transient(
            _Forced(),
            TransientStudy(t0=0.0, t1=0.1, dt=0.1, integrator=IntegratorKind.BDF1),
            np.zeros(1),
            on_landing=lambda _time, _state: indices,
        )


@pytest.mark.parametrize('eventful', [False, True])
def test_accepted_observer_gets_initial_and_canonical_states_not_event_probes(eventful):
    problem = _Forced()
    observed = []

    def observe(time, state):
        assert not state.flags.writeable
        observed.append(time)
        return False

    study = TransientStudy(t0=0.0, t1=0.4, dt=0.1, integrator=IntegratorKind.BDF2)
    if eventful:
        solution = integrate_transient_with_events(
            problem,
            study,
            np.zeros(1),
            event_fns=(lambda _state, time: time - 0.15,),
            on_event=lambda _index, _time, _state: setattr(problem, 'forcing', 3.0),
            on_accept=observe,
        )
        assert solution.events[0][1] == pytest.approx(0.15, abs=1e-7)
    else:
        solution = integrate_transient(problem, study, np.zeros(1), on_accept=observe)
    assert observed == solution.times


@pytest.mark.parametrize('eventful', [False, True])
def test_accepted_law_change_restarts_its_actual_history(eventful):
    problem = _Forced()

    def observe(time, _state):
        if time == pytest.approx(0.2):
            problem.forcing = 3.0
            return True
        return False

    driver = integrate_transient_with_events if eventful else integrate_transient
    solution = driver(
        problem,
        TransientStudy(t0=0.0, t1=0.4, dt=0.1, integrator=IntegratorKind.BDF2),
        np.zeros(1),
        on_accept=observe,
    )
    assert np.asarray(solution.states).ravel() == pytest.approx([0.0, 0.0, 0.0, 0.3, 0.6])
    assert solution.accepted_discontinuities == [pytest.approx(0.2)]


@pytest.mark.parametrize('adaptive', [False, True])
def test_law_change_checkpoint_retains_real_step_spacing_and_bootstraps_after_resume(adaptive):
    problem = _Forced()
    observed = []

    def observe(time, state):
        assert not state.flags.writeable
        observed.append(time)
        if time == pytest.approx(0.2):
            problem.forcing = 3.0
            return True
        return False

    options = {'adaptive': adaptive, 'h_min': 0.1, 'h_max': 0.1, 'checkpoint_key': 'forcing-transition'}
    first = integrate_transient(
        problem,
        TransientStudy(t0=0.0, t1=0.2, dt=0.1, integrator=IntegratorKind.BDF2),
        np.zeros(1),
        on_accept=observe,
        **options,
    )
    assert observed == first.times
    assert first.accepted_discontinuities == [pytest.approx(0.2)]
    saved = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping())))
    assert saved.previous_dt == pytest.approx(0.1)
    assert len(saved.states) == 1
    resumed = integrate_transient(
        problem,
        TransientStudy(t0=0.2, t1=0.4, dt=0.1, integrator=IntegratorKind.BDF2),
        np.zeros(1),
        restart=saved,
        **options,
    )
    assert np.asarray(resumed.states).ravel() == pytest.approx([0.0, 0.3, 0.6])


def test_refined_midpoint_observation_invalidates_the_speculative_second_half():
    class Curved(_Forced):
        def residual(self, u, u_dot, u_ddot, t):
            return u_dot - t * t - self.forcing

    problem = Curved()
    observed = []

    def observe(time, state):
        assert not state.flags.writeable
        observed.append(time)
        if time == pytest.approx(0.1):
            problem.forcing = 3.0
            return True
        return False

    solution = integrate_transient_with_events(
        problem,
        TransientStudy(t0=0.0, t1=0.2, dt=0.2, integrator=IntegratorKind.BDF2),
        np.zeros(1),
        refine=0.01,
        on_accept=observe,
    )
    assert observed == solution.times
    assert solution.times == pytest.approx([0.0, 0.1, 0.2])
    assert np.asarray(solution.states).ravel() == pytest.approx([0.0, 0.001, 0.305])
    assert solution.accepted_discontinuities == [pytest.approx(0.1)]
