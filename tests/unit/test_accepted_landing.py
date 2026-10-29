"""Accepted lattice commands restart BDF history without off-grid observations."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import get_integrator, integrate_transient
from numerics_lab.time_integrators import march as march_runtime
from numerics_lab.time_integrators._landing import AcceptedProjectionResult
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('kind', [IntegratorKind.TRAPEZOID, IntegratorKind.TR_BDF2])
def test_read_only_landing_observer_preserves_actual_rate_scheme_without_restart(kind):
    problem = _Forced()
    problem.forcing = 1.0
    observed = []

    def observe(time, state):
        assert not state.flags.writeable
        observed.append(time)
        return ()

    solution = integrate_transient(
        problem,
        TimeWindow(0.0, 0.2, 0.1, kind),
        np.zeros(1),
        v0=np.ones(1),
        on_landing=observe,
    )
    assert observed == solution.times[1:]
    assert not solution.history_restarts
    np.testing.assert_allclose(solution.states[-1], [0.2], atol=1e-14)


@pytest.mark.parametrize('integrator', [IntegratorKind.BDF1, IntegratorKind.BDF2])
def test_event_only_owner_projects_actual_right_state_preserving_storage_and_incoming_rate(integrator):
    class Algebraic:
        forcing = 0.0

        def residual(self, u, u_dot, u_ddot, t):
            return np.array([u_dot[0], u[1] - self.forcing])

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.diag([weights.alpha1.real, weights.alpha0.real])

    problem = Algebraic()
    projected = []

    def event(_index, time, _state):
        assert time == pytest.approx(0.15, abs=1e-7)
        problem.forcing = 3.0

    def project(time, state):
        assert not state.flags.writeable
        projected.append(time)
        return np.array([state[0], problem.forcing])

    solution = integrate_transient(
        problem,
        TimeWindow(0.0, 0.3, 0.1, integrator),
        np.array([0.4, 0.0]),
        event_fns=(lambda _state, time: time - 0.15,),
        on_event=event,
        project_accepted_state=project,
    )
    assert len(projected) == 1
    row = int(np.argmin(np.abs(np.asarray(solution.times) - projected[0])))
    np.testing.assert_allclose(solution.states[row], [0.4, 3.0], atol=1e-12)
    np.testing.assert_allclose(solution.rate_state_overrides[row - 1], [0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(solution.states)[:, 0], 0.4, atol=1e-12)
    np.testing.assert_allclose(solution.state_rates, 0.0, atol=1e-12)


@pytest.mark.parametrize('owner', ['none', 'surface_only', 'callback_only'])
def test_projection_still_requires_actual_mutation_owner(owner):
    options = {}
    if owner == 'surface_only':
        options['event_fns'] = (lambda _state, time: time - 0.15,)
    if owner == 'callback_only':
        options['on_event'] = lambda *_args: None
    with pytest.raises(ValueError, match='accepted state projection requires'):
        integrate_transient(
            _Forced(),
            TimeWindow(0.0, 0.3, 0.1, IntegratorKind.BDF1),
            np.zeros(1),
            project_accepted_state=lambda _time, state: state.copy(),
            **options,
        )


def test_unknown_integrator_without_projected_pair_authority_refuses_actual_outgoing_rate(monkeypatch):
    base = get_integrator(IntegratorKind.BDF1)
    scheme = SimpleNamespace(
        name='unknown',
        order=base.order,
        stages=base.stages,
        error_constant=base.error_constant,
        weights=base.weights,
        start=base.start,
        prepare=base.prepare,
        advance=base.advance,
        restart=base.restart,
    )
    monkeypatch.setattr(march_runtime, 'get_integrator', lambda _kind: scheme)
    with pytest.raises(TypeError, match='actual integrator must own explicit projected derivative restart'):
        integrate_transient(
            _Forced(),
            TimeWindow(0.0, 0.3, 0.1),
            np.zeros(1),
            on_accept=lambda time, _state: time == 0.2,
            project_accepted_state=lambda _time, state: AcceptedProjectionResult(state, np.array([3.0])),
        )


def test_trbdf2_actual_outgoing_rate_survives_macro_boundary_checkpoint():
    problem = _Forced()

    def observe(time, _state):
        if time == 0.2:
            problem.forcing = 3.0
            return True
        return False

    options = {
        'on_accept': observe,
        'project_accepted_state': lambda _time, state: AcceptedProjectionResult(state, np.array([problem.forcing])),
        'checkpoint_key': 'actual-trbdf2-pair',
    }
    first = integrate_transient(
        problem,
        TimeWindow(0.0, 0.2, 0.1, IntegratorKind.TR_BDF2),
        np.zeros(1),
        **options,
    )
    saved = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping())))
    np.testing.assert_array_equal(first.final_integrator_state.v, [3.0])
    resumed = integrate_transient(
        problem,
        TimeWindow(0.2, 0.4, 0.1, IntegratorKind.TR_BDF2),
        np.zeros(1),
        restart=saved,
        checkpoint_key='actual-trbdf2-pair',
    )
    np.testing.assert_allclose(np.asarray(resumed.states).ravel(), [0.0, 0.3, 0.6], atol=1e-14)


@pytest.mark.parametrize('kind', [IntegratorKind.NEWMARK, IntegratorKind.GENERALIZED_ALPHA])
def test_second_order_checkpoint_retains_actual_projected_derivative_history(kind):
    class AccelerationLaw:
        acceleration = -3.0

        def residual(self, u, u_dot, u_ddot, t):
            return u_ddot - self.acceleration

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha2.real]])

    problem = AccelerationLaw()

    def changed(_time, _state):
        if _time == 0.1:
            problem.acceleration = 4.0
            return True
        return False

    first = integrate_transient(
        problem,
        TimeWindow(0.0, 0.1, 0.1, kind),
        np.array([0.4]),
        v0=np.array([-2.0]),
        a0=np.array([-3.0]),
        on_accept=changed,
        project_accepted_state=lambda _time, state: AcceptedProjectionResult(state, np.array([2.0]), np.array([4.0])),
        checkpoint_key='actual-second-order-pair',
    )
    saved = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping())))
    assert saved.velocity == (2.0,) and saved.acceleration == (4.0,)
    resumed = integrate_transient(
        problem,
        TimeWindow(0.1, 0.2, 0.1, kind),
        np.zeros(1),
        restart=saved,
        checkpoint_key='actual-second-order-pair',
        max_newton=3,
    )
    np.testing.assert_allclose(resumed.states[-1], [0.405], atol=1e-14)


def _march(eventful, adaptive=False):
    """The ONE march; ``eventful`` routes it through a never-crossing surface, ``adaptive`` picks the
    controller each former entry point meant by the word (curvature on the plain grid, Newton-failure
    halving on the eventful one).
    """

    def run(*args, **kwargs):
        if eventful:
            kwargs.setdefault('event_fns', (lambda _state, _time: -1.0,))
            return integrate_transient(*args, adaptive=adaptive, **kwargs)
        if adaptive:
            kwargs['curvature_atol'] = 1e-6
        return integrate_transient(*args, **kwargs)

    return run


class _Forced:
    forcing = 0.0

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot - self.forcing

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * weights.alpha1.real


def test_checkpoint_after_actual_landing_restores_right_law_and_bootstrap_without_refiring():
    class OwnedForce(_Forced):
        has_history = True

        def commit_history(self, state, time):
            pass

        def checkpoint_history(self):
            return {'forcing': self.forcing}

        def restore_checkpoint_history(self, data):
            self.forcing = data['forcing']

    observed = []

    def run(t0, t1, restart=None):
        owner = OwnedForce()

        def landing(time, _state):
            if time == 0.1:
                observed.append(time)
                owner.forcing = 2.0
                return (0,)
            return ()

        return integrate_transient(
            owner,
            TimeWindow(t0, t1, 0.1, IntegratorKind.BDF2),
            np.zeros(1),
            on_landing=landing,
            checkpoint_key='actual-owned-scheduled-law',
            restart=restart,
        )

    whole = run(0.0, 0.3)
    first = run(0.0, 0.1)
    saved = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping())))
    assert len(saved.states) == 1 and saved.material_history['forcing'] == 2.0
    assert saved.previous_dt == 0.1
    before = tuple(observed)
    resumed = run(saved.time, 0.3, saved)
    assert tuple(observed) == before
    np.testing.assert_allclose(resumed.states, whole.states[1:], atol=1e-14)


def test_incoming_step_uses_owned_endpoint_law_without_perturbing_time():
    times = []

    class OwnedEndpoint(_Forced):
        def step_problem(self, time):
            times.append(time)
            view = _Forced()
            view.forcing = 1.0
            return view

        def residual(self, u, u_dot, u_ddot, t):
            pytest.fail('an incoming solve must use its explicitly owned endpoint view')

    solution = integrate_transient(
        OwnedEndpoint(),
        TimeWindow(t0=0.0, t1=0.2, dt=0.1, integrator=IntegratorKind.BDF1),
        np.zeros(1),
    )
    assert times == [0.1, 0.2]
    assert np.asarray(solution.states).ravel() == pytest.approx([0.0, 0.1, 0.2])


@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('eventful', [False, True])
def test_right_state_projection_preserves_left_rate_and_bootstraps_from_right(adaptive, eventful):
    class Algebraic:
        forcing = 0.0

        def residual(self, u, u_dot, u_ddot, t):
            return np.array([u_dot[0], u[1] - self.forcing])

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.diag([weights.alpha1.real, weights.alpha0.real])

    problem = Algebraic()

    def observe(time, _state):
        if time == 0.1:
            problem.forcing = 3.0
            return True
        return False

    def project(_time, state):
        assert not state.flags.writeable
        return np.array([state[0], problem.forcing])

    solution = _march(eventful, adaptive)(
        problem,
        TimeWindow(t0=0.0, t1=0.2, dt=0.1, integrator=IntegratorKind.BDF2),
        np.zeros(2),
        on_accept=observe,
        project_accepted_state=project,
    )
    assert solution.times == pytest.approx([0.0, 0.1, 0.2])
    assert solution.states[1] == pytest.approx([0.0, 3.0])
    assert solution.rate_state_overrides[0] == pytest.approx([0.0, 0.0])
    assert np.asarray(solution.state_rates) == pytest.approx(np.zeros((2, 2)))


@pytest.mark.parametrize('adaptive', [False, True])
@pytest.mark.parametrize('eventful', [False, True])
def test_live_breakpoint_created_on_accept_is_not_skipped(adaptive, eventful):
    deadlines = []
    observed = []

    def observe(time, _state):
        observed.append(time)
        if time == 0.0:
            deadlines.append(0.005)
        elif time == 0.005:
            deadlines.append(0.007)
        return False

    solution = _march(eventful, adaptive)(
        _Forced(),
        TimeWindow(t0=0.0, t1=0.1, dt=0.1, integrator=IntegratorKind.BDF2),
        np.zeros(1),
        on_accept=observe,
        next_breakpoint=lambda time: next((value for value in deadlines if value > time), None),
    )
    assert 0.005 in solution.times
    assert 0.007 in solution.times
    assert observed == solution.times
    assert solution.times[-1] == 0.1


@pytest.mark.parametrize('deadline', [True, float('nan'), float('inf'), 0.0, -1.0])
def test_live_breakpoint_rejects_nonfuture_or_invalid_owner_time(deadline):
    with pytest.raises(ValueError, match='strictly future'):
        integrate_transient(
            _Forced(),
            TimeWindow(t0=0.0, t1=0.1, dt=0.1, integrator=IntegratorKind.BDF1),
            np.zeros(1),
            next_breakpoint=lambda _time: deadline,
        )


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
        TimeWindow(t0=0.0, t1=0.4, dt=0.1, integrator=IntegratorKind.BDF2),
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
            TimeWindow(t0=0.0, t1=0.1, dt=0.1, integrator=IntegratorKind.BDF1),
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

    study = TimeWindow(t0=0.0, t1=0.4, dt=0.1, integrator=IntegratorKind.BDF2)
    if eventful:
        solution = integrate_transient(
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

    solution = _march(eventful)(
        problem,
        TimeWindow(t0=0.0, t1=0.4, dt=0.1, integrator=IntegratorKind.BDF2),
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

    options = {'checkpoint_key': 'forcing-transition'}
    if adaptive:
        options |= {'curvature_atol': 1e-6, 'h_min': 0.1, 'h_max': 0.1}
    first = integrate_transient(
        problem,
        TimeWindow(t0=0.0, t1=0.2, dt=0.1, integrator=IntegratorKind.BDF2),
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
        TimeWindow(t0=0.2, t1=0.4, dt=0.1, integrator=IntegratorKind.BDF2),
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

    solution = integrate_transient(
        problem,
        TimeWindow(t0=0.0, t1=0.2, dt=0.2, integrator=IntegratorKind.BDF2),
        np.zeros(1),
        refine=0.01,
        on_accept=observe,
    )
    assert observed == solution.times
    assert solution.times == pytest.approx([0.0, 0.1, 0.2])
    assert np.asarray(solution.states).ravel() == pytest.approx([0.0, 0.001, 0.305])
    assert solution.accepted_discontinuities == [pytest.approx(0.1)]
