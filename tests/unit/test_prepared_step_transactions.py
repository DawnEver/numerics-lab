"""Prepared numerical stages select explicit candidates, never the last Newton probe."""

import json
from dataclasses import dataclass, replace

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import (
    BDF1Integrator,
    GeneralizedAlphaIntegrator,
    StepNotConverged,
    StepTolerance,
    get_integrator,
    integrate_transient,
)
from numerics_lab.time_integrators._forms import IntegratorState
from numerics_lab.time_integrators._landing import AcceptedProjectionResult, accepted_projection
from numerics_lab.time_integrators._step import take_step
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint, _checkpoint_of
from numerics_lab.time_integrators.prepared import restart_prepared


@dataclass(frozen=True)
class Token:
    number: int


def test_accepted_projection_result_owns_explicit_state_and_rate():
    state, rate = np.array([0.2]), np.array([3.0])
    token = Token(4)
    result = AcceptedProjectionResult(state, rate, prepared_restart=token)
    state[:] = 9.0
    rate[:] = 8.0
    selected = accepted_projection(lambda _time, _state: result, 0.3, np.array([0.1]))
    np.testing.assert_array_equal(selected.state, [0.2])
    np.testing.assert_array_equal(selected.rate, [3.0])
    assert selected.prepared_restart is token
    assert not selected.state.flags.writeable and not selected.rate.flags.writeable


@pytest.mark.parametrize('kind', [IntegratorKind.BDF1, IntegratorKind.TRAPEZOID, IntegratorKind.TR_BDF2])
def test_projected_pair_restart_uses_actual_scheme_derivative_history(kind):
    integrator = get_integrator(kind)
    before = integrator.start(np.array([0.2]), 0.0, np.array([-1.0]))
    result = AcceptedProjectionResult(np.array([0.3]), np.array([4.0]))
    restarted = restart_prepared(object(), integrator, before, result, 0.1)
    np.testing.assert_array_equal(restarted.last_u, [0.3])
    if kind == IntegratorKind.BDF1:
        assert restarted.v is None
    else:
        np.testing.assert_array_equal(restarted.v, [4.0])
    assert restarted.h is None and restarted.stage == 0


class PreparedOwner:
    """Analytically eliminate x from x' + x = u, retaining its real scheme history."""

    has_history = True

    def __init__(self) -> None:
        self.inner = None
        self.snapshots = {}
        self.attempts = []
        self.accepted = []
        self.rejected = []
        self.material_commits = []
        self.restarts = []
        self.level = 1.0

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + u - self.level

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * (weights.alpha0.real + weights.alpha1.real)

    def begin_step(self, context):
        if context.parent_candidate is not None:
            inner = self.snapshots[context.parent_candidate].copy()
        elif self.inner is None:
            inner = context.integrator.start(np.zeros(1), context.state.t, np.zeros(1), np.ones(1))
        else:
            inner = self.inner.copy()
        attempt = Attempt(self, context, inner)
        self.attempts.append(attempt)
        return attempt

    def accept_step(self, token):
        assert token not in self.accepted and token not in self.rejected
        self.inner = self.snapshots[token].copy()
        self.accepted.append(token)

    def reject_step(self, token):
        assert token not in self.accepted and token not in self.rejected
        self.rejected.append(token)

    def restart_step(self, context):
        if self.inner is not None:
            before = self.inner.last_u.copy()
            self.inner = context.integrator.restart(self.inner, self.inner.last_u, context.state.t)
            np.testing.assert_array_equal(self.inner.last_u, before)
            self.restarts.append(self.inner.copy())

    def checkpoint_history(self):
        state = self.inner
        return {
            'us': [u.tolist() for u in state.us],
            't': state.t,
            'h': state.h,
            'stage': state.stage,
            'v': None if state.v is None else state.v.tolist(),
            'a': None if state.a is None else state.a.tolist(),
            'level': self.level,
            'material_commits': self.material_commits.copy(),
        }

    def restore_checkpoint_history(self, data):
        self.inner = IntegratorState(
            us=[np.array(u) for u in data['us']],
            t=data['t'],
            h=data['h'],
            stage=int(data['stage']),
            v=None if data['v'] is None else np.array(data['v']),
            a=None if data['a'] is None else np.array(data['a']),
        )
        self.level = data['level']
        self.material_commits = data['material_commits'].copy()

    def commit_history(self, u, t):
        self.material_commits.append(t)


class Attempt:
    def __init__(self, owner, context, inner) -> None:
        self.owner, self.context, self.inner = owner, context, inner
        self.forms = []
        self.selected = []
        self.aborted = False

    def prepare_stage(self, context):
        self.stage = context
        self.form = self.context.integrator.prepare(self.inner, self.context.dt, self.context.t_endpoint)
        self.forms.append(context.form)
        return self

    def field(self, u):
        form = self.form
        forcing = u
        offset = form.u_dot(np.zeros(1)) + form.u_arg(np.zeros(1))
        return (forcing - offset) / (form.weights.alpha0.real + form.weights.alpha1.real)

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + u + self.form.u_arg(self.field(u)) - self.owner.level

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        inner_weights = self.form.weights
        tangent = weights.alpha1.real + weights.alpha0.real
        tangent += (
            inner_weights.alpha0.real * weights.alpha0.real / (inner_weights.alpha0.real + inner_weights.alpha1.real)
        )
        return np.array([[tangent]])

    def stage_solved(self, u):
        self.selected.append(np.array(u, copy=True))
        self.inner = self.context.integrator.advance(
            self.inner, self.field(self.stage.form.u_arg(u)), self.context.dt, self.context.t_endpoint, self.form
        )

    def finish(self):
        token = Token(len(self.owner.snapshots))
        self.owner.snapshots[token] = self.inner.copy()
        return token

    def abort(self):
        self.aborted = True


class FullProblem:
    def __init__(self) -> None:
        self.level = 1.0

    def residual(self, u, u_dot, u_ddot, t):
        return np.array([u_dot[0] + u[0] + u[1] - self.level, u_dot[1] + u[1] - u[0]])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        a, b = weights.alpha0.real, weights.alpha1.real
        return np.array([[a + b, a], [-a, a + b]])


@pytest.mark.parametrize('kind', [IntegratorKind.BDF1, IntegratorKind.TR_BDF2])
def test_prepared_cut_retains_actual_stage_clock_and_explicit_endpoint_side(kind):
    contexts = []

    class ClockAttempt(Attempt):
        def prepare_stage(self, context):
            contexts.append(context)
            return super().prepare_stage(context)

    class ClockOwner(PreparedOwner):
        def begin_step(self, context):
            inner = context.integrator.start(np.zeros(1), context.state.t, np.zeros(1))
            return ClockAttempt(self, context, inner)

    integrator = get_integrator(kind)
    take_step(
        ClockOwner(),
        integrator,
        integrator.start(np.zeros(1), 0.0, np.ones(1)),
        0.1,
        0.1,
        endpoint_left=True,
        max_newton=3,
        solve=np.linalg.solve,
        tol=1e-12,
    )
    assert contexts[-1].form.t_eval == 0.1
    assert contexts[-1].endpoint_left is True
    if kind == IntegratorKind.TR_BDF2:
        assert 0.0 < contexts[0].form.t_eval < 0.1
        assert contexts[0].endpoint_left is False


@pytest.mark.parametrize('kind', [IntegratorKind.BDF1, IntegratorKind.BDF2, IntegratorKind.TR_BDF2])
def test_prepared_stages_match_independent_full_system(kind):
    owner = PreparedOwner()
    study = TimeWindow(0.0, 0.7, 0.1, kind)
    kwargs = {'max_newton': 3, 'step_times': [0.0, 0.1, 0.23, 0.3, 0.55, 0.7]}
    actual = integrate_transient(owner, study, np.zeros(1), v0=np.ones(1), **kwargs)
    expected = integrate_transient(FullProblem(), study, np.zeros(2), v0=np.array([1.0, 0.0]), **kwargs)
    assert owner.attempts, 'prepared stage authority was never called'
    np.testing.assert_allclose(np.array(actual.states)[:, 0], np.array(expected.states)[:, 0], atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(owner.inner.last_u, expected.states[-1][1:], atol=1e-13, rtol=1e-13)
    assert len(owner.accepted) == len(actual.times) - 1
    assert owner.material_commits == actual.times[1:]
    assert all(len(attempt.selected) == get_integrator(kind).stages for attempt in owner.attempts)


def test_copied_numerical_state_carries_explicit_speculative_parent():
    owner = PreparedOwner()
    integrator = get_integrator(IntegratorKind.BDF2)
    solve_kw = {'max_newton': 3, 'solve': np.linalg.solve, 'tol': 1e-12}
    first = take_step(owner, integrator, integrator.start(np.zeros(1), 0.0), 0.1, 0.1, **solve_kw)
    assert owner.attempts, 'prepared attempt authority was never called'
    copied = first.state.copy()
    assert copied.prepared_candidate is first.candidate
    second = take_step(owner, integrator, copied, 0.17, 0.27, **solve_kw)
    assert owner.attempts[-1].context.parent_candidate is first.candidate
    assert second.candidate != first.candidate
    assert not owner.accepted and not owner.material_commits


@pytest.mark.parametrize('kind', [IntegratorKind.BDF2, IntegratorKind.TR_BDF2])
def test_checkpoint_restores_actual_hidden_history_not_an_opaque_token(kind):
    owner = PreparedOwner()
    study = TimeWindow(0.0, 0.8, 0.1, kind)
    whole = integrate_transient(owner, study, np.zeros(1), v0=np.ones(1), max_newton=3)
    first_owner = PreparedOwner()
    first = integrate_transient(
        first_owner,
        TimeWindow(0.0, 0.4, 0.1, kind),
        np.zeros(1),
        v0=np.ones(1),
        max_newton=3,
        checkpoint_key='prepared-inner',
    )
    checkpoint = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping())))
    assert first_owner.inner.h == pytest.approx(0.1)
    assert first_owner.inner.stage == 0
    resumed_owner = PreparedOwner()
    resumed = integrate_transient(
        resumed_owner,
        TimeWindow(0.4, 0.8, 0.1, kind),
        np.zeros(1),
        restart=checkpoint,
        checkpoint_key='prepared-inner',
        max_newton=3,
    )
    assert resumed_owner.attempts[0].context.parent_candidate is None
    np.testing.assert_allclose(resumed.states[-1], whole.states[-1], atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(resumed_owner.inner.last_u, owner.inner.last_u, atol=1e-13, rtol=1e-13)
    assert resumed_owner.inner.h == pytest.approx(owner.inner.h)
    assert resumed_owner.inner.stage == owner.inner.stage
    for actual, expected in zip(resumed_owner.inner.us, owner.inner.us, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
    if owner.inner.v is not None:
        np.testing.assert_allclose(resumed_owner.inner.v, owner.inner.v, atol=1e-13, rtol=1e-13)
    assert resumed_owner.material_commits == owner.material_commits


def test_event_probes_never_commit_and_real_restart_preserves_hidden_state():
    owner, full = PreparedOwner(), FullProblem()
    study = TimeWindow(0.0, 0.8, 0.2, IntegratorKind.BDF2)

    def event(_u, t):
        return t - 0.31

    actual = integrate_transient(
        owner,
        study,
        np.zeros(1),
        max_newton=3,
        event_fns=(event,),
        on_event=lambda _i, _t, _u: setattr(owner, 'level', -0.5),
    )
    expected = integrate_transient(
        full,
        study,
        np.zeros(2),
        max_newton=3,
        event_fns=(event,),
        on_event=lambda _i, _t, _u: setattr(full, 'level', -0.5),
    )
    assert owner.rejected and len(owner.attempts) > len(owner.accepted)
    assert len(owner.accepted) == len(actual.times) - 1
    assert owner.material_commits == actual.times[1:]
    assert len(owner.restarts) == 1
    assert len(owner.restarts[0].us) == 1 and owner.restarts[0].h is None
    assert np.linalg.norm(owner.restarts[0].last_u) > 0.01
    np.testing.assert_allclose(actual.states[-1], expected.states[-1][:1], atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(owner.inner.last_u, expected.states[-1][1:], atol=1e-13, rtol=1e-13)


def test_refinement_accepts_earlier_midpoint_after_later_speculative_probe():
    owner = PreparedOwner()
    actual = integrate_transient(
        owner,
        TimeWindow(0.0, 0.4, 0.2),
        np.zeros(1),
        max_newton=3,
        refine=1e-8,
        h_min=0.05,
    )
    assert owner.rejected
    first = owner.accepted[0]
    assert owner.snapshots[first].t == pytest.approx(0.1)
    first_position = next(i for i, attempt in enumerate(owner.attempts) if attempt.inner.t == 0.1)
    assert owner.attempts[first_position + 1].context.parent_candidate is first
    assert owner.snapshots[owner.rejected[0]].t == pytest.approx(0.2)
    assert len(owner.accepted) == len(actual.times) - 1
    assert owner.material_commits == actual.times[1:]


def test_second_tr_stage_exception_discards_first_stage_without_a_commit():
    class FailingOwner(PreparedOwner):
        def begin_step(self, context):
            attempt = super().begin_step(context)
            original = attempt.prepare_stage

            def prepare(context):
                if context.index == 1:
                    raise ArithmeticError
                return original(context)

            attempt.prepare_stage = prepare
            return attempt

    owner = FailingOwner()
    with pytest.raises(ArithmeticError):
        integrate_transient(owner, TimeWindow(0.0, 0.2, 0.2, IntegratorKind.TR_BDF2), np.zeros(1), max_newton=3)
    assert owner.attempts[0].aborted and len(owner.attempts[0].selected) == 1
    assert owner.inner is None and not owner.accepted and not owner.material_commits


def test_nonconverged_macro_candidate_is_rejected_when_march_raises():
    class NoRoot:
        def residual(self, u, u_dot, u_ddot, t):
            return np.ones(1)

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.eye(1)

    class FailingOwner(PreparedOwner):
        def begin_step(self, context):
            attempt = super().begin_step(context)
            original = attempt.prepare_stage

            def prepare(context):
                original(context)
                return NoRoot()

            attempt.prepare_stage = prepare
            return attempt

    owner = FailingOwner()
    with pytest.raises(StepNotConverged):
        integrate_transient(owner, TimeWindow(0.0, 0.2, 0.2), np.zeros(1), max_newton=2)
    assert len(owner.rejected) == 1 and not owner.accepted and not owner.material_commits
    assert owner.inner is None


def test_intermediate_tr_stage_is_not_a_portable_macrostep_checkpoint():
    integrator = get_integrator(IntegratorKind.TR_BDF2)
    state = integrator.start(np.zeros(1), 0.0, np.ones(1))
    form = integrator.prepare(state, 0.2, 0.2)
    middle = integrator.advance(state, np.array([0.1]), 0.2, 0.2, form)
    assert middle.stage == 1
    with pytest.raises(ValueError, match='accepted macrostep'):
        _checkpoint_of(FullProblem(), middle, 0.2, 'middle', integrator.name)


def test_shifted_form_is_passed_exactly_not_reconstructed_from_dt():
    class ShiftedIntegrator(BDF1Integrator):
        def prepare(self, state, dt, t_new):
            return replace(super().prepare(state, dt, t_new), t_eval=t_new - 0.3 * dt)

    owner = PreparedOwner()
    integrator = ShiftedIntegrator()
    out = take_step(
        owner,
        integrator,
        integrator.start(np.zeros(1), 0.0),
        0.2,
        0.2,
        max_newton=3,
        solve=np.linalg.solve,
        tol=1e-12,
    )
    form = owner.attempts[0].forms[0]
    assert form.t_eval == pytest.approx(0.14)
    assert out.residual_time == form.t_eval
    assert owner.attempts[0].context.t_endpoint == 0.2
    assert owner.attempts[0].context.dt == 0.2


def test_checkpoint_refuses_prepared_owner_without_real_committed_history():
    class StatelessPrepared:
        begin_step = PreparedOwner.begin_step
        accept_step = PreparedOwner.accept_step
        reject_step = PreparedOwner.reject_step
        restart_step = PreparedOwner.restart_step
        residual = PreparedOwner.residual
        jacobian = PreparedOwner.jacobian

    with pytest.raises(ValueError, match='committed hidden-history owner'):
        integrate_transient(
            StatelessPrepared(),
            TimeWindow(0.0, 0.2, 0.2),
            np.zeros(1),
            checkpoint_key='missing',
        )


def test_bootstrap_halfstep_remarches_after_actual_midpoint_material_commit():
    class ChangingOwner(PreparedOwner):
        def commit_history(self, u, t):
            super().commit_history(u, t)
            if len(self.material_commits) == 1:
                self.level = -0.5

    owner = ChangingOwner()
    actual = integrate_transient(
        owner,
        TimeWindow(0.0, 0.2, 0.2, IntegratorKind.BDF1),
        np.zeros(1),
        max_newton=3,
        step_control=StepTolerance(atol=1.0, rtol=1.0),
    )
    full = FullProblem()
    integrator = BDF1Integrator()
    kwargs = {'solve': np.linalg.solve, 'max_newton': 3, 'tol': 1e-12}
    first = take_step(full, integrator, integrator.start(np.zeros(2), 0.0), 0.1, 0.1, **kwargs)
    full.level = -0.5
    expected = take_step(full, integrator, first.state, 0.1, 0.2, **kwargs)
    assert actual.times == [0.0, 0.1, 0.2]
    assert len(owner.accepted) == 2 and len(owner.rejected) >= 2
    assert owner.attempts[2].context.parent_candidate is owner.accepted[0]
    assert owner.attempts[-1].context.parent_candidate is owner.accepted[0]
    np.testing.assert_allclose(actual.states[-1], expected.u[:1], atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(owner.inner.last_u, expected.u[1:], atol=1e-13, rtol=1e-13)


def test_scheduled_source_cut_restarts_actual_hidden_history_at_conserved_state():
    owner, full = PreparedOwner(), FullProblem()
    study = TimeWindow(0.0, 0.6, 0.2, IntegratorKind.BDF2)
    common = {'time_breakpoints': (0.31,), 'max_newton': 3}
    actual = integrate_transient(
        owner,
        study,
        np.zeros(1),
        on_breakpoint=lambda _t, _u: setattr(owner, 'level', -0.5),
        **common,
    )
    expected = integrate_transient(
        full,
        study,
        np.zeros(2),
        on_breakpoint=lambda _t, _u: setattr(full, 'level', -0.5),
        **common,
    )
    assert len(owner.restarts) == 1 and owner.restarts[0].t == 0.31
    assert len(owner.restarts[0].us) == 1 and owner.restarts[0].h is None
    assert np.linalg.norm(owner.restarts[0].last_u) > 0.01
    assert owner.attempts[1].forms[0].t_eval == 0.31
    assert owner.attempts[1].stage.endpoint_left is True
    assert owner.attempts[2].context.parent_candidate is None
    np.testing.assert_allclose(actual.states[-1], expected.states[-1][:1], atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(owner.inner.last_u, expected.states[-1][1:], atol=1e-13, rtol=1e-13)


def test_generalized_alpha_retains_actual_stage_and_endpoint_coordinates():
    owner, full = PreparedOwner(), FullProblem()
    integrator = GeneralizedAlphaIntegrator(rho_inf=0.4)
    actual_state = integrator.start(np.zeros(1), 0.0, np.ones(1), -np.ones(1))
    full_state = integrator.start(np.zeros(2), 0.0, np.array([1.0, 0.0]), np.array([-1.0, 1.0]))
    kwargs = {'solve': np.linalg.solve, 'max_newton': 3, 'tol': 1e-12}
    for dt, endpoint in ((0.1, 0.1), (0.17, 0.27), (0.03, 0.3)):
        actual = take_step(owner, integrator, actual_state, dt, endpoint, **kwargs)
        expected = take_step(full, integrator, full_state, dt, endpoint, **kwargs)
        form = owner.attempts[-1].forms[0]
        assert form.weights.alpha0.real != 1.0 and form.t_eval != endpoint
        assert actual.residual_time == form.t_eval and actual.state.t == endpoint
        np.testing.assert_allclose(actual.u, expected.u[:1], atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(owner.snapshots[actual.candidate].last_u, expected.u[1:], atol=1e-13, rtol=1e-13)
        actual_state, full_state = actual.state.copy(), expected.state.copy()
    assert not owner.accepted and not owner.material_commits


@pytest.mark.parametrize('reuse', ['parent', 'older_accepted', 'pending'])
def test_reused_candidate_handles_refuse_and_abort_without_alias_commits(reuse):
    class ReusingOwner(PreparedOwner):
        def begin_step(self, context):
            attempt = super().begin_step(context)
            original = attempt.finish

            def finish():
                fresh = original()
                if reuse == 'parent' and context.parent_candidate is not None:
                    return context.parent_candidate
                if reuse == 'older_accepted' and len(self.accepted) >= 2:
                    return self.accepted[0]
                if reuse == 'pending' and len(self.attempts) >= 2:
                    return next(iter(self.snapshots))
                return fresh

            attempt.finish = finish
            return attempt

    owner = ReusingOwner()
    kwargs = {'refine': 1e-8, 'h_min': 0.05} if reuse == 'pending' else {}
    with pytest.raises(ValueError, match='fresh.*candidate'):
        integrate_transient(owner, TimeWindow(0.0, 0.6, 0.2), np.zeros(1), max_newton=3, **kwargs)
    assert owner.attempts[-1].aborted
    assert len(owner.accepted) == len(owner.material_commits)
    assert not any(token in owner.rejected for token in owner.accepted)


@pytest.mark.parametrize('kind', [IntegratorKind.BDF1, IntegratorKind.BDF2])
def test_uniform_eec_owns_actual_canonical_endpoint_and_full_step_spacing(kind):
    class UniformClock:
        def __init__(self) -> None:
            self.slopes = []

        def residual(self, u, u_dot, u_ddot, t):
            return u_dot - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            self.slopes.append((t, weights.alpha1.real))
            return np.array([[weights.alpha1.real]])

    clock = UniformClock()
    dt = 0.05
    endpoint = 7 * dt
    result = integrate_transient(
        clock,
        TimeWindow(0.0, 0.35, dt, kind),
        np.zeros(1),
        max_newton=3,
        eec_period=1.0,
        eec_affine_first_order=True,
        checkpoint_key='uniform-eec',
    )
    assert result.times[-1] == endpoint and result.checkpoint.time == endpoint
    # The record carries the INTEGRATOR'S step (`state.h`, what BDF2 forms its ratio from), not the clock
    # difference `endpoint - 6 dt`, which sits ulps off it and resumed other coefficients (2026-10-13).
    assert result.checkpoint.previous_dt == result.final_integrator_state.h == dt
    assert result.checkpoint.previous_dt == pytest.approx(endpoint - 6 * dt, rel=1e-12, abs=0.0)
    for t, slope in clock.slopes:
        assert t in [tick * dt for tick in range(1, 8)]
        expected = 1 / dt if kind is IntegratorKind.BDF1 or t == dt else 1.5 / dt
        assert slope == expected
    normal = integrate_transient(UniformClock(), TimeWindow(0.0, 0.35, dt, kind), np.zeros(1), max_newton=3)
    assert normal.times[-1] == 0.35 and normal.times[-1] != endpoint
    offgrid = endpoint
    for _ in range(5):
        offgrid = np.nextafter(offgrid, np.inf)
    with pytest.raises(ValueError, match='uniform'):
        integrate_transient(
            UniformClock(),
            TimeWindow(0.0, offgrid, dt, kind),
            np.zeros(1),
            eec_period=1.0,
            eec_affine_first_order=True,
        )


def test_source_breakpoint_retains_explicit_outgoing_rate():
    class Ramp:
        def residual(self, u, u_dot, u_ddot, t):
            return u_dot - (1.0 if t < 0.1 else 4.0 + t - 0.1)

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    result = integrate_transient(
        Ramp(),
        TimeWindow(0.0, 0.2, 0.1, IntegratorKind.TRAPEZOID),
        np.zeros(1),
        v0=np.ones(1),
        time_breakpoints=(0.1,),
        on_breakpoint=lambda _t, u: AcceptedProjectionResult(u, np.array([4.0])),
        max_newton=3,
    )
    np.testing.assert_allclose(result.states[-1], [0.505], atol=1e-14, rtol=0)


def test_prepared_restart_handle_requires_actual_prepared_owner():
    integrator = get_integrator(IntegratorKind.BDF1)
    state = integrator.start(np.zeros(1), 0.0)
    with pytest.raises(TypeError, match='prepared restart handle.*owner'):
        restart_prepared(
            object(), integrator, state, AcceptedProjectionResult(np.zeros(1), prepared_restart=object()), 0.0
        )


@pytest.mark.parametrize('outgoing', [False, True])
def test_event_without_outgoing_rate_uses_actual_tr_startup_not_incoming_slope(outgoing):
    class SwitchedRamp:
        switched = False

        def residual(self, u, u_dot, u_ddot, t):
            return u_dot - (4.0 + t - 0.1 if self.switched else 1.0)

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    problem = SwitchedRamp()

    def switch(_index, _time, _state):
        problem.switched = True

    result = integrate_transient(
        problem,
        TimeWindow(0.0, 0.2, 0.1, IntegratorKind.TRAPEZOID),
        np.zeros(1),
        v0=np.ones(1),
        event_fns=(lambda _u, t: t - 0.1,),
        on_event=switch,
        max_newton=3,
        project_accepted_state=(lambda t, u: AcceptedProjectionResult(u, np.array([4.0 + t - 0.1])))
        if outgoing
        else None,
    )
    np.testing.assert_allclose(result.states[-1], [0.505 if outgoing else 0.51], atol=1e-8, rtol=0)


def test_event_projection_restarts_actual_hidden_owner_once():
    owner = PreparedOwner()
    integrate_transient(
        owner,
        TimeWindow(0.0, 0.5, 0.2),
        np.zeros(1),
        max_newton=3,
        event_fns=(lambda _u, t: t - 0.31,),
        on_event=lambda _index, _time, _state: None,
        project_accepted_state=lambda _t, u: AcceptedProjectionResult(u),
    )
    assert len(owner.restarts) == 1


@pytest.mark.parametrize('kind', [IntegratorKind.NEWMARK, IntegratorKind.GENERALIZED_ALPHA])
@pytest.mark.parametrize('missing', ['rate', 'acceleration', 'neither'])
def test_second_order_restart_requires_and_owns_actual_outgoing_pair(kind, missing):
    integrator = get_integrator(kind)
    before = integrator.start(np.array([0.2]), 0.0, np.array([-2.0]), np.array([-3.0]))
    rate = None if missing == 'rate' else np.array([2.0])
    acceleration = None if missing == 'acceleration' else np.array([4.0])
    projection = AcceptedProjectionResult(np.array([0.4]), rate, acceleration)
    if missing != 'neither':
        with pytest.raises(ValueError, match='actual outgoing rate and acceleration'):
            restart_prepared(object(), integrator, before, projection, 0.1)
        np.testing.assert_array_equal(before.v, [-2.0])
        np.testing.assert_array_equal(before.a, [-3.0])
    else:
        restarted = restart_prepared(object(), integrator, before, projection, 0.1)
        np.testing.assert_array_equal(restarted.v, [2.0])
        np.testing.assert_array_equal(restarted.a, [4.0])
        assert restarted.v.flags.writeable and restarted.a.flags.writeable
        assert not np.shares_memory(restarted.v, projection.rate)
        assert not np.shares_memory(restarted.a, projection.acceleration)
