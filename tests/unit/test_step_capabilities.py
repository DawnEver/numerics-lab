"""Optional problem protocols are resolved once for a numerical run."""

import numpy as np
import pytest

import numerics_lab.time_integrators._step as step
from numerics_lab.time_integrators._integrators import INTEGRATOR_REGISTRY, TrapezoidIntegrator
from numerics_lab.time_integrators._step import ProblemCapabilities, take_step


class LinearProblem:
    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + u

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * (weights.alpha0.real + weights.alpha1.real)


def test_explicit_run_capabilities_do_not_repeat_protocol_inspection(monkeypatch):
    problem = LinearProblem()
    capabilities = ProblemCapabilities.resolve(problem)

    def forbidden(*_args):
        msg = 'protocol inspection inside a resolved run'
        raise AssertionError(msg)

    monkeypatch.setattr(step, 'isinstance', forbidden, raising=False)
    integrator = TrapezoidIntegrator()
    state = integrator.start(np.ones(1), 0.0, v0=-np.ones(1))
    outcome = take_step(
        problem,
        integrator,
        state,
        0.1,
        0.1,
        capabilities=capabilities,
        solve=np.linalg.solve,
        max_newton=1,
        tol=1e-14,
    )
    np.testing.assert_allclose(outcome.u, [19 / 21], rtol=1e-14, atol=1e-14)
    assert outcome.residual_time == 0.1


def test_direct_step_resolves_one_scope_and_matches_explicit_capabilities(monkeypatch):
    problem = LinearProblem()
    integrator = TrapezoidIntegrator()
    state = integrator.start(np.ones(1), 0.0, v0=-np.ones(1))
    original = ProblemCapabilities.resolve
    calls = []

    def resolve(_cls, value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(ProblemCapabilities, 'resolve', classmethod(resolve))
    options = {'solve': np.linalg.solve, 'max_newton': 1, 'tol': 1e-14}
    direct = take_step(problem, integrator, state, 0.1, 0.1, **options)
    assert calls == [problem]
    explicit = take_step(
        problem,
        integrator,
        state,
        0.1,
        0.1,
        capabilities=original(problem),
        **options,
    )
    assert calls == [problem]
    np.testing.assert_array_equal(direct.u, explicit.u)
    np.testing.assert_array_equal(direct.rate, explicit.rate)
    assert direct.residual == explicit.residual
    assert direct.on_floor == explicit.on_floor


def test_capabilities_record_structure_not_mutable_physical_flags():
    class OptionalProblem(LinearProblem):
        has_tangent_jacobian = False

        def jacobian_tangent(self, *args):
            return self.jacobian(*args)

        def factor_reuse_key(self, *args):
            return args[-1]

        def residual_roundoff_floor(self, u, form, residual=None):
            return None

        def commit_history(self, u, t):
            pass

    problem = OptionalProblem()
    capabilities = ProblemCapabilities.resolve(problem)
    assert capabilities == ProblemCapabilities(True, True, True, True, False, False)
    problem.has_tangent_jacobian = True
    assert capabilities == ProblemCapabilities.resolve(problem)
    assert ProblemCapabilities.resolve(LinearProblem()) == ProblemCapabilities(False, False, False, False, False, False)


def test_trapezoid_step_prepares_one_form_and_keeps_left_limit_provenance(monkeypatch):
    integrator = TrapezoidIntegrator()
    original = integrator.prepare
    forms = []

    def prepare(*args):
        form = original(*args)
        forms.append(form)
        return form

    monkeypatch.setattr(integrator, 'prepare', prepare)
    state = integrator.start(np.ones(1), 0.0, v0=-np.ones(1))
    outcome = take_step(
        LinearProblem(),
        integrator,
        state,
        0.1,
        0.1,
        endpoint_left=True,
        solve=np.linalg.solve,
        max_newton=1,
        tol=1e-14,
    )
    assert len(forms) == 1
    assert outcome.residual_time == np.nextafter(0.1, -np.inf)
    assert outcome.state.t == 0.1
    np.testing.assert_array_equal(outcome.rate, forms[0].u_dot(outcome.u))
    np.testing.assert_array_equal(outcome.state.v, outcome.rate)


@pytest.mark.parametrize('integrator', list(INTEGRATOR_REGISTRY.values()), ids=lambda scheme: scheme.name)
def test_prepared_form_history_is_bit_identical_to_reprepared_form(integrator):
    state = integrator.start(np.array([1.3]), 0.2, v0=np.array([-0.7]), a0=np.array([0.1]))
    for dt in [0.01, 0.023, 0.001]:
        for _ in range(integrator.stages):
            target = 0.2 + dt
            form = integrator.prepare(state, dt, target)
            value = np.array([0.95])
            supplied = integrator.advance(state, value, dt, target, form)
            reconstructed = integrator.advance(state, value, dt, target, integrator.prepare(state, dt, target))
            for name in ['v', 'a']:
                np.testing.assert_array_equal(getattr(supplied, name), getattr(reconstructed, name))
            np.testing.assert_array_equal(supplied.us, reconstructed.us)
            assert (supplied.t, supplied.h, supplied.stage) == (
                reconstructed.t,
                reconstructed.h,
                reconstructed.stage,
            )
            state = supplied


class NativeProblem(LinearProblem):
    """A problem with its own step kernel; ``answer`` is what the kernel returns."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls = []

    def native_step(self, form, guess, **options):
        self.calls.append(options)
        return self.answer


def _native_take(problem, **extra):
    integrator = TrapezoidIntegrator()
    state = integrator.start(np.ones(1), 0.0, v0=-np.ones(1))
    options = {'solve': np.linalg.solve, 'max_newton': 1, 'tol': 1e-14, **extra}
    return take_step(problem, integrator, state, 0.1, 0.1, **options)


def test_a_problem_that_steps_itself_answers_the_default_newton():
    planted = (np.array([0.5]), True, 1, 0.0, False)
    problem = NativeProblem(planted)
    assert ProblemCapabilities.resolve(problem).native_step
    outcome = _native_take(problem)
    np.testing.assert_array_equal(outcome.u, [0.5])
    assert problem.calls == [{'tol': 1e-14, 'max_newton': 1, 'backtrack_cap': 8, 'backtrack_band': None}]


def test_a_declining_kernel_and_a_non_default_option_both_keep_the_python_loop():
    declined = NativeProblem(None)
    np.testing.assert_allclose(_native_take(declined).u, [19 / 21], rtol=1e-14, atol=1e-14)
    assert len(declined.calls) == 1
    bypassed = NativeProblem((np.array([0.5]), True, 1, 0.0, False))
    np.testing.assert_allclose(_native_take(bypassed, tol_rel=1e-3).u, [19 / 21], rtol=1e-14, atol=1e-14)
    assert bypassed.calls == []
