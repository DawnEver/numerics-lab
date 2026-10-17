"""Unit tests for the physics-blind time-integration runtime (``gauss.time_integrators``).

The real correctness gate is a CONVERGENCE-ORDER test: integrate a scalar linear ODE with a
known analytic solution and assert the global error shrinks at the scheme's expected order as
``dt`` halves (BDF1 -> O(dt), BDF2 -> O(dt^2)). The remaining tests pin the textbook
DerivativeWeights of each integrator and exercise a tiny fake SemiDiscreteProblem through the
``integrate_transient`` driver.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numerics_lab import time_integrators
from numerics_lab.formulation import IntegratorKind, SemiDiscreteProblem, TimeWindow
from numerics_lab.time_integrators import (
    INTEGRATOR_REGISTRY,
    BDF1Integrator,
    BDF2Integrator,
    GeneralizedAlphaIntegrator,
    NewmarkIntegrator,
    _forms,
    _step,
    get_integrator,
    integrate_transient,
    march,
)


class DecayProblem:
    """First-order linear decay ``u_dot = -lambda u`` as a :class:`SemiDiscreteProblem`.

    Residual ``R = C u_dot + K u`` with ``K = lambda I`` (stiffness) and ``C = I`` (mass on the
    first derivative), so ``R = 0`` reproduces ``u_dot = -lambda u`` whose solution is
    ``u(t) = u0 exp(-lambda t)``. The jacobian is the weighted operator ``alpha0 K + alpha1 C``.
    """

    def __init__(self, lam: float, n: int = 1) -> None:
        self.lam = lam
        self.k = lam * np.eye(n)
        self.c = np.eye(n)

    def residual(self, u, u_dot, u_ddot, t):
        return self.c @ u_dot + self.k @ u

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return weights.alpha0.real * self.k + weights.alpha1.real * self.c


class OscillatorProblem:
    """Undamped SHM ``u_ddot + omega^2 u = 0`` as a second-order :class:`SemiDiscreteProblem`.

    ``R = M u_ddot + K u`` with ``M = I`` and ``K = omega^2 I``; solution ``u(t) = cos(omega t)``
    for ``u(0) = 1``, ``u_dot(0) = 0``. Drives the Newmark / generalized-alpha second-order path.
    """

    def __init__(self, omega: float) -> None:
        self.m = np.eye(1)
        self.k = (omega**2) * np.eye(1)

    def residual(self, u, u_dot, u_ddot, t):
        return self.m @ u_ddot + self.k @ u

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return weights.alpha0.real * self.k + weights.alpha2.real * self.m


def test_the_public_march_uses_the_single_step_owner_and_canonical_state(monkeypatch):
    calls = []
    original = _step._solve_step

    def observe(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(_step, '_solve_step', observe)
    study = TimeWindow(t0=0.0, t1=0.2, dt=0.1, integrator=IntegratorKind.BDF2)
    result = integrate_transient(DecayProblem(1.0), study, np.array([1.0]))
    assert len(calls) == 2
    assert all(isinstance(form, time_integrators.StepForm) for form in calls)
    assert time_integrators.StepForm is _forms.StepForm
    assert time_integrators.IntegratorState is _forms.IntegratorState
    assert time_integrators.TransientSolution is _forms.TransientSolution
    assert time_integrators.HeldFactor is _step.HeldFactor
    assert result.converged


def test_event_marches_reach_the_same_step_owner(monkeypatch):
    study = TimeWindow(t0=0.0, t1=0.2, dt=0.1, integrator=IntegratorKind.BDF2)
    kwargs = {'time_breakpoints': [0.15]}
    expected = march.integrate_transient(DecayProblem(1.0), study, np.array([1.0]), **kwargs)
    calls = []
    original = _step._solve_step

    def observe(*args, **options):
        calls.append(args[1])
        return original(*args, **options)

    monkeypatch.setattr(_step, '_solve_step', observe)
    actual = march.integrate_transient(DecayProblem(1.0), study, np.array([1.0]), **kwargs)
    assert len(calls) == 3
    assert all(isinstance(form, _forms.StepForm) for form in calls)
    np.testing.assert_array_equal(actual.times, expected.times)
    np.testing.assert_array_equal(actual.states, expected.states)
    assert actual.converged == expected.converged


def test_fake_problem_satisfies_protocol() -> None:
    """The tiny fakes structurally implement the runtime-checkable SemiDiscreteProblem Protocol."""
    assert isinstance(DecayProblem(1.0), SemiDiscreteProblem)
    assert isinstance(OscillatorProblem(1.0), SemiDiscreteProblem)


@pytest.mark.parametrize('kind', [IntegratorKind.BDF1, IntegratorKind.BDF2, IntegratorKind.NEWMARK])
def test_fixed_grid_lands_a_partial_final_step_at_the_declared_end(kind: IntegratorKind) -> None:
    """Every fixed-grid scheme must use the true partial cell, not an approximate step count."""
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.3, integrator=kind)
    problem = OscillatorProblem(1.0) if kind == IntegratorKind.NEWMARK else DecayProblem(1.0)
    solution = integrate_transient(problem, study, np.array([1.0]))
    assert solution.times == pytest.approx([0.0, 0.3, 0.6, 0.9, 1.0])
    assert np.all(np.isfinite(solution.states))
    if kind == IntegratorKind.BDF2:
        # Unequal final and previous spacings require the nonuniform BDF2
        # derivative, not the uniform weights evaluated at 0.1 s.
        h, previous = 0.1, 0.3
        a = (2 * h + previous) / (h * (h + previous))
        history = -(h + previous) / (h * previous) * solution.states[-2][0]
        history += h / (previous * (h + previous)) * solution.states[-3][0]
        assert solution.states[-1][0] == pytest.approx(-history / (a + 1.0))


# ---- DerivativeWeights: each integrator emits the textbook alpha triple -------------------


def test_bdf1_weights() -> None:
    w = BDF1Integrator().weights(dt=0.1)
    assert w.alpha0 == 1.0
    assert w.alpha1 == pytest.approx(1.0 / 0.1)  # 1/dt
    assert w.alpha2 == 0.0


def test_bdf2_weights() -> None:
    w = BDF2Integrator().weights(dt=0.1)
    assert w.alpha0 == 1.0
    assert w.alpha1 == pytest.approx(1.5 / 0.1)  # 3/(2 dt)
    assert w.alpha2 == 0.0


@pytest.mark.parametrize('integrator', [BDF1Integrator(), BDF2Integrator()])
def test_fixed_step_bdf_reuses_its_immutable_weights(integrator) -> None:
    """A fixed-step march has one derivative operator; it must not allocate it at every step."""
    assert integrator.weights(dt=0.1) is integrator.weights(dt=0.1)
    assert integrator.weights(dt=0.2) is not integrator.weights(dt=0.1)


def test_newmark_weights() -> None:
    dt = 0.1
    w = NewmarkIntegrator().weights(dt)  # average-acceleration beta=1/4, gamma=1/2
    assert w.alpha1 == pytest.approx(0.5 / (0.25 * dt))  # gamma/(beta dt) = 2/dt
    assert w.alpha2 == pytest.approx(1.0 / (0.25 * dt * dt))  # 1/(beta dt^2) = 4/dt^2


def test_generalized_alpha_undamped_at_rho_one() -> None:
    """rho_inf = 1 is the undamped limit: alpha_m = alpha_f = 1/2 (equal), so beta=1/4, gamma=1/2."""
    integ = GeneralizedAlphaIntegrator(rho_inf=1.0)
    assert integ.alpha_m == pytest.approx(0.5)
    assert integ.alpha_f == pytest.approx(0.5)
    assert integ.beta == pytest.approx(0.25)
    assert integ.gamma == pytest.approx(0.5)


def test_generalized_alpha_params_standard_forms() -> None:
    """Chung-Hulbert parameter forms for a general spectral radius."""
    rho = 0.5
    integ = GeneralizedAlphaIntegrator(rho_inf=rho)
    assert integ.alpha_m == pytest.approx((2 * rho - 1) / (rho + 1))
    assert integ.alpha_f == pytest.approx(rho / (rho + 1))
    assert integ.beta == pytest.approx(0.25 * (1 - integ.alpha_m + integ.alpha_f) ** 2)
    assert integ.gamma == pytest.approx(0.5 - integ.alpha_m + integ.alpha_f)


# ---- Convergence order: the real correctness gate -----------------------------------------


def _decay_error(kind: IntegratorKind, dt: float, lam: float = 2.0, t1: float = 1.0) -> float:
    """Max abs error of the marched solution vs u(t)=exp(-lambda t) at step size dt."""
    study = TimeWindow(t0=0.0, t1=t1, dt=dt, integrator=kind)
    sol = integrate_transient(DecayProblem(lam), study, u0=np.array([1.0]))
    err = 0.0
    for t, u in zip(sol.times, sol.states, strict=True):
        err = max(err, abs(float(u[0]) - np.exp(-lam * t)))
    return err


def _observed_order(kind: IntegratorKind) -> float:
    """Empirical convergence order from halving dt: log2(err(dt) / err(dt/2))."""
    e1 = _decay_error(kind, dt=0.02)
    e2 = _decay_error(kind, dt=0.01)
    return float(np.log2(e1 / e2))


def test_bdf1_first_order_convergence() -> None:
    order = _observed_order(IntegratorKind.BDF1)
    assert order == pytest.approx(1.0, abs=0.1), f'BDF1 observed order {order}'


def test_bdf2_second_order_convergence() -> None:
    order = _observed_order(IntegratorKind.BDF2)
    assert order == pytest.approx(2.0, abs=0.15), f'BDF2 observed order {order}'


def test_bdf_error_shrinks_with_dt() -> None:
    """Sanity: halving dt reduces the error for both first-order backward-difference schemes."""
    for kind in (IntegratorKind.BDF1, IntegratorKind.BDF2):
        assert _decay_error(kind, dt=0.01) < _decay_error(kind, dt=0.02)


# ---- Driver: fake problem reproduces the analytic ODE -------------------------------------


def test_backward_euler_reproduces_recurrence() -> None:
    """BDF1 on the decay ODE matches the closed-form backward-Euler recurrence u/(1+lambda dt)."""
    lam, dt = 3.0, 0.05
    study = TimeWindow(t0=0.0, t1=0.5, dt=dt, integrator=IntegratorKind.BDF1)
    sol = integrate_transient(DecayProblem(lam), study, u0=np.array([1.0]))
    expected = 1.0
    factor = 1.0 / (1.0 + lam * dt)
    for u in sol.states[1:]:
        expected *= factor
        assert float(u[0]) == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_2x2_decoupled_system() -> None:
    """A 2x2 decoupled decay system marches each component to its own exponential."""
    lam = 1.5
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.01, integrator=IntegratorKind.BDF2)
    sol = integrate_transient(DecayProblem(lam, n=2), study, u0=np.array([1.0, -2.0]))
    final = sol.states[-1]
    ref = np.array([1.0, -2.0]) * np.exp(-lam * 1.0)
    assert final == pytest.approx(ref, abs=2e-3)


def test_newmark_oscillator_energy_bounded() -> None:
    """Undamped Newmark on SHM stays bounded and tracks cos(omega t) to leading order."""
    omega, dt = 2.0, 0.005
    study = TimeWindow(t0=0.0, t1=1.0, dt=dt, integrator=IntegratorKind.NEWMARK)
    sol = integrate_transient(OscillatorProblem(omega), study, u0=np.array([1.0]), v0=np.array([0.0]))
    for t, u in zip(sol.times, sol.states, strict=True):
        assert abs(float(u[0])) <= 1.05  # average-acceleration Newmark conserves energy (no growth)
        assert float(u[0]) == pytest.approx(np.cos(omega * t), abs=5e-2)


def test_registry_covers_every_integrator_kind() -> None:
    """Every IntegratorKind resolves to an integrator whose declared kind matches the enum value."""
    for kind in IntegratorKind:
        integ = get_integrator(kind)
        assert INTEGRATOR_REGISTRY[kind] is integ
        assert integ.name == kind.value


def test_unknown_integrator_raises() -> None:
    with pytest.raises(KeyError):
        get_integrator('not-a-kind')  # type: ignore[arg-type]


def test_an_exhausted_newton_ceiling_labels_the_solution(caplog) -> None:
    """The CONVERGENCE CONTRACT: a step that exhausts its Newton ceiling is LABELLED, never
    silent -- the solution carries the worst step residual and the offending times, the
    marcher warns (AGENTS.md: the caller can tell), and the caller reads the ``converged``
    tag. The DC-op precedent is the raise; the transient's contract is label-don't-block
    (the FVS plan's replacement for the old silent accept, measured 2026-08-07: the
    transient's per-step Newton used to return the unconverged iterate with no flag).
    """

    class _QuadraticDrift:
        """``u_dot = u^2``: the implicit BDF step's Newton needs several iterations at a coarse dt."""

        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - np.asarray(u) ** 2

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real - 2.0 * float(u[0]) * weights.alpha0.real]])

    problem = _QuadraticDrift()
    study = TimeWindow(t0=0.0, t1=0.5, dt=0.1, integrator=IntegratorKind.BDF2)
    with caplog.at_level('WARNING', logger='numerics_lab.time_integrators'):
        coarse = integrate_transient(problem, study, np.array([1.0]), max_newton=1, tol=1e-12, on_nonconverged='label')
    assert not coarse.converged
    assert coarse.nonconverged_times, 'an exhausted step must be named on the solution'
    assert coarse.worst_step_residual > 0.0
    assert any('did not converge' in rec.message for rec in caplog.records)
    # The controlled comparison: with enough Newton iterations the same march converges.
    fine = integrate_transient(problem, study, np.array([1.0]), max_newton=20, tol=1e-12)
    assert fine.converged and not fine.nonconverged_times


def test_adaptive_stepping_keeps_accuracy_with_fewer_steps_on_the_smooth_tail() -> None:
    """The BDF2 predictor-curvature controller sizes the steps -- the march takes
    the large steps on the smooth tail where the fixed march wastes them, at the same
    accuracy per tolerance. ``u_dot = -u`` has the exact ``u = e^{-t}``.
    """

    class _Decay:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) + np.asarray(u)

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real + weights.alpha0.real]])

    problem = _Decay()
    study = TimeWindow(t0=0.0, t1=5.0, dt=1e-3, integrator=IntegratorKind.BDF2)
    fixed = integrate_transient(problem, study, np.array([1.0]))
    # atol = 1e-6: the predictor discrepancy ~ u'' h^2 puts the controller's equilibrium at
    # h ~ sqrt(atol/u'') ~ 1e-3 -- the fixed march's own scale, so the comparison is fair.
    adaptive = integrate_transient(problem, study, np.array([1.0]), curvature_atol=1e-6)

    def err(sol):
        return max(abs(float(u[0]) - math.exp(-t)) for t, u in zip(sol.times, sol.states, strict=True))

    assert err(adaptive) <= max(err(fixed) * 10.0, 1e-8), (
        f'adaptive true error {err(adaptive):g} vs fixed {err(fixed):g} -- the curvature controller '
        'must not trade accuracy for steps'
    )
    steps = np.diff(np.asarray(adaptive.times))
    assert steps.max() > 10.0 * steps.min(), (
        f'adaptive steps span only {steps.min():.2e}..{steps.max():.2e} -- the controller must '
        'size the steps (small in the transient, large on the smooth tail)'
    )
    assert adaptive.converged


def test_adaptive_stepping_rejects_bdf1_with_a_named_error() -> None:
    """The implemented predictor-curvature controller is BDF2-specific."""
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF1)

    class _Decay:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) + np.asarray(u)

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real + weights.alpha0.real]])

    with pytest.raises(ValueError, match='BDF2 only'):
        integrate_transient(_Decay(), study, np.array([1.0]), curvature_atol=1e-6)
