"""Unit tests for the physics-blind time-integration runtime (``gauss.time_integrators``).

The real correctness gate is a CONVERGENCE-ORDER test: integrate a scalar linear ODE with a
known analytic solution and assert the global error shrinks at the scheme's expected order as
``dt`` halves (BDF1 -> O(dt), BDF2 -> O(dt^2)). The remaining tests pin the textbook
DerivativeWeights of each integrator and exercise a tiny fake SemiDiscreteProblem through the
``integrate_transient`` driver.
"""

from __future__ import annotations

import numpy as np
import pytest

from motronics.contracts.analysis_formulation import SemiDiscreteProblem
from motronics.contracts.study_spec import IntegratorKind, TransientStudy
from motronics.gauss.time_integrators import (
    INTEGRATOR_REGISTRY,
    BDF1Integrator,
    BDF2Integrator,
    GeneralizedAlphaIntegrator,
    NewmarkIntegrator,
    get_integrator,
    integrate_transient,
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


def test_fake_problem_satisfies_protocol() -> None:
    """The tiny fakes structurally implement the runtime-checkable SemiDiscreteProblem Protocol."""
    assert isinstance(DecayProblem(1.0), SemiDiscreteProblem)
    assert isinstance(OscillatorProblem(1.0), SemiDiscreteProblem)


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
    study = TransientStudy(t0=0.0, t1=t1, dt=dt, integrator=kind)
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
    study = TransientStudy(t0=0.0, t1=0.5, dt=dt, integrator=IntegratorKind.BDF1)
    sol = integrate_transient(DecayProblem(lam), study, u0=np.array([1.0]))
    expected = 1.0
    factor = 1.0 / (1.0 + lam * dt)
    for u in sol.states[1:]:
        expected *= factor
        assert float(u[0]) == pytest.approx(expected, rel=1e-12)


def test_2x2_decoupled_system() -> None:
    """A 2x2 decoupled decay system marches each component to its own exponential."""
    lam = 1.5
    study = TransientStudy(t0=0.0, t1=1.0, dt=0.01, integrator=IntegratorKind.BDF2)
    sol = integrate_transient(DecayProblem(lam, n=2), study, u0=np.array([1.0, -2.0]))
    final = sol.states[-1]
    ref = np.array([1.0, -2.0]) * np.exp(-lam * 1.0)
    assert final == pytest.approx(ref, abs=2e-3)


def test_newmark_oscillator_energy_bounded() -> None:
    """Undamped Newmark on SHM stays bounded and tracks cos(omega t) to leading order."""
    omega, dt = 2.0, 0.005
    study = TransientStudy(t0=0.0, t1=1.0, dt=dt, integrator=IntegratorKind.NEWMARK)
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
