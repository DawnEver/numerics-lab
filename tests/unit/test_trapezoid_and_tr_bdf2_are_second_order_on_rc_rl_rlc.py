"""Trapezoid and TR-BDF2 converge at second order on analytic RC, RL and RLC circuits.

Each circuit is a linear ``M u_dot + K u = f(t)`` written as a :class:`SemiDiscreteProblem`; the
reference is closed form (RC, RL) or the exact matrix exponential of the state-space form (RLC).
The OBSERVED order ``log2(e(dt) / e(dt/2))`` is printed and asserted. The planted control runs the
same instrument on BDF1 and must read first order -- an order test that cannot tell a first-order
scheme from a second-order one proves nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import INTEGRATOR_REGISTRY, _step, integrate_transient


class _Linear:
    """``R = M u_dot + K u - f(t)``."""

    def __init__(self, m, k, f) -> None:
        self.m, self.k, self.f = np.atleast_2d(m), np.atleast_2d(k), f

    def residual(self, u, u_dot, u_ddot, t):
        return self.m @ u_dot + self.k @ u - self.f(t)

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return weights.alpha0.real * self.k + weights.alpha1.real * self.m


_R, _L, _C = 2.0, 0.5, 0.1
_W = 7.0


def _rc():
    """RC driven by ``sin(w t)``: ``R C v' + v = sin(w t)``, ``v(0) = 0``."""
    tau = _R * _C

    def exact(t):
        a = 1.0 / (1.0 + (_W * tau) ** 2)
        return a * (math.sin(_W * t) - _W * tau * math.cos(_W * t) + _W * tau * math.exp(-t / tau))

    return _Linear(tau, 1.0, lambda t: np.array([math.sin(_W * t)])), np.array([0.0]), lambda t: np.array([exact(t)])


def _rl():
    """RL switched onto 1 V: ``L i' + R i = 1``, ``i(0) = 0``."""
    return (
        _Linear(_L, _R, lambda _t: np.array([1.0])),
        np.array([0.0]),
        lambda t: np.array([(1.0 - math.exp(-_R * t / _L)) / _R]),
    )


def _rlc():
    """Series RLC on 1 V, state ``[i, v_C]``: ``L i' + R i + v_C = 1``, ``C v_C' - i = 0``."""
    m = np.diag([_L, _C])
    k = np.array([[_R, 1.0], [-1.0, 0.0]])
    a = -np.linalg.solve(m, k)
    b = np.linalg.solve(m, np.array([1.0, 0.0]))
    x_inf = -np.linalg.solve(a, b)

    def exact(t):
        return x_inf + expm(a * t) @ (np.zeros(2) - x_inf)

    return _Linear(m, k, lambda _t: np.array([1.0, 0.0])), np.zeros(2), exact


_CIRCUITS = {'rc': _rc, 'rl': _rl, 'rlc': _rlc}


def _error(kind: IntegratorKind, circuit: str, dt: float) -> float:
    problem, u0, exact = _CIRCUITS[circuit]()
    sol = integrate_transient(problem, TimeWindow(t0=0.0, t1=0.5, dt=dt, integrator=kind), u0)
    return max(float(np.max(np.abs(u - exact(t)))) for t, u in zip(sol.times, sol.states, strict=True))


def _observed_order(kind: IntegratorKind, circuit: str) -> float:
    dts = (0.01, 0.005, 0.0025)
    errors = [_error(kind, circuit, dt) for dt in dts]
    order = math.log2(errors[-2] / errors[-1])
    print(f'{kind.value} on {circuit}: errors {[f"{e:.3e}" for e in errors]} -> observed order {order:.3f}')  # noqa: T201
    return order


@pytest.mark.parametrize('circuit', sorted(_CIRCUITS))
@pytest.mark.parametrize('kind', [IntegratorKind.TRAPEZOID, IntegratorKind.TR_BDF2])
def test_the_scheme_converges_at_second_order(kind: IntegratorKind, circuit: str) -> None:
    assert 1.9 <= _observed_order(kind, circuit) <= 2.2


@pytest.mark.parametrize('circuit', sorted(_CIRCUITS))
def test_planted_control_the_instrument_reads_bdf1_as_first_order(circuit: str) -> None:
    assert 0.9 <= _observed_order(IntegratorKind.BDF1, circuit) <= 1.1


def test_both_schemes_are_registered_and_resolvable() -> None:
    assert INTEGRATOR_REGISTRY[IntegratorKind.TRAPEZOID].name == 'trapezoid'
    assert INTEGRATOR_REGISTRY[IntegratorKind.TR_BDF2].name == 'tr_bdf2'


def test_tr_bdf2_is_l_stable_where_the_trapezoid_rings() -> None:
    """A stiff decay at ``lambda dt = 1e4``: the trapezoid's amplification is ~ -1, TR-BDF2's ~ 0."""
    problem = _Linear(1.0, 1.0e4, lambda _t: np.array([0.0]))

    def ratio(kind):
        sol = integrate_transient(
            problem, TimeWindow(t0=0.0, t1=5.0, dt=1.0, integrator=kind), np.array([1.0]), v0=np.array([-1.0e4])
        )
        return float(sol.states[-1][0] / sol.states[-2][0])

    assert ratio(IntegratorKind.TRAPEZOID) < -0.99
    assert abs(ratio(IntegratorKind.TR_BDF2)) < 1e-3


def test_tr_bdf2_takes_two_solves_per_step_with_one_jacobian_weight(monkeypatch) -> None:
    weights = []
    original = _step._solve_step

    def observe(problem, form, *args, **kwargs):
        weights.append(complex(form.weights.alpha1))
        return original(problem, form, *args, **kwargs)

    monkeypatch.setattr(_step, '_solve_step', observe)
    problem, u0, _exact = _rl()
    v0 = np.array([1.0 / _L])
    integrate_transient(problem, TimeWindow(t0=0.0, t1=0.4, dt=0.1, integrator=IntegratorKind.TR_BDF2), u0, v0=v0)
    assert len(weights) == 8
    assert np.allclose(weights, (2.0 + 2.0**0.5) / 0.1)


class _SolveRoundoff(_Linear):
    """The macOS residue, planted: a tangent off by ``4 eps`` relative -- the backward error a stable solve may commit.

    One exact Newton step then lands at ``4 eps |J| |du|``, the rounding of the residual it REMOVED: at the
    TR-BDF2 stage that decays ``u`` from ~1 to ~5e-4 that is ~9e-12, far above ``eps |J| |u|`` (~1e-15).
    """

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return super().jacobian(u, u_dot, u_ddot, t, weights) * (1.0 + 4.0 * np.finfo(float).eps)


def test_the_step_floor_scales_with_the_residual_the_newton_step_removed() -> None:
    problem = _SolveRoundoff(1.0, 1.0e4, lambda _t: np.array([0.0]))
    sol = integrate_transient(
        problem,
        TimeWindow(t0=0.0, t1=5.0, dt=1.0, integrator=IntegratorKind.TR_BDF2),
        np.array([1.0]),
        v0=np.array([-1.0e4]),
    )
    assert abs(float(sol.states[-1][0] / sol.states[-2][0])) < 1e-3
