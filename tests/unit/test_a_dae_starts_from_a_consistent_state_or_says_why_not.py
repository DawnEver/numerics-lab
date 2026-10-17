"""Consistent index-1 initialisation: the algebraic unknowns are derived, the rule is stated, a conflict raises."""

from __future__ import annotations

import numpy as np
import pytest

from motronics.gauss.formulation import DerivativeWeights, IntegratorKind, TimeWindow
from motronics.gauss.time_integrators import integrate_transient
from motronics.gauss.time_integrators.initial import InconsistentInitialCondition, consistent_initial_condition

_R, _C = 2.0, 0.5


class _RcWithResistorCurrent:
    """``u = [v, i]``: ``C v' + i = 0`` (KCL), ``R i - v = 0`` (the resistor) -- v differential, i algebraic."""

    m = np.array([[_C, 0.0], [0.0, 0.0]])
    k = np.array([[0.0, 1.0], [-1.0, _R]])

    def residual(self, u, u_dot, u_ddot, t):
        return self.m @ u_dot + self.k @ u

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return weights.alpha0.real * self.k + weights.alpha1.real * self.m


class _CapacitorAcrossASource:
    """``u = [v, i]``: ``C v' + i = 0``, ``v - E = 0`` -- the source fixes the capacitor's state (index 2)."""

    m = np.array([[_C, 0.0], [0.0, 0.0]])
    k = np.array([[0.0, 1.0], [1.0, 0.0]])

    def residual(self, u, u_dot, u_ddot, t):
        return self.m @ u_dot + self.k @ u - np.array([0.0, 1.0])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return weights.alpha0.real * self.k + weights.alpha1.real * self.m


def test_the_algebraic_current_and_the_rate_are_derived_from_the_held_voltage() -> None:
    problem = _RcWithResistorCurrent()
    declared = np.array([1.0, 0.0])
    assert np.max(np.abs(problem.residual(declared, np.zeros(2), np.zeros(2), 0.0))) > 0.1  # planted: inconsistent
    start = consistent_initial_condition(problem, declared)
    np.testing.assert_allclose(start.u, [1.0, 1.0 / _R])
    np.testing.assert_allclose(start.u_dot[0], -1.0 / (_R * _C))
    assert start.algebraic == (1,)
    assert start.adjusted == (1,)
    assert np.max(np.abs(problem.residual(start.u, start.u_dot, np.zeros(2), 0.0))) < 1e-12


def test_a_pinned_algebraic_value_that_conflicts_raises_naming_it() -> None:
    with pytest.raises(InconsistentInitialCondition, match=r'algebraic unknown\(s\) \[1\] were pinned'):
        consistent_initial_condition(_RcWithResistorCurrent(), np.array([1.0, 0.0]), fixed=(1,))


def test_a_pinned_algebraic_value_that_agrees_is_accepted() -> None:
    start = consistent_initial_condition(_RcWithResistorCurrent(), np.array([1.0, 0.5]), fixed=(1,))
    assert start.adjusted == ()


def test_an_index_two_start_raises_with_its_residual() -> None:
    with pytest.raises(InconsistentInitialCondition, match='index > 1') as info:
        consistent_initial_condition(_CapacitorAcrossASource(), np.array([0.0, 0.0]))
    assert info.value.residual > 0.0


def test_the_consistent_rate_seeds_the_trapezoid_and_removes_its_bootstrap_error() -> None:
    problem = _RcWithResistorCurrent()
    start = consistent_initial_condition(problem, np.array([1.0, 0.0]))
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.TRAPEZOID)
    exact = np.exp(-1.0 / (_R * _C))
    seeded = integrate_transient(problem, study, start.u, v0=start.u_dot)
    bootstrapped = integrate_transient(problem, study, start.u)
    assert abs(seeded.states[-1][0] - exact) < 0.5 * abs(bootstrapped.states[-1][0] - exact)


def test_the_weights_used_are_the_pure_derivative_split() -> None:
    w = DerivativeWeights(alpha0=0.0, alpha1=1.0)
    np.testing.assert_allclose(_RcWithResistorCurrent().jacobian(None, None, None, 0.0, w), _RcWithResistorCurrent.m)
