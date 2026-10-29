"""Consistent index-1 initialisation: the algebraic unknowns are derived, the rule is stated, a conflict raises."""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.formulation import DerivativeWeights, IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.initial import InconsistentInitialCondition, consistent_initial_condition

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


def test_a_start_inside_the_tolerance_is_still_solved_not_returned_as_declared() -> None:
    """``tol`` says when to stop iterating, not that the declared values are the answer.

    A settled state whose residual is already under ``tol * scale`` was returned untouched: its
    algebraic current stayed at the declared 0 and its rate at 0. That is a tolerance-sized error
    at every sampled hit of a settled loop (case_q tracking 5.4e-15 -> 4.4e-13, 2026-10-18).
    """
    problem = _RcWithResistorCurrent()
    v = 1e-11
    declared = np.array([v, 0.0])
    assert np.max(np.abs(problem.residual(declared, np.zeros(2), np.zeros(2), 0.0))) <= 1e-10  # planted: inside tol
    start = consistent_initial_condition(problem, declared)
    assert start.u[1] == pytest.approx(v / _R, rel=1e-12, abs=0.0)
    assert start.u_dot[0] == pytest.approx(-v / (_R * _C), rel=1e-12, abs=0.0)


def test_an_index_two_start_that_is_already_consistent_is_accepted() -> None:
    """The one correction is skipped where none exists: a consistent index-2 start is returned as it was."""
    start = consistent_initial_condition(_CapacitorAcrossASource(), np.array([1.0, 0.0]))
    np.testing.assert_array_equal(start.u, [1.0, 0.0])


class _RcWithASmallResistor(_RcWithResistorCurrent):
    """The same RC with ``R = 1e-3``: a residual of 1e-11 on the resistor row is a 1e-8 current."""

    k = np.array([[0.0, 1.0], [-1.0, 1e-3]])


class _Empty:
    """A system with no unknowns: a pure-source subsystem."""

    def residual(self, u, u_dot, u_ddot, t):
        return np.zeros(0)

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.zeros((0, 0))


def test_an_empty_system_is_its_own_consistent_start() -> None:
    start = consistent_initial_condition(_Empty(), np.zeros(0))
    assert start.u.size == 0 and start.u_dot.size == 0 and start.residual == 0.0


def test_one_newton_step_suffices_for_a_linear_start() -> None:
    """The residual is read AFTER the step: one allowed step that lands the start is a success, not a failure."""
    start = consistent_initial_condition(_RcWithResistorCurrent(), np.array([1.0, 0.0]), max_newton=1)
    np.testing.assert_allclose(start.u, [1.0, 1.0 / _R])


def test_a_pin_inside_the_tolerance_is_held_not_moved() -> None:
    """A pinned value is the caller's: it is held out of the solve, so a consistent pin is never pushed past its test.

    PLANTED: the resistor row's residual is 1e-11 (inside tol); solving that row for the pinned current would move
    it by 1e-8, past its 1e-10 move test, and refuse a start that is consistent.
    """
    declared = np.array([1e-11, 0.0])
    start = consistent_initial_condition(_RcWithASmallResistor(), declared, fixed=(1,))
    assert start.u[1] == 0.0
    assert start.adjusted == ()
