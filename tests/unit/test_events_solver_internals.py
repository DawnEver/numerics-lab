"""Per-step Newton internals (``_solve_step``) exercised through the events seam.

Split from the original ``test_events.py``: locator/rollback behaviors live in
``test_events_location.py``, ``time_breakpoints`` in ``test_events_breakpoints.py``, and
adaptive-step/refine accuracy tests in ``test_events_step_control.py``.
"""
# ruff: noqa: PLC0415 -- test-module lazy imports: the module under test is imported inside
# the test/fixture, keeping collection cheap and loading platform-conditional deps (win32com,
# OCP, FEMM) only on the path that needs them. Hoist a file's imports and DELETE this header;
# the population is ratcheted by tests/architecture/ratchets/test_suppression_ratchet.py.

from __future__ import annotations

import math
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest
from tests.unit._helpers import _RampWithToggle

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import _step as _step_mod
from numerics_lab.time_integrators import integrate_transient


def test_the_relative_tolerance_labels_a_scale_stalled_step_converged() -> None:
    """MANUAL-628: the per-step Newton's absolute tol (1e-12) is unreachable when the
    residual's scale is huge -- the SRDAB closed loop's switch ramp: the KCL rows carry
    ~5400 A (270 V / 0.05 ohm), so a residual of 1.1e-1 is 2e-5 RELATIVE, and the Newton
    stalls there by SCALE, not by error. The opt-in ``tol_rel`` accepts a step whose
    residual dropped below ``tol_rel * r0`` (r0 = the first iteration's norm): a stall
    at 1e-1 from a 1e3-scale start is a 1e-4-relative residual -- converged with
    tol_rel = 1e-4, exhausted without it.
    """
    from numerics_lab.time_integrators._forms import IntegratorState
    from numerics_lab.time_integrators._integrators import _bdf2_variable_form
    from numerics_lab.time_integrators._step import _solve_step

    class _Stalled:
        """A residual that reads 1e3 on the first evaluation and 1e-1 forever after --
        the Newton stalls by scale (the solve is an identity, the residual never drops
        further).
        """

        calls = 0

        def residual(self, u_arg, u_dot, u_ddot, t):
            self.calls += 1
            norm = 1.0e3 if self.calls == 1 else 1.0e-1
            return np.array([norm])

        def jacobian(self, u_arg, u_dot, u_ddot, t, weights):
            return np.array([[1.0]])

    problem = _Stalled()
    state = IntegratorState(us=[np.array([0.0]), np.array([0.0])], t=0.0)
    form = _bdf2_variable_form(state, 0.1, 0.1, 0.1)

    _u, converged_abs, _it, residual, _on_floor = _solve_step(
        problem, form, state.last_u, solve=lambda _j, r: -r, max_newton=5, tol=1e-12
    )
    assert not converged_abs, 'the absolute contract must label the scale stall exhausted'
    assert residual == pytest.approx(1.0e-1, rel=1e-9, abs=0.0)

    problem.calls = 0  # the relative run starts its own stall
    _u, converged_rel, _it, residual, _on_floor = _solve_step(
        problem, form, state.last_u, solve=lambda _j, r: -r, max_newton=5, tol=1e-12, tol_rel=1e-4
    )
    assert converged_rel, (
        f'the relative contract must accept a 1e-4-relative stall (residual {residual:g} vs '
        'tol_rel * r0 = 1e-4 * 1e3 = 0.1)'
    )


def test_the_backtracking_line_search_halves_an_overshooting_step() -> None:
    """MANUAL-632/634: a Newton step that RAISES the residual (the sharp GaN channel's
    overshoot at the commutation) is halved up to ``backtrack_cap`` times, accepting the
    first half-step whose residual drops below ``backtrack_band * r_current``. The
    calibration (band 1.5, cap 4): the GaN bridge's exhausted steps 233 -> 82 at the same
    converged trajectory. The bare Newton (``backtrack_cap = 0``) takes the full step and
    the divergence runs away.
    """
    from numerics_lab.time_integrators._forms import IntegratorState
    from numerics_lab.time_integrators._integrators import _bdf2_variable_form
    from numerics_lab.time_integrators._step import _solve_step

    class _Quadratic:
        """f(u) = u^2 with the solve's 10x OVERSTEP: the full step u: 1 -> -9 RAISES the
        residual (1 -> 81); the 4th halving lands at u = 0.25 (r = 0.0625 < 1.5 * r0).
        """

        def __init__(self) -> None:
            self.calls = 0

        def residual(self, u_arg, u_dot, u_ddot, t):
            self.calls += 1
            u = float(u_arg[0])
            # Saturated at 1e300: the bare Newton's u explodes (1 -> -9 -> -819 -> ...),
            # and the raw square overflows float64 at |u| ~ 1e154 -- the saturation keeps
            # the divergence measurable instead of crashing the test.
            return np.array([u * u if abs(u) < 1.0e150 else 1.0e300])

        def jacobian(self, u_arg, u_dot, u_ddot, t, weights):
            return np.array([[1.0]])

    state = IntegratorState(us=[np.array([1.0]), np.array([1.0])], t=0.0)
    form = _bdf2_variable_form(state, 0.1, 0.1, 0.1)

    # The seam hands the solve the NEGATED residual, so 10 * (-r) = -10r: the right
    # direction at 10x the honest Newton step (the sharp channel's overshoot).
    def overshoot_solve(_j, r):  # the seam's 10x overstep
        return 10.0 * r

    problem_bt = _Quadratic()
    u_bt, _converged, _it, _residual, _on_floor = _solve_step(
        problem_bt, form, np.array([1.0]), solve=overshoot_solve, max_newton=2, tol=1e-12
    )
    problem_bare = _Quadratic()
    u_bare, _converged, _it, _residual, _on_floor = _solve_step(
        problem_bare,
        form,
        np.array([1.0]),
        solve=overshoot_solve,
        max_newton=2,
        tol=1e-12,
        backtrack_cap=0,
    )
    # The backtracking's first iteration evaluates the residual at the halved steps --
    # the 10x full step (r = 81) is rejected, the 4th halving (u = 0.25, r = 0.0625)
    # accepted. The bare Newton takes the full step both times (u ~ -9, then -819).
    assert problem_bt.calls > problem_bare.calls, 'the backtracking must probe the halved steps'
    assert abs(float(u_bt[0])) < 1.0, f'backtracked u {u_bt[0]:g} -- the full overshoot was taken'
    assert abs(float(u_bare[0])) > 100.0, f'bare u {u_bare[0]:g} -- the overshoot did not run away'


def test_the_stall_polish_switches_to_the_tangent_and_converges() -> None:
    """MANUAL-627 (i): the chord jacobian (the switch's secant i/v) stalls at a floor off
    the root -- the measured three-cycle (creep ~0.996, jump 8-260, drop 0.004) on the
    GaN channel-toggle steps. The stall-polish: after TWO consecutive stalled iterations
    (the per-iteration residual ratio > 0.9 -- the normal phase's ratios are all below),
    the Newton switches its jacobian source to the problem's ``jacobian_tangent`` (the
    local derivative): the chord's basin keeps the global exploration, the tangent
    finishes the convergence the chord cannot (the bridge's toggle step: 60-exhausted
    at the ~1e-3-relative floor vs 16 iterations to 4e-5).
    """
    from numerics_lab.time_integrators._forms import IntegratorState
    from numerics_lab.time_integrators._integrators import _bdf2_variable_form
    from numerics_lab.time_integrators._step import _solve_step

    class _CouplingChordBase:
        """Two switches coupled through G: the ON channel's saturation
        (i = c u/(1+m u), the chord = i/u, the tangent = 1/(1+m u)^2) beside the OFF
        channel's near-zero chord (joff) -- the chord jacobian is near-singular and the
        Newton stalls; the tangent's fixed point is the root. The chord-alone exhausts
        (measured: 30 iterations at the 3.6e-1 residual); the polish converges (16
        iterations, rel 1.6e-9).
        """

        def __init__(self) -> None:
            self.c, self.m, self.G, self.joff, self.s = 1000.0, 50.0, 1.0, 1.0e-6, 20.0
            self.tangent_calls = 0

        def residual(self, u_arg, u_dot, u_ddot, t):
            u1, u2 = u_arg
            i1 = self.c * u1 / (1.0 + self.m * u1)
            return np.array([i1 + self.G * (u1 - u2) - self.s, self.joff * u2 + self.G * (u2 - u1)])

        def jacobian(self, u_arg, u_dot, u_ddot, t, weights):
            u1, _u2 = u_arg
            d1 = self.c / (1.0 + self.m * u1) if u1 != 0.0 else self.c
            return np.array([[d1 + self.G, -self.G], [-self.G, self.joff + self.G]])

    class _CouplingChord(_CouplingChordBase):
        @property
        def has_tangent_jacobian(self):
            return True

        def jacobian_tangent(self, u_arg, u_dot, u_ddot, t, weights):
            self.tangent_calls += 1
            u1, _u2 = u_arg
            d1 = self.c / (1.0 + self.m * u1) ** 2
            return np.array([[d1 + self.G, -self.G], [-self.G, self.joff + self.G]])

    class _ChordOnly(_CouplingChordBase):
        """No alternative-Jacobian capability: the pure chord Newton's pre-polish behavior."""

    def solve(j, r):  # the seam's solve
        return np.linalg.solve(np.asarray(j), r)

    state = IntegratorState(us=[np.array([0.5, 0.5]), np.array([0.5, 0.5])], t=0.0)
    form = _bdf2_variable_form(state, 0.1, 0.1, 0.1)
    guess = np.array([0.5, 0.5])

    problem = _ChordOnly()
    _u, converged_chord, _it, residual, _on_floor = _solve_step(
        problem, form, guess, solve=solve, max_newton=30, tol=1e-12, tol_rel=1e-6
    )
    assert not converged_chord, f'the chord jacobian must stall (the precondition, residual {residual:g})'

    problem = _CouplingChord()
    _u, converged, _it, residual, _on_floor = _solve_step(
        problem, form, guess, solve=solve, max_newton=30, tol=1e-12, tol_rel=1e-6
    )
    assert converged, f'the stall-polish must converge (residual {residual:g})'
    assert problem.tangent_calls > 0, "the stall-polish must engage the problem's jacobian_tangent"


def test_a_polish_whose_step_the_line_search_cannot_accept_hands_back_to_the_newton() -> None:
    """The polish is a TRIAL, not a one-way switch: its first capped step returns the default.

    MEASURED 2026-09-26 on `case_a --point rated` under the vacuum B-H tail: the exact Newton
    tangent's first iterations overshoot deep saturation (ratios > 0.9 while it backtracks), the
    stall-polish engaged the magnetic SECANT operator, and the step then exhausted 100 iterations
    at residual 469 against a 177 start; with the polish withheld the same step converged in 15.
    Here the default jacobian is the exact tangent of a residual whose Newton contracts by only
    ~0.96 per iteration far from the root, so the polish engages; the offered alternative can
    never be accepted by the line search. Held to it, the step exhausts; handed back, the Newton
    that was already on its way finishes.
    """
    from numerics_lab.time_integrators._forms import IntegratorState
    from numerics_lab.time_integrators._integrators import _bdf2_variable_form
    from numerics_lab.time_integrators._step import _solve_step

    class _SlowFarField:
        #: The declared capability `AlternativeJacobianProblem` reads, so the polish can engage.
        has_tangent_jacobian = True

        def __init__(self) -> None:
            self.alternative_calls = 0

        def residual(self, u_arg, u_dot, u_ddot, t):
            return u_arg / (1.0 + u_arg * u_arg) ** 0.24

        def jacobian(self, u_arg, u_dot, u_ddot, t, weights):
            q = 1.0 + u_arg * u_arg
            return np.diag(q**-0.24 - 0.48 * u_arg * u_arg * q**-1.24)

        def jacobian_tangent(self, u_arg, u_dot, u_ddot, t, weights):
            self.alternative_calls += 1
            return -self.jacobian(u_arg, u_dot, u_ddot, t, weights)

    def solve(j, r):  # the seam's solve
        return np.linalg.solve(np.asarray(j), r)

    state = IntegratorState(us=[np.zeros(1), np.zeros(1)], t=0.0)
    form = _bdf2_variable_form(state, 0.1, 0.1, 0.1)
    # Both line searches: the strict one, and the march's own tolerant band (`integrate_transient`
    # accepts a trial up to 1.5x the current residual, so a bad alternative is never CAPPED there --
    # it is accepted while the residual climbs, which is what case_a's t = 4.44 ms step did).
    for band, cap in ((None, 8), (1.5, 4)):
        problem = _SlowFarField()
        _u, converged, iterations, residual, _on_floor = _solve_step(
            problem,
            form,
            np.array([20.0]),
            solve=solve,
            max_newton=100,
            tol=1e-10,
            backtrack_band=band,
            backtrack_cap=cap,
        )
        assert problem.alternative_calls > 0, 'the precondition: the slow far field must engage the polish'
        assert converged, f'band {band}: the step exhausted at residual {residual:g} after {iterations} iterations'


def test_a_discarded_bisection_probe_does_not_count_as_a_nonconverged_STEP(monkeypatch) -> None:
    """``converged`` must mean what it says: every step OF THE SOLUTION met the tolerance.

    The event locator bisects by RE-STEPPING from a checkpoint and throwing the result away
    (``_rollback``). Those probe steps go through the same ``_solve_step`` as a real step, so an
    exhausted probe used to land in ``nonconverged_times`` -- the list ``converged`` is derived
    from. A march whose every LANDED step converged perfectly therefore reported
    ``converged == False``, and no caller could ask the question it actually cares about.

    Measured consequence (srdab_closed_loop, 2026-08-15): 2094 "non-converged" at 100 steps per
    switching period, of which only 239 landed -- 1855 were discarded probes. The tag was ~9x
    pessimistic and read as a broken march.

    The discriminator: a landed step's time is IN ``solution.times``; a discarded probe's is not.
    """

    class _NonlinearRampWithToggle(_RampWithToggle):
        """``u_dot = slope + u^2`` -- nonlinear, so a starved Newton ceiling genuinely exhausts.

        A LINEAR problem is useless here: its single Newton step is exact, nothing ever fails,
        and the assertion passes vacuously without the probe path ever reporting anything.
        """

        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - self.slope - np.asarray(u) ** 2

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real - 2.0 * float(u[0]) * weights.alpha0.real]])

    problem = _NonlinearRampWithToggle(level=0.35)
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    events = import_module('numerics_lab.time_integrators.march')
    solve_step, locate = _step_mod._solve_step, events._locate_bracket
    discarded_probe = False
    probe_failures = []

    def classified_solve(problem, form, *args, **kwargs):
        result = solve_step(problem, form, *args, **kwargs)
        if discarded_probe or form.t_eval >= 0.9:
            # Explicit independent solver verdicts: discarded locator probes
            # fail, as does a non-event fixed step. The physical event landing
            # itself must be accepted; a starved landing cannot fire a switch.
            if discarded_probe:
                probe_failures.append(form.t_eval)
            u, _ok, iterations, residual, floor = result
            return u, False, iterations, residual, floor
        return result

    def classified_locator(lo_t, g_lo, hi_t, g_hi, probe, *, tol):
        def failed_probe(t):
            nonlocal discarded_probe
            discarded_probe = True
            try:
                return probe(t)
            finally:
                discarded_probe = False

        return locate(lo_t, g_lo, hi_t, g_hi, failed_probe, tol=tol)

    monkeypatch.setattr(_step_mod, '_solve_step', classified_solve)
    monkeypatch.setattr(events, '_locate_bracket', classified_locator)
    sol = integrate_transient(
        problem,
        study,
        np.array([0.0]),
        event_fns=[problem.event_fn],
        on_event=lambda _i, _t, _u: problem.toggle(),
        max_newton=100,
        tol=1.0e-12,
        on_nonconverged='label',
    )
    assert problem.toggles == 1, 'the event must actually fire (else the probe path never runs)'
    assert probe_failures, 'the locator must genuinely discard failed solve verdicts'
    assert sol.nonconverged_probe_times
    assert sol.nonconverged_times, 'the starved ceiling must exhaust somewhere (the precondition)'
    landed = set(sol.times)
    stray = [t for t in sol.nonconverged_times if t not in landed]
    assert not stray, (
        f'nonconverged_times must contain only times that LANDED in the solution; '
        f'{len(stray)} of {len(sol.nonconverged_times)} are discarded bisection probes: {stray[:5]}'
    )


def test_an_adaptive_march_does_not_locate_a_switch_from_an_unsolved_probe() -> None:
    """A rejected Newton iterate has no physical event sign for a field-circuit switch."""

    class _UnsolvableBetweenGridPoints(_RampWithToggle):
        def residual(self, u, u_dot, u_ddot, t):
            if abs(t / 0.3 - round(t / 0.3)) > 1e-8:
                return np.array([1.0])
            return super().residual(u, u_dot, u_ddot, t)

    problem = _UnsolvableBetweenGridPoints(level=1.0)
    study = TimeWindow(t0=0.0, t1=1.5, dt=0.3, integrator=IntegratorKind.BDF2)
    with pytest.raises(RuntimeError, match='event probe.*converge'):
        integrate_transient(
            problem,
            study,
            np.array([0.0]),
            event_fns=(problem.event_fn,),
            on_event=lambda _i, _t, _u: problem.toggle(),
            adaptive=True,
            max_newton=2,
        )


def test_an_adaptive_march_does_not_fire_a_switch_from_an_unsolved_landing() -> None:
    """Even an exact root time cannot authorize a switch if its state did not solve."""

    class _UnsolvableAtCrossing(_RampWithToggle):
        def residual(self, u, u_dot, u_ddot, t):
            if abs(t - 1.0) < 1e-8:
                return np.array([1.0])
            return super().residual(u, u_dot, u_ddot, t)

    problem = _UnsolvableAtCrossing(level=1.0)
    study = TimeWindow(t0=0.0, t1=1.5, dt=0.3, integrator=IntegratorKind.BDF2)
    with (
        patch('numerics_lab.time_integrators.march._locate_bracket', return_value=(1.0, 1.0)),
        pytest.raises(RuntimeError, match='event landing.*converge'),
    ):
        integrate_transient(
            problem,
            study,
            np.array([0.0]),
            event_fns=(problem.event_fn,),
            on_event=lambda _i, _t, _u: problem.toggle(),
            adaptive=True,
            max_newton=2,
        )
    assert problem.toggles == 0


def test_the_event_root_is_solved_once_not_once_for_ranking_and_again_for_landing() -> None:
    """The ranked candidates need times only; a second root solve changes no decision."""

    class _CountRoot(_RampWithToggle):
        def __init__(self) -> None:
            super().__init__(level=1.0)
            self.root_residuals = 0

        def residual(self, u, u_dot, u_ddot, t):
            if abs(t - 1.0) < 1e-8:
                self.root_residuals += 1
            return super().residual(u, u_dot, u_ddot, t)

    problem = _CountRoot()
    study = TimeWindow(t0=0.0, t1=1.5, dt=0.3, integrator=IntegratorKind.BDF2)
    with patch('numerics_lab.time_integrators.march._locate_bracket', return_value=(1.0, 1.0)):
        solution = integrate_transient(
            problem,
            study,
            np.array([0.0]),
            event_fns=(problem.event_fn,),
            on_event=lambda _i, _t, _u: problem.toggle(),
        )
    assert len(solution.events) == 1
    assert problem.root_residuals == 2, 'a redundant candidate solve reassembled the field at the same event root'


def test_a_polish_that_only_creeps_linearly_hands_back_to_the_exact_newton() -> None:
    """The polish must BEAT the Newton it replaced, not merely lower the residual (case_i, 2026-10-17).

    MEASURED on `case_i --scenario current_loop` at t = 12.4 ms: the exact Newton's global phase into
    saturation ran at ratios 0.995 / 0.987, the polish engaged the magnetic secant (5.6e-3 off the exact
    Jacobian), and the secant then lowered the residual by ~0.88 every iteration -- never capped, never
    raised -- until the 50-iteration ceiling, while one exact step from that iterate took 6.7e-5 -> 4.8e-10.
    Here the alternative is ``8 J``: a chord that contracts by exactly 7/8 per iteration, forever.
    """
    from numerics_lab.time_integrators._forms import IntegratorState
    from numerics_lab.time_integrators._integrators import _bdf2_variable_form
    from numerics_lab.time_integrators._step import HELD_STALL_RATIO, _solve_step

    class _CreepingSecant:
        has_tangent_jacobian = True

        def __init__(self) -> None:
            self.alternative_calls = 0

        def residual(self, u_arg, u_dot, u_ddot, t):
            return u_arg / (1.0 + u_arg * u_arg) ** 0.24

        def jacobian(self, u_arg, u_dot, u_ddot, t, weights):
            q = 1.0 + u_arg * u_arg
            return np.diag(q**-0.24 - 0.48 * u_arg * u_arg * q**-1.24)

        def jacobian_tangent(self, u_arg, u_dot, u_ddot, t, weights):
            self.alternative_calls += 1
            return 8.0 * self.jacobian(u_arg, u_dot, u_ddot, t, weights)

    def solve(j, r):
        return np.linalg.solve(np.asarray(j), r)

    state = IntegratorState(us=[np.zeros(1), np.zeros(1)], t=0.0)
    form = _bdf2_variable_form(state, 0.1, 0.1, 0.1)
    problem = _CreepingSecant()
    _u, converged, iterations, residual, _on_floor = _solve_step(
        problem, form, np.array([20.0]), solve=solve, max_newton=50, tol=1e-10
    )
    assert problem.alternative_calls > 0, 'the precondition: the slow global phase must engage the polish'
    # PLANTED CONTROL: held to the 7/8 chord the step would need log(1e-10)/log(7/8) > 170 iterations.
    assert HELD_STALL_RATIO < 7.0 / 8.0 and math.log(1e-10) / math.log(7.0 / 8.0) > 50
    assert converged, f'the step exhausted at residual {residual:g} after {iterations} iterations'
    assert problem.alternative_calls <= 2
