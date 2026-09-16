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

import numpy as np
import pytest
from tests.unit.gauss._helpers import _RampWithToggle

from motronics.contracts.spec.study_spec import IntegratorKind, TransientStudy
from motronics.gauss.time_integrators.events import integrate_transient_with_events


def test_the_relative_tolerance_labels_a_scale_stalled_step_converged() -> None:
    """MANUAL-628: the per-step Newton's absolute tol (1e-12) is unreachable when the
    residual's scale is huge -- the SRDAB closed loop's switch ramp: the KCL rows carry
    ~5400 A (270 V / 0.05 ohm), so a residual of 1.1e-1 is 2e-5 RELATIVE, and the Newton
    stalls there by SCALE, not by error. The opt-in ``tol_rel`` accepts a step whose
    residual dropped below ``tol_rel * r0`` (r0 = the first iteration's norm): a stall
    at 1e-1 from a 1e3-scale start is a 1e-4-relative residual -- converged with
    tol_rel = 1e-4, exhausted without it.
    """
    from motronics.gauss.time_integrators import IntegratorState, _solve_step
    from motronics.gauss.time_integrators.events import _bdf2_variable_form

    class _Stalled:
        """A residual that reads 1e3 on the first evaluation and 1e-1 forever after --
        the Newton stalls by scale (the solve is an identity, the residual never drops
        further).
        """

        def residual(self, u_arg, u_dot, u_ddot, t):
            self.calls = getattr(self, 'calls', 0) + 1
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
    from motronics.gauss.time_integrators import IntegratorState, _solve_step
    from motronics.gauss.time_integrators.events import _bdf2_variable_form

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
    overshoot_solve = lambda _j, r: 10.0 * r  # noqa: E731 -- the seam's 10x overstep

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
    from motronics.gauss.time_integrators import IntegratorState, _solve_step
    from motronics.gauss.time_integrators.events import _bdf2_variable_form

    class _CouplingChord:
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

        def jacobian_tangent(self, u_arg, u_dot, u_ddot, t, weights):
            self.tangent_calls += 1
            u1, _u2 = u_arg
            d1 = self.c / (1.0 + self.m * u1) ** 2
            return np.array([[d1 + self.G, -self.G], [-self.G, self.joff + self.G]])

    class _ChordOnly(_CouplingChord):
        """No jacobian_tangent seam (the attribute shadowed to None) -- the pure chord
        Newton, the pre-polish behavior.
        """

        jacobian_tangent = None  # type: ignore[assignment] -- the seam's absence

    solve = lambda j, r: np.linalg.solve(np.asarray(j), r)  # noqa: E731 -- the seam's solve
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


def test_a_discarded_bisection_probe_does_not_count_as_a_nonconverged_STEP() -> None:
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
    study = TransientStudy(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    # A starved ceiling on a nonlinear problem: steps DO exhaust, so the bisection's discarded
    # probes exhaust too -- which is the situation the tag has to classify correctly.
    sol = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=[problem.event_fn],
        on_event=lambda _i, _t, _u: problem.toggle(),
        max_newton=1,
        tol=1.0e-12,
    )
    assert problem.toggles == 1, 'the event must actually fire (else the probe path never runs)'
    assert sol.nonconverged_times, 'the starved ceiling must exhaust somewhere (the precondition)'
    landed = set(sol.times)
    stray = [t for t in sol.nonconverged_times if t not in landed]
    assert not stray, (
        f'nonconverged_times must contain only times that LANDED in the solution; '
        f'{len(stray)} of {len(sol.nonconverged_times)} are discarded bisection probes: {stray[:5]}'
    )
