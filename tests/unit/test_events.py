"""Event location + checkpoint/rollback in the time march -- the physics-blind W2 seam.

Roadmap W2: the event locator and checkpoint/rollback belong IN gauss (a root-find on an OPAQUE
scalar event function plus a rollback of OPAQUE state, behind the existing ``SemiDiscreteProblem``
seam); what a zero-crossing MEANS stays in the caller. These tests pin that contract on a problem
that is deliberately not a circuit: a ramp whose slope the event callback flips, so a march that
merely DETECTS the event without rolling back to it overshoots, and only a true restart-from-event
turns at the located instant.
"""
# ruff: noqa: PLC0415 -- test-module lazy imports: the module under test is imported inside
# the test/fixture, keeping collection cheap and loading platform-conditional deps (win32com,
# OCP, FEMM) only on the path that needs them. Hoist a file's imports and DELETE this header;
# the population is ratcheted by tests/unit/scripts/test_waiver_principle.py.

from __future__ import annotations

import itertools

import numpy as np
import pytest

from motronics.contracts.study_spec import IntegratorKind, TransientStudy
from motronics.gauss.time_integrators import integrate_transient
from motronics.gauss.time_integrators.events import integrate_transient_with_events


class _RampWithToggle:
    """``u_dot = slope`` with a callback-flipped slope -- the minimal stateful ``SemiDiscreteProblem``.

    The event ``u - level`` crosses zero at a known time (``level`` with slope +1 from ``u(0)=0``);
    the callback negates the slope, so the trajectory is a triangle turning exactly at ``level``.
    """

    def __init__(self, level: float) -> None:
        self.slope = 1.0
        self.level = level
        self.toggles = 0

    def residual(self, u, u_dot, u_ddot, t):
        return np.asarray(u_dot) - self.slope

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.array([[weights.alpha1.real]])

    def event_fn(self, u, t):
        # One-sided by construction: once toggled, u falls away from the level, so the same
        # function cannot re-trigger -- the state-dependent event is the laplace switch's shape.
        return float(u[0]) - self.level if self.slope > 0.0 else -1.0

    def toggle(self):
        self.slope = -1.0
        self.toggles += 1


def test_an_event_inside_a_quadratic_ramp_costs_no_accuracy_at_all() -> None:
    """The variable-step BDF2 form introduces no error of its own -- the trap it closes.

    Fixed-weight BDF2 assumes uniform history spacing; an event-landing step breaks that, and the
    derivative coefficient is then wrong by an O(1) factor ``(3 dt' - h)/(2 dt')`` -- measured in
    development as a bisection converging to the BRACKET EDGE (a 1.0 s crossing "located" at
    0.9 s). The variable-spacing form is the derivative of the quadratic interpolant itself: given
    an exact history it steps a quadratic EXACTLY (the first assertion below), and a march through
    an event adds NOTHING to the plain march's own truncation error (the second) -- the landing
    and remainder steps are as accurate as the uniform steps around them.
    """

    class _Quadratic:
        """``u_dot = t`` (so ``u = t^2/2``); firing records only, it does not change the problem."""

        def __init__(self) -> None:
            self.fired = False

        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - t

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

        def event_fn(self, u, t):
            return -1.0 if self.fired else float(u[0]) - 0.32

    problem = _Quadratic()
    study = TransientStudy(t0=0.0, t1=2.0, dt=0.3, integrator=IntegratorKind.BDF2)
    # The form itself, on an exact history, steps a quadratic exactly (direct check, u(0.75)).
    from motronics.gauss.time_integrators import IntegratorState, _solve_step
    from motronics.gauss.time_integrators.events import _bdf2_variable_form

    state = IntegratorState(us=[np.array([0.045]), np.array([0.18])], t=0.6)
    form = _bdf2_variable_form(state, 0.15, 0.3, 0.75)
    u_step, converged, iterations, residual = _solve_step(
        problem, form, state.last_u, solve=lambda j, r: np.linalg.solve(np.atleast_2d(j), r), max_newton=1, tol=1e-12
    )
    assert abs(float(u_step[0]) - 0.28125) <= 1e-12
    assert converged and iterations == 1 and residual <= 1e-12  # the convergence contract

    # A mid-cell event: the event march's deviation from the TRUE solution stays at the plain
    # march's own truncation scale -- no O(1) corruption from the landing/remainder steps. The two
    # marches solve slightly different discrete recurrences once the grid shifts, so bit-equality
    # is NOT the expectation; same-scale true error is.
    plain = integrate_transient(problem, study, np.array([0.0]))
    eventful = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: setattr(problem, 'fired', True),
    )
    assert len(eventful.events) == 1

    def true_err(sol):
        return max(abs(float(u[0]) - 0.5 * t * t) for t, u in zip(sol.times, sol.states, strict=True))

    err_plain, err_eventful = true_err(plain), true_err(eventful)
    assert err_eventful <= 2.0 * err_plain, (
        f'event march true error {err_eventful:g} vs plain {err_plain:g} -- the landing is corrupting '
        'the trajectory (fixed BDF2 weights against a nonuniform history), not just reshuffling it'
    )

    # An event landing exactly ON the grid: identical steps, identical trajectory to round-off.
    on_grid = _Quadratic()
    on_grid.event_fn = lambda u, _t: -1.0 if on_grid.fired else float(u[0]) - 0.24  # marched u(0.6) = 0.24
    grid_march = integrate_transient_with_events(
        on_grid,
        study,
        np.array([0.0]),
        event_fns=(on_grid.event_fn,),
        on_event=lambda _i, _t, _u: setattr(on_grid, 'fired', True),
    )
    assert len(grid_march.events) == 1
    # Same grid, up to the locator's own tolerance: an event EXACTLY on a grid point
    # used to land bit-exactly (the `g_new == 0.0` shortcut), but numpy 2.5.1's macOS
    # wheel (Apple Accelerate) computes the 1x1 linear solve as b*(1/a) -- 1 ulp high
    # (measured 2026-08-07: t_event = 0.6 - 1.4305e-07), so the bisection branch runs
    # and lands within its documented bracket_tol = dt*1e-6 = 3e-7. The assertion is
    # the locator's contract, not the former bonus.
    assert all(abs(a - b) <= study.dt * 1e-6 for a, b in zip(grid_march.times, plain.times, strict=True))
    # Bit-identity is NOT the expectation even on-grid: the restart at the event bootstraps one
    # BDF1 step, whose O(dt^2) local error is real but bounded -- so the assertion is the
    # true-error scale, not equality.
    err_grid = true_err(grid_march)
    assert err_grid <= 2.0 * err_plain, (
        f'on-grid event march true error {err_grid:g} vs plain {err_plain:g}: the restart costs '
        'one bounded BDF1 step, not a trajectory corruption'
    )


def test_a_march_without_events_overshoots_the_turning_level() -> None:
    """The baseline the locator must beat: on a fixed grid the ramp crosses ``level`` mid-cell."""
    problem = _RampWithToggle(level=1.0)
    study = TransientStudy(t0=0.0, t1=2.0, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(problem, study, np.array([0.0]))
    assert max(float(u[0]) for u in solution.states) > problem.level, (
        'the grid does not land on the event -- a plain march sails past the turning point'
    )


def test_the_march_turns_AT_the_located_event_not_at_the_next_grid_point() -> None:
    """Rollback + restart: the peak lands on ``level`` to the locator's tolerance, not on the grid."""
    problem = _RampWithToggle(level=1.0)
    study = TransientStudy(t0=0.0, t1=2.0, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: problem.toggle(),
    )
    peak = max(float(u[0]) for u in solution.states)
    assert problem.toggles == 1
    assert abs(peak - problem.level) <= 1e-6, (
        f'turning peak {peak} -- a march that does not roll back to the event overshoots to the '
        f'next grid point (1.2 on this grid)'
    )
    (index, t_event), *rest = solution.events
    assert index == 0 and not rest
    assert abs(t_event - 1.0) <= 1e-6, f'located event at {t_event}, the crossing is at 1.0'
    # The kink itself must not be rounded: past the toggle the slope is -1, and dragging the
    # pre-event history into the next step would lift every later point off the true triangle
    # (measured +9.5% on the first post-event step before the history restart). The restart makes
    # the piecewise-linear trajectory EXACT.
    for t, u in zip(solution.times, solution.states, strict=True):
        true = t if t <= t_event else 2.0 * t_event - t
        assert abs(float(u[0]) - true) <= 1e-9, f'kink rounded: u({t}) = {u[0]}, true {true}'


def test_the_event_time_is_located_far_below_the_step_size() -> None:
    """The locator is a ROOT-FIND, not a grid refinement: dt=0.3, crossing found to ~1e-9."""
    problem = _RampWithToggle(level=1.0)
    study = TransientStudy(t0=0.0, t1=1.5, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: problem.toggle(),
        time_tol=1e-9,
    )
    ((_, t_event),) = solution.events
    assert abs(t_event - 1.0) <= 1e-8


def test_a_circuit_with_no_event_functions_marches_identically_to_the_plain_driver() -> None:
    """The events path must not tax or perturb a march with nothing to locate."""
    problem = _RampWithToggle(level=1.0)
    study = TransientStudy(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    plain = integrate_transient(problem, study, np.array([0.0]))
    eventful = integrate_transient_with_events(problem, study, np.array([0.0]))
    assert eventful.events == []
    assert eventful.times == plain.times
    for a, b in zip(eventful.states, plain.states, strict=True):
        np.testing.assert_array_equal(a, b)


def test_an_event_landing_exactly_on_a_grid_point_is_still_handled_once() -> None:
    """``t* = t_target`` exactly: no bisection, no double-toggle, no zero-length step."""
    problem = _RampWithToggle(level=0.6)  # crossed exactly at a multiple of dt
    study = TransientStudy(t0=0.0, t1=1.2, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: problem.toggle(),
    )
    assert problem.toggles == 1
    ((_, t_event),) = solution.events
    # Locator tolerance, not bit-exactness: the exact-grid-point landing was a bonus
    # of the old numpy's correctly-rounded 1x1 solve; numpy 2.5.1/Accelerate solves it
    # as b*(1/a) (1 ulp high), so the bisection branch lands within its documented
    # bracket_tol = 3e-7 (measured 2026-08-07: t_event - 0.6 = -1.4305e-07).
    assert abs(t_event - 0.6) <= 1e-6


def test_two_events_crossing_at_the_same_instant_fire_together() -> None:
    """Two functions rooting at the SAME instant: BOTH fire at it.

    The earliest-only rule drops the second, and -- the mechanism that makes it a STUCK device
    rather than a delayed event -- each function reads only its OWN state (the H-bridge's
    complementary pair: the S4's event function reads the S4's own flag, which the S1's
    callback does not change). The dropped function's sign at the event (``g == 0`` reads as
    the far side) therefore matches its post-event sign, so the re-detection never sees it.
    Measured on hbridge_deadtime 2026-08-07: the S4's OFF edge at 37.4 us was lost and the
    switch stayed ON through the whole dead-time, changing the circuit.
    """

    class _TwinIndependentToggles:
        """``u_dot = 1`` with two identical level events; each callback flips its OWN flag."""

        def __init__(self) -> None:
            self.flag_a = False
            self.flag_b = False
            self.level = 0.2

        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

        def fn_a(self, u, t):
            return float(u[0]) - self.level if not self.flag_a else -1.0

        def fn_b(self, u, t):
            return float(u[0]) - self.level if not self.flag_b else -1.0

        def on_a(self, *_args):
            self.flag_a = True

        def on_b(self, *_args):
            self.flag_b = True

    problem = _TwinIndependentToggles()
    study = TransientStudy(t0=0.0, t1=1.0, dt=1.0, integrator=IntegratorKind.BDF2)  # ONE cell, both root at 0.2
    solution = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.fn_a, problem.fn_b),
        on_event=lambda i, _t, _u: (problem.on_a if i == 0 else problem.on_b)(),
    )
    assert problem.flag_a and problem.flag_b, f'both callbacks must fire, got a={problem.flag_a} b={problem.flag_b}'
    times = sorted(t for _, t in solution.events)
    assert len(times) == 2, f'expected two events, got {solution.events}'
    assert abs(times[0] - times[1]) <= 1e-9, f'simultaneous events not co-located: {times}'
    assert abs(times[0] - 0.2) <= 1e-6


def test_two_events_in_one_step_are_handled_earliest_first() -> None:
    """Two crossings inside one nominal cell: the EARLIER is located and handled first."""

    class _TwoLevels(_RampWithToggle):
        def __init__(self) -> None:
            super().__init__(level=0.2)
            self.level_hi = 0.8

        def hi_fn(self, u, t):
            return float(u[0]) - self.level_hi if self.slope > 0.0 else -1.0

    problem = _TwoLevels()
    order: list[int] = []
    study = TransientStudy(t0=0.0, t1=1.0, dt=1.0, integrator=IntegratorKind.BDF2)  # ONE cell, both crossings inside
    solution = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn, problem.hi_fn),
        on_event=lambda i, _t, _u: order.append(i),
    )
    assert order == [0, 1], f'events handled out of order: {order}'
    times = [t for _, t in solution.events]
    assert abs(times[0] - 0.2) <= 1e-6 and abs(times[1] - 0.8) <= 1e-6


def test_a_checkpoint_rolls_the_solution_back_without_aliasing() -> None:
    """States appended past the event during probing are retracted, not left in the output."""
    problem = _RampWithToggle(level=1.0)
    study = TransientStudy(t0=0.0, t1=2.0, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: problem.toggle(),
    )
    # Times strictly increase -- a leaked probe point would sit between the bracket ends.
    ts = solution.times
    assert all(b > a for a, b in itertools.pairwise(ts)), f'non-monotone march times: {ts}'
    # And the event point itself is IN the march: the restart is from the located state.
    ((_, t_event),) = solution.events
    assert any(abs(t - t_event) <= 1e-12 for t in ts)


def test_a_second_crossing_after_the_toggle_does_not_retrigger() -> None:
    """Hysteresis-free but state-flipped: the same fn must not fire again on the way back down."""
    problem = _RampWithToggle(level=1.0)
    study = TransientStudy(t0=0.0, t1=3.0, dt=0.3, integrator=IntegratorKind.BDF2)
    integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: problem.toggle(),
    )
    assert problem.toggles == 1


def test_an_event_function_that_never_crosses_costs_no_extra_points() -> None:
    problem = _RampWithToggle(level=99.0)
    study = TransientStudy(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    solution = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: problem.toggle(),
    )
    assert solution.events == []
    assert len(solution.times) == 11


def test_an_event_within_time_tol_of_the_checkpoint_is_handled_there_not_bisected_forever() -> None:
    """A crossing closer to the step start than the locator tolerance: no zero-length steps, no hang."""
    problem = _RampWithToggle(level=1.0e-4)
    study = TransientStudy(t0=0.0, t1=1.0, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: problem.toggle(),
        time_tol=1.0e-3,  # the crossing at 1e-4 sits INSIDE the tolerance of the t=0 checkpoint
    )
    assert problem.toggles == 1
    ((_, t_event),) = solution.events
    assert t_event <= 1.0e-3
    ts = solution.times
    assert all(b > a for a, b in itertools.pairwise(ts))


def test_time_breakpoints_land_on_the_grid_and_do_not_change_a_breakpointless_march() -> None:
    """``time_breakpoints`` pins the source edges onto the marched grid -- the W1b seam.

    A breakpoint is a KNOWN non-smooth instant (a PWM edge), so the march clips the cell to it
    instead of sampling past it: every breakpoint appears in ``solution.times``, the plain march
    with ``time_breakpoints=()`` is bit-identical to not passing the argument at all, and a
    breakpoint list that is not strictly inside the study window is refused loudly.
    """

    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    study = TransientStudy(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    problem = _Linear()
    solution = integrate_transient_with_events(
        problem, study, np.array([0.0]), time_breakpoints=(0.05, 0.27, 0.8)
    )
    ts = np.asarray(solution.times)
    for bp in (0.05, 0.27, 0.8):
        assert np.isclose(ts, bp).any(), f'breakpoint {bp} not on the grid'

    plain = integrate_transient_with_events(problem, study, np.array([0.0]))
    # A clipped cell replaces its nominal target: 0.1 is displaced by the 0.05 breakpoint.
    assert 0.1 not in ts

    no_bps = integrate_transient_with_events(problem, study, np.array([0.0]), time_breakpoints=())
    assert np.array_equal(no_bps.times, plain.times)


def test_time_breakpoints_and_state_events_coexist() -> None:
    """A source edge (time breakpoint) must not disturb the state-event locator, and vice versa.

    The laplace shape this pins: a PWM source edge at ``t = 0.3`` plus a threshold switch whose
    crossing ``u = 0.5`` lies on the nominal grid. Both land exactly -- the breakpoint on the
    grid, the event by bisection at ``0.5`` -- and a callback-flipping event still fires once.
    (A breakpoint does NOT change what BDF2 integrates on the edge cell: the endpoint-evaluated
    residual reads the post-edge value, so the edge cell carries the jump's O(h) share -- the
    grid alignment, not the integral, is what the seam guarantees.)
    """
    problem = _RampWithToggle(level=0.5)
    study = TransientStudy(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    solution = integrate_transient_with_events(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: problem.toggle(),
        time_breakpoints=(0.3,),
    )
    ts = np.asarray(solution.times)
    assert np.isclose(ts, 0.3).any(), 'the time breakpoint must land on the grid'
    assert problem.toggles == 1
    ((_, t_event),) = solution.events
    assert np.isclose(t_event, 0.5, atol=1e-9), 'the state event is located independently of the breakpoint'


def test_time_breakpoints_outside_the_study_window_are_refused() -> None:
    study = TransientStudy(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)

    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    with pytest.raises(ValueError, match='study window'):
        integrate_transient_with_events(_Linear(), study, np.array([0.0]), time_breakpoints=(1.5,))
    with pytest.raises(ValueError, match='strictly increasing'):
        integrate_transient_with_events(_Linear(), study, np.array([0.0]), time_breakpoints=(0.2, 0.2))


def test_thousands_of_time_breakpoints_all_land_on_the_grid() -> None:
    """SR-014: the breakpoint queue is consumed by a cursor, not ``pop(0)`` -- a PWM march's
    thousands of edges must all land (behavior identical to the list queue, pinned here at
    the scale that would expose the per-edge O(n) list shift as O(n^2)).
    """
    n_edges = 4000  # a 20 kHz carrier over 100 ms -- the SR-014 scale
    t1 = 0.1
    study = TransientStudy(t0=0.0, t1=t1, dt=1.0e-5, integrator=IntegratorKind.BDF2)
    bps = tuple(np.linspace(t1 / n_edges, t1, n_edges))  # every 2.5 cells, off-grid

    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    solution = integrate_transient_with_events(_Linear(), study, np.array([0.0]), time_breakpoints=bps)
    ts = np.asarray(solution.times)
    for bp in (bps[0], bps[1000], bps[3000], bps[-1]):
        assert np.isclose(ts, bp).any(), f'breakpoint {bp} not on the grid'
    us = np.asarray([float(s[0]) for s in solution.states])
    assert np.max(np.abs(us - ts)) < 1.0e-12, f'time base drifted: max |u - t| = {np.max(np.abs(us - ts))}'
    assert ts[-1] == pytest.approx(t1)


def test_breakpoint_clipping_never_drifts_the_time_base() -> None:
    """SR-022 regression: the clipped march's states ride their labels EXACTLY.

    The W1b round shipped a march whose nominal targets drifted ahead of the physics after
    the first clipped cell (a linear ramp with a breakpoint at 0.05, dt=0.1: times carried
    [0, 0.05, 0.2, ...] while the states were [0, 0.05, 0.15, ...] -- max |u - t| = 0.08, and
    every switched-circuit transient after the first edge was silently wrong). The grid must
    re-anchor at each breakpoint: u(t) == t to the ulp, all the way to t1.
    """

    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    study = TransientStudy(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    solution = integrate_transient_with_events(
        _Linear(), study, np.array([0.0]), time_breakpoints=(0.05, 0.27, 0.8)
    )
    ts = np.asarray(solution.times)
    us = np.asarray([float(s[0]) for s in solution.states])
    assert np.max(np.abs(us - ts)) < 1.0e-12, f'time base drifted: max |u - t| = {np.max(np.abs(us - ts))}'
    assert ts[-1] == pytest.approx(1.0)  # the march ends exactly at t1


def test_a_final_cell_breakpoint_is_a_partial_step_not_a_full_dt() -> None:
    """SR-025 regression: a breakpoint inside the LAST cell clips to t1 with a PARTIAL step.

    The round-1 re-anchor marched the final clipped cell a full dt while labelling it t1
    (measured max |u - t| = 0.05 on a linear ramp) -- the regression test passed only because
    its last breakpoint landed exactly on t1. Here the breakpoint sits at 0.95 (dt = 0.1):
    the march must land 0.95 then finish with a 0.05 step to t1, riding its labels exactly.
    """
    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    study = TransientStudy(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    solution = integrate_transient_with_events(
        _Linear(), study, np.array([0.0]), time_breakpoints=(0.95,)
    )
    ts = np.asarray(solution.times)
    us = np.asarray([float(s[0]) for s in solution.states])
    assert np.max(np.abs(us - ts)) < 1.0e-12
    assert ts[-1] == pytest.approx(1.0)
    assert 0.95 in ts


def test_a_non_integral_window_is_bit_identical_to_the_plain_march() -> None:
    """SR-038 regression: without breakpoints the eventful march takes EXACTLY the plain
    march's steps -- truncating at its fixed count (round((t1-t0)/dt)) when t1 is not a
    multiple of dt, never a full step past t1. The round-2 adjudication wrongly claimed
    integrate_transient lands past t1; the eventful march did (plain [0,.3,.6,.9] vs
    eventful [0,.3,.6,.9,1.2] for t1=1.0, dt=0.3) and the bit-identity promise broke."""

    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    study = TransientStudy(t0=0.0, t1=1.0, dt=0.3, integrator=IntegratorKind.BDF2)
    plain = integrate_transient(_Linear(), study, np.array([0.0]))
    eventful = integrate_transient_with_events(_Linear(), study, np.array([0.0]))
    assert np.array_equal(plain.times, eventful.times)
    assert np.array_equal(np.asarray(plain.states), np.asarray(eventful.states))
    assert eventful.times[-1] == pytest.approx(0.9)  # the truncating count, not 1.2
