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
    u_step = _solve_step(
        problem, form, state.last_u, solve=lambda j, r: np.linalg.solve(np.atleast_2d(j), r), max_newton=1, tol=1e-12
    )
    assert abs(float(u_step[0]) - 0.28125) <= 1e-12

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
