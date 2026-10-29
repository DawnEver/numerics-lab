"""Event location + checkpoint/rollback in the time march -- the physics-blind W2 seam.

Roadmap W2: the event locator and checkpoint/rollback belong IN gauss (a root-find on an OPAQUE
scalar event function plus a rollback of OPAQUE state, behind the existing ``SemiDiscreteProblem``
seam); what a zero-crossing MEANS stays in the caller. These tests pin that contract on a problem
that is deliberately not a circuit: a ramp whose slope the event callback flips, so a march that
merely DETECTS the event without rolling back to it overshoots, and only a true restart-from-event
turns at the located instant.

Split from the original ``test_events.py``: this file carries the locator/rollback/ordering/
suppression behaviors on ``_RampWithToggle``; step-control/accuracy and solver-internals tests
live in the sibling ``test_events_step_control.py`` / ``test_events_solver_internals.py``, and
``time_breakpoints`` behaviors live in ``test_events_breakpoints.py``.
"""

# the test/fixture, keeping collection cheap and loading platform-conditional deps (win32com,
# OCP, FEMM) only on the path that needs them. Hoist a file's imports and DELETE this header;
# the population is ratcheted by tests/architecture/ratchets/test_suppression_ratchet.py.

from __future__ import annotations

import itertools

import numpy as np
import pytest
from tests.unit._helpers import _RampWithToggle

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient


@pytest.mark.parametrize('time_tol', [0.0, -1e-9, np.nan, np.inf])
def test_event_bracket_width_must_be_finite_and_positive(time_tol: float) -> None:
    """A zero-width event bracket cannot terminate a switched field-circuit march."""
    problem = _RampWithToggle(level=1.0)
    study = TimeWindow(t0=0.0, t1=2.0, dt=0.3, integrator=IntegratorKind.BDF2)
    with pytest.raises(ValueError, match='time_tol.*finite and positive'):
        integrate_transient(problem, study, np.array([0.0]), event_fns=(problem.event_fn,), time_tol=time_tol)


def test_a_march_without_events_overshoots_the_turning_level() -> None:
    """The baseline the locator must beat: on a fixed grid the ramp crosses ``level`` mid-cell."""
    problem = _RampWithToggle(level=1.0)
    study = TimeWindow(t0=0.0, t1=2.0, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(problem, study, np.array([0.0]))
    assert max(float(u[0]) for u in solution.states) > problem.level, (
        'the grid does not land on the event -- a plain march sails past the turning point'
    )


def test_the_march_turns_AT_the_located_event_not_at_the_next_grid_point() -> None:
    """Rollback + restart: the peak lands on ``level`` to the locator's tolerance, not on the grid."""
    problem = _RampWithToggle(level=1.0)
    study = TimeWindow(t0=0.0, t1=2.0, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(
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
    study = TimeWindow(t0=0.0, t1=1.5, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(
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
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    plain = integrate_transient(problem, study, np.array([0.0]))
    eventful = integrate_transient(problem, study, np.array([0.0]))
    assert eventful.events == []
    assert eventful.times == plain.times
    for a, b in zip(eventful.states, plain.states, strict=True):
        np.testing.assert_array_equal(a, b)


def test_an_event_landing_exactly_on_a_grid_point_is_still_handled_once() -> None:
    """``t* = t_target`` exactly: no bisection, no double-toggle, no zero-length step."""
    problem = _RampWithToggle(level=0.6)  # crossed exactly at a multiple of dt
    study = TimeWindow(t0=0.0, t1=1.2, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(
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
    study = TimeWindow(t0=0.0, t1=1.0, dt=1.0, integrator=IntegratorKind.BDF2)  # ONE cell, both root at 0.2
    solution = integrate_transient(
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
    study = TimeWindow(t0=0.0, t1=1.0, dt=1.0, integrator=IntegratorKind.BDF2)  # ONE cell, both crossings inside
    solution = integrate_transient(
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
    study = TimeWindow(t0=0.0, t1=2.0, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(
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


def test_a_pure_time_sawtooth_event_fires_every_step() -> None:
    """MANUAL-635: a pure-time event -- the continuous-PI's sample clock, ``t - next``
    with ``next`` advancing per firing -- must fire at EVERY cell target. The old
    suppression measured the distance from the cell START, which after an event sits
    exactly at the last firing, so the dense samples were suppressed as duplicates
    forever (measured: the sampling_divisor = 200 PI fired exactly once). The
    suppression now measures to the cell TARGET (a genuinely later crossing reads its
    true distance) and drops the re-detected landings by their located INSTANT.
    """
    problem = _RampWithToggle(level=99.0)  # the ramp never crosses -- only the clock fires

    class _Clock:
        def __init__(self, dt: float) -> None:
            self.dt = dt
            self.next = dt

        def __call__(self, u, t) -> float:
            return float(t) - self.next

        def advance(self) -> None:
            self.next += self.dt

    clock = _Clock(0.3)
    study = TimeWindow(t0=0.0, t1=0.9, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(
        problem,
        study,
        np.array([0.0]),
        event_fns=(clock,),
        on_event=lambda _i, _t, _u: clock.advance(),
    )
    assert len(solution.events) == 3, f'every step must fire the clock, got {solution.events}'


def test_a_second_crossing_after_the_toggle_does_not_retrigger() -> None:
    """Hysteresis-free but state-flipped: the same fn must not fire again on the way back down."""
    problem = _RampWithToggle(level=1.0)
    study = TimeWindow(t0=0.0, t1=3.0, dt=0.3, integrator=IntegratorKind.BDF2)
    integrate_transient(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: problem.toggle(),
    )
    assert problem.toggles == 1


def test_an_event_function_that_never_crosses_costs_no_extra_points() -> None:
    problem = _RampWithToggle(level=99.0)
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(
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
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(
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


class _CountingRamp(_RampWithToggle):
    """The ramp, recording every instant the integrator STEPS to -- each is one full field solve."""

    def __init__(self, level: float) -> None:
        super().__init__(level)
        self.instants: set[float] = set()

    def residual(self, u, u_dot, u_ddot, t):
        self.instants.add(float(t))
        return super().residual(u, u_dot, u_ddot, t)


def test_a_crossing_is_located_in_few_re_steps_not_one_per_halving() -> None:
    """A locator probe is a whole re-step, which on a field march is a whole nonlinear FEA solve.

    MEASURED 2026-09-25: bisection to ``dt * 1e-6`` spent ~21 re-steps per crossing, ~80 s each on
    case_p's chopper, so ONE switch event cost ~25 min. A curved event function (``u^3``)
    keeps a secant locator honest: it cannot land exactly on the first probe, and it must still
    converge to the SAME bracket width the bisection did.
    """
    problem = _CountingRamp(level=0.5)
    study = TimeWindow(t0=0.0, t1=0.9, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(
        problem,
        study,
        np.array([0.0]),
        event_fns=(lambda u, _t: float(u[0]) ** 3 - problem.level if problem.slope > 0.0 else -1.0,),
        on_event=lambda _i, _t, _u: problem.toggle(),
    )
    ((_, t_event),) = solution.events
    assert abs(t_event - 0.5 ** (1.0 / 3.0)) <= 0.3e-6
    grid = {0.3, 0.6, 0.9}
    probes = {t for t in problem.instants if min(abs(t - g) for g in grid) > 1e-12}
    assert len(probes) <= 10, f'{len(probes)} distinct re-step instants to locate one crossing'
