"""``time_breakpoints`` -- the W1b seam pinning source edges onto the marched grid.

Split from the original ``test_events.py``: locator/rollback behaviors live in the sibling
``test_events_location.py``; step-control/accuracy and solver-internals tests live in
``test_events_step_control.py`` / ``test_events_solver_internals.py``.
"""

# the test/fixture, keeping collection cheap and loading platform-conditional deps (win32com,
# OCP, FEMM) only on the path that needs them. Hoist a file's imports and DELETE this header;
# the population is ratcheted by tests/architecture/ratchets/test_suppression_ratchet.py.

from __future__ import annotations

import numpy as np
import pytest
from tests.unit._helpers import _RampWithToggle

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient


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

    study = TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    problem = _Linear()
    solution = integrate_transient(problem, study, np.array([0.0]), time_breakpoints=(0.05, 0.27, 0.8))
    ts = np.asarray(solution.times)
    for bp in (0.05, 0.27, 0.8):
        assert np.isclose(ts, bp).any(), f'breakpoint {bp} not on the grid'

    plain = integrate_transient(problem, study, np.array([0.0]))
    # A clipped cell replaces its nominal target: 0.1 is displaced by the 0.05 breakpoint.
    assert 0.1 not in ts

    no_bps = integrate_transient(problem, study, np.array([0.0]), time_breakpoints=())
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
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(
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
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)

    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    with pytest.raises(ValueError, match='march window'):
        integrate_transient(_Linear(), study, np.array([0.0]), time_breakpoints=(1.5,))
    with pytest.raises(ValueError, match='strictly increasing'):
        integrate_transient(_Linear(), study, np.array([0.0]), time_breakpoints=(0.2, 0.2))


def test_thousands_of_time_breakpoints_all_land_on_the_grid() -> None:
    """SR-014: the breakpoint queue is consumed by a cursor, not ``pop(0)`` -- a PWM march's
    thousands of edges must all land (behavior identical to the list queue, pinned here at
    the scale that would expose the per-edge O(n) list shift as O(n^2)).
    """
    n_edges = 4000  # a 20 kHz carrier over 100 ms -- the SR-014 scale
    t1 = 0.1
    study = TimeWindow(t0=0.0, t1=t1, dt=1.0e-5, integrator=IntegratorKind.BDF2)
    bps = tuple(np.linspace(t1 / n_edges, t1, n_edges))  # every 2.5 cells, off-grid

    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    solution = integrate_transient(_Linear(), study, np.array([0.0]), time_breakpoints=bps)
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

    study = TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(_Linear(), study, np.array([0.0]), time_breakpoints=(0.05, 0.27, 0.8))
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

    study = TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(_Linear(), study, np.array([0.0]), time_breakpoints=(0.95,))
    ts = np.asarray(solution.times)
    us = np.asarray([float(s[0]) for s in solution.states])
    assert np.max(np.abs(us - ts)) < 1.0e-12
    assert ts[-1] == pytest.approx(1.0)
    assert 0.95 in ts


@pytest.mark.parametrize('bp', [0.95, 1.0])
def test_a_breakpoint_in_a_non_integral_final_interval_is_consumed(bp: float) -> None:
    """SR-049: a breakpoint after the last nominal step still lands and finishes at t1."""

    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    study = TimeWindow(t0=0.0, t1=1.0, dt=0.3, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(_Linear(), study, np.array([0.0]), time_breakpoints=(bp,))

    assert solution.times[-1] == pytest.approx(study.t1)
    assert any(t == pytest.approx(bp) for t in solution.times)


def test_a_non_integral_window_is_bit_identical_to_the_plain_march() -> None:
    """Without breakpoints both marches share the same partial final cell and states."""

    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    study = TimeWindow(t0=0.0, t1=1.0, dt=0.3, integrator=IntegratorKind.BDF2)
    plain = integrate_transient(_Linear(), study, np.array([0.0]))
    eventful = integrate_transient(_Linear(), study, np.array([0.0]))
    assert np.array_equal(plain.times, eventful.times)
    assert np.array_equal(np.asarray(plain.states), np.asarray(eventful.states))
    assert eventful.times[-1] == pytest.approx(1.0)


def test_event_location_tolerance_cannot_erase_a_short_observation_window() -> None:
    class _Linear:
        def residual(self, u, u_dot, u_ddot, t):
            return np.asarray(u_dot) - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1.real]])

    study = TimeWindow(t0=0.0, t1=0.1, dt=1.0, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(_Linear(), study, np.array([0.0]), time_tol=0.2)
    assert solution.times == pytest.approx([0.0, 0.1])
