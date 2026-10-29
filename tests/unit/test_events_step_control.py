"""Adaptive step rejection on the eventful march (local-error control lives in its own test file).

Split from the original ``test_events.py``: locator/rollback behaviors live in
``test_events_location.py``, ``time_breakpoints`` in ``test_events_breakpoints.py``, and
Newton-internal step-solving (``_solve_step``) tests in ``test_events_solver_internals.py``.
"""
# ruff: noqa: PLC0415 -- test-module lazy imports: the module under test is imported inside
# the test/fixture, keeping collection cheap and loading platform-conditional deps (win32com,
# OCP, FEMM) only on the path that needs them. Hoist a file's imports and DELETE this header;
# the population is ratcheted by tests/architecture/ratchets/test_suppression_ratchet.py.

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient


class _SharpLayer:
    """``u_dot = -C tanh((u - level)/eps)`` -- a switching layer sharp enough to defeat a full step.

    This is the shape the eventful march actually meets: a device whose conductance turns over
    within a window far narrower than the grid. Away from the layer the tangent is ``1/h`` and
    knows nothing about the wall ahead, so a full step drives the iterate straight across it and
    Newton rings; a shorter step stays on one side and converges. Nothing here is rigged per
    step -- the residual and the jacobian are the honest ones, and the step size is the only
    thing that decides.
    """

    def __init__(self, level: float = 0.5, c: float = 50.0, eps: float = 1.0e-3) -> None:
        self.level, self.c, self.eps = level, c, eps

    def residual(self, u, u_dot, u_ddot, t):
        return np.asarray(u_dot) + self.c * np.tanh((np.asarray(u) - self.level) / self.eps)

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        sech2 = 1.0 - np.tanh((float(np.asarray(u)[0]) - self.level) / self.eps) ** 2
        return np.array([[weights.alpha1.real + self.c * sech2 / self.eps]])

    @staticmethod
    def never(*_seam) -> float:
        """An event function that never crosses: the cell loop runs, the locator stays out of it.

        Written ``*_seam`` rather than ``(u, t)`` with a suppression: the arguments are unused BY
        CONSTRUCTION here, and a starred parameter says that in the signature instead of asking a
        linter to look away.
        """
        return 1.0


def test_the_eventful_march_REJECTS_a_step_its_newton_could_not_take() -> None:
    """``adaptive=True`` must actually shorten a failed step and land the shortened one.

    The gap it closes: without it this march accepts ``u`` from a step its OWN solver rejected --
    it records the instant in ``nonconverged_times`` and continues from that state. The plain
    ``integrate_transient(adaptive=True)`` has had reject-halve-retry since W6; the eventful march
    never did, and its own docstring named the omission.

    What this pins is the MECHANISM, and only that: the march must land points the fixed grid does
    not contain. It deliberately does NOT assert that the exhausted count drops -- MEASURED on this
    problem and on srdab_closed_loop, it does not reliably, and an assertion that the count improves
    would be a claim the implementation does not support. The step rejection is real; "and therefore
    the hard case converges" is the part that is false, so it is not asserted here.

    ``backtrack_cap=0`` removes the line search from both arms. The subject is what happens when a
    step CANNOT be taken, and the default backtracking would rescue this problem before the step
    size ever came into it -- leaving the test green with the feature under test never exercised.
    """
    dt, t1 = 1.0e-3, 2.0e-2
    study = TimeWindow(t0=0.0, t1=t1, dt=dt, integrator=IntegratorKind.BDF1)
    common = {
        'event_fns': (_SharpLayer.never,),
        'max_newton': 6,
        'tol': 1.0e-10,
        'backtrack_cap': 0,
        'on_nonconverged': 'label',
    }
    fixed = integrate_transient(_SharpLayer(), study, np.array([0.0]), **common)
    # The premise: without the retry this problem DOES exhaust. If it stopped doing so the test
    # would pass vacuously, proving nothing about the retry -- so the premise is asserted, not
    # assumed.
    assert fixed.nonconverged_times, 'the fixed march converged everywhere -- nothing to reject'

    shrunk = integrate_transient(_SharpLayer(), study, np.array([0.0]), adaptive=True, h_min=dt / 1000.0, **common)
    off_grid = [t for t in shrunk.times if abs(t / dt - round(t / dt)) > 1.0e-9]
    assert off_grid, (
        'no marched instant lies off the nominal grid, so no step was ever shortened -- '
        f'the retry did not engage (exhausted: {len(shrunk.nonconverged_times)})'
    )
    # A shortened step must not corrupt the march: the cell endpoints still arrive, the times stay
    # monotonic, and the window still closes at t1.
    times = np.asarray(shrunk.times, dtype=float)
    assert np.all(np.diff(times) > 0.0), 'the shortened steps left the march non-monotonic'
    assert times[-1] == pytest.approx(t1, abs=dt), f'the march stopped at {times[-1]}, not {t1}'


class _StiffOnlyAtLargeSteps:
    """``u_dot = 1`` whose Jacobian is WRONG (5x too soft) unless the step is short.

    Newton on a 5x-too-soft Jacobian overshoots by 4x per iteration and diverges, so a nominal
    step exhausts and its halves converge in one iteration. The trajectory is exact either way.
    """

    def residual(self, u, u_dot, u_ddot, t):
        return np.asarray(u_dot) - 1.0

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        alpha = weights.alpha1.real
        return np.array([[alpha if alpha > 12.0 else 0.2 * alpha]])


def test_a_step_rejected_and_then_re_landed_converged_is_not_labelled_non_converged() -> None:
    """The rejected full step's failure is a PROBE's; the halves that reach its instant converged.

    MEASURED 2026-09-25 on case_i's chopper: every step the retry rescued was still listed in
    ``nonconverged_times`` -- the landing at the same instant promoted the rejected attempt's record
    -- so the run was labelled `coarse` with every landed step solved.
    """
    dt = 0.1
    study = TimeWindow(t0=0.0, t1=0.3, dt=dt, integrator=IntegratorKind.BDF1)
    common = {
        'event_fns': (_SharpLayer.never,),
        'max_newton': 4,
        'tol': 1e-10,
        'backtrack_cap': 0,
        'on_nonconverged': 'label',
    }
    fixed = integrate_transient(_StiffOnlyAtLargeSteps(), study, np.array([0.0]), **common)
    assert fixed.nonconverged_times, 'PREMISE: the nominal step must exhaust, or nothing is rescued'
    shrunk = integrate_transient(_StiffOnlyAtLargeSteps(), study, np.array([0.0]), adaptive=True, **common)
    assert any(abs(t / dt - round(t / dt)) > 1e-9 for t in shrunk.times), 'no step was shortened'
    assert shrunk.nonconverged_times == []
    assert shrunk.nonconverged_probe_times, 'the rejected attempts stay on record as probes'
    np.testing.assert_allclose([u[0] for u in shrunk.states], shrunk.times, atol=1e-9)


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
    study = TimeWindow(t0=0.0, t1=2.0, dt=0.3, integrator=IntegratorKind.BDF2)
    # The form itself, on an exact history, steps a quadratic exactly (direct check, u(0.75)).
    from numerics_lab.time_integrators._forms import IntegratorState
    from numerics_lab.time_integrators._integrators import _bdf2_variable_form
    from numerics_lab.time_integrators._step import _solve_step

    state = IntegratorState(us=[np.array([0.045]), np.array([0.18])], t=0.6)
    form = _bdf2_variable_form(state, 0.15, 0.3, 0.75)
    u_step, converged, iterations, residual, _on_floor = _solve_step(
        problem, form, state.last_u, solve=lambda j, r: np.linalg.solve(np.atleast_2d(j), r), max_newton=1, tol=1e-12
    )
    assert abs(float(u_step[0]) - 0.28125) <= 1e-12
    assert converged and iterations == 1 and residual <= 1e-12  # the convergence contract

    # A mid-cell event: the event march's deviation from the TRUE solution stays at the plain
    # march's own truncation scale -- no O(1) corruption from the landing/remainder steps. The two
    # marches solve slightly different discrete recurrences once the grid shifts, so bit-equality
    # is NOT the expectation; same-scale true error is.
    def fire(_i, _t, _u) -> None:
        problem.fired = True

    plain = integrate_transient(problem, study, np.array([0.0]))
    eventful = integrate_transient(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=fire,
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

    def fire_on_grid(_i, _t, _u) -> None:
        on_grid.fired = True

    grid_march = integrate_transient(
        on_grid,
        study,
        np.array([0.0]),
        event_fns=(on_grid.event_fn,),
        on_event=fire_on_grid,
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
