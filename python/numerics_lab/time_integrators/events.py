"""Event location + checkpoint/rollback for the time march -- physics-blind, roadmap W2.

A switch, a comparator, a contact: several physics have instants where the problem itself changes,
and a fixed-grid march past such an instant integrates a smeared version of it. This module adds
the two pieces of machinery, in the only shape gauss may honestly carry them:

* **checkpoint/rollback of OPAQUE state** -- an :class:`IntegratorState` plus the length of the
  solution and the last step size taken at the same instant. gauss owns the integrator history, so
  it owns rolling it back; the caller's own state (a device's on/off flag) is the caller's to
  snapshot, through its own seam. Neither half knows anything about the other's.
* **a root-find on an OPAQUE scalar event function** -- ``g(u, t) -> float``, whose zero is "an
  instant worth stopping at". The locator brackets a sign change across a step, then bisects by
  RE-STEPPING from the checkpoint with a candidate step size: the probe states are marched, never
  interpolated, so the located state is one the integrator could actually have produced. What the
  crossing MEANS is the caller's ``on_event`` callback -- gauss never learns what switched.

THE NONUNIFORM-HISTORY TRAP, and why this module carries its own BDF2 step form. Landing on an
event at ``t*`` and marching on means steps whose size differs from the spacing of the history
they differentiate against. Fixed-weight BDF2 assumes uniform spacing: with step ``dt'`` and
history spacing ``h`` its derivative coefficient is off by a factor ``(3dt' - h)/(2dt')`` -- an
O(1) wrong equation, not an order reduction (measured in development: a probe bisection
converged to the BRACKET EDGE, locating a 1.0 s crossing at 0.9 s). So any step whose size
differs from the previous spacing is prepared with the standard VARIABLE-step BDF2 formula (the
derivative of the quadratic through the three unevenly spaced points)::

    u_dot = u * (2 dt + h) / (dt (dt + h))  -  u_n * (dt + h) / (dt h)  +  u_{n-1} * dt / (h (dt + h))

which reduces to the uniform weights at ``h = dt`` exactly. Only BDF2 carries the
uniform-spacing assumption: BDF1 has no second state, and the second-order structural schemes
propagate ``v`` / ``a`` with formulas already valid at arbitrary ``dt``. The variable form is
used ONLY when the spacings differ (``prev_dt != dt_step``); otherwise the integrator's own
``prepare`` is called, so an event-free march is bit-identical to
:func:`~motronics.gauss.time_integrators.integrate_transient`.

AND THE COMPANION TRAP, which the variable form does NOT fix: an event is an instant where the
problem CHANGES, so the pre-event history estimates the post-event derivative with contaminated
data -- the quadratic through the pre-toggle states drags the OLD slope into the first post-event
step, rounding every derivative kink over ~1-2 steps (measured on a slope-toggling ramp: the step
after the toggle errs by +9.5% of the true value; on a switched buck the inductor ripple loses
1.4% to the rounding while mid-piece slopes stay exact). The honest treatment of a model
discontinuity is to RESTART the history at the event (``integrator.start``): the first post-event
step is then BDF1, whose derivative estimate ``(u_new - u*) / h`` carries no pre-event slope at
all -- exact for the piecewise-linear-between-events physics a switch produces, and one
bounded O(dt^2) step for smooth physics. Later steps re-enter the variable-spacing form, whose
history is by then entirely post-event. A ceremonial event (one whose callback changes nothing)
pays that one BDF1 step for no benefit; that is the price of the semantics, stated, not hidden.
For a second-order integrator the restart carries the landing step's ``v`` / ``a`` (continuous
state is preserved; only the derivative HISTORY is dropped).

This module imports only contracts + numpy and the sibling integrator module; it touches no
circuit, no device, no physics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from motronics.contracts.analysis_formulation import DerivativeWeights
from motronics.contracts.study_spec import TransientStudy
from motronics.gauss.time_integrators import (
    FactorCache,
    IntegratorState,
    StepForm,
    TransientSolution,
    _default_solve,
    _solve_step,
    get_integrator,
)

__all__ = ['EventfulTransientSolution', 'integrate_transient_with_events']

#: Events tolerated inside ONE nominal cell before the function is declared chattering. A genuine
#: circuit sees a handful of crossings per cell at most (a PWM edge and a state ripple); past this
#: the event function is re-triggering on its own zero, which no march can progress past.
_MAX_EVENTS_PER_CELL = 32


@dataclass
class EventfulTransientSolution(TransientSolution):
    """A :class:`TransientSolution` plus the located events as ``(function index, time)`` pairs."""

    events: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class _Checkpoint:
    """The march state at an instant, deep enough that probing past it cannot alias into it.

    ``prev_dt`` is part of the history (BDF2's weights read the previous spacing), so it is
    checkpointed and restored with the state vectors.
    """

    state: IntegratorState
    prev_dt: float | None
    n_points: int


def _take_checkpoint(state: IntegratorState, prev_dt: float | None, solution: TransientSolution) -> _Checkpoint:
    """Snapshot the integrator history (arrays COPIED -- a rollback must not restore an alias)."""
    us = [np.array(u, dtype=float, copy=True) for u in state.us]
    v = None if state.v is None else np.array(state.v, dtype=float, copy=True)
    a = None if state.a is None else np.array(state.a, dtype=float, copy=True)
    return _Checkpoint(state=IntegratorState(us=us, t=state.t, v=v, a=a), prev_dt=prev_dt, n_points=len(solution.times))


def _rollback(checkpoint: _Checkpoint, solution: TransientSolution) -> tuple[IntegratorState, float | None]:
    """Retract every point appended past the checkpoint; hand back its history (re-copied) + spacing."""
    del solution.times[checkpoint.n_points :]
    del solution.states[checkpoint.n_points :]
    restored = _take_checkpoint(checkpoint.state, checkpoint.prev_dt, solution)
    return restored.state, restored.prev_dt


def _sign(value: float) -> bool:
    """The bisection's sign bit: ``g >= 0``. A landing exactly on zero reads as the far side."""
    return value >= 0.0


def _bdf2_variable_form(state: IntegratorState, dt: float, h: float, t_new: float) -> StepForm:
    """The variable-spacing BDF2 step form (see the module docstring); ``h`` is the previous step."""
    u_n, u_nm1 = state.us[-1], state.us[-2]
    return StepForm(
        weights=DerivativeWeights(alpha0=1.0, alpha1=(2.0 * dt + h) / (dt * (dt + h))),
        h_u=np.zeros_like(u_n),
        h_dot=-(dt + h) / (dt * h) * u_n + dt / (h * (dt + h)) * u_nm1,
        h_ddot=np.zeros_like(u_n),
        t_eval=t_new,
    )


def integrate_transient_with_events(
    problem,
    study: TransientStudy,
    u0: NDArray[np.floating],
    *,
    event_fns: Sequence[Callable[[NDArray[np.floating], float], float]] = (),
    on_event: Callable[[int, float, NDArray[np.floating]], None] | None = None,
    v0: NDArray[np.floating] | None = None,
    a0: NDArray[np.floating] | None = None,
    solve=_default_solve,
    max_newton: int = 1,
    tol: float = 1e-12,
    time_tol: float | None = None,
    factor_cache: FactorCache | None = None,
) -> EventfulTransientSolution:
    """March ``problem`` over ``study``, stopping the grid at every zero of ``event_fns``.

    Identical to :func:`~motronics.gauss.time_integrators.integrate_transient` when ``event_fns``
    is empty (same times, same states, bit-for-bit). Otherwise each step is taken under a
    checkpoint; a sign change in any ``g_i`` between the step's ends brackets a crossing, which is
    bisected (by re-stepping from the checkpoint) down to ``time_tol`` and landed on exactly: the
    located point joins the march, ``on_event(i, t*, u*)`` runs -- the caller's whole knowledge of
    what the crossing means -- and the remainder of the nominal cell is re-marched under a fresh
    checkpoint. Steps whose size differs from their history's spacing use the variable-step BDF2
    form, so event-landing costs no accuracy (see the module docstring for the trap this avoids).

    ``time_tol`` is the bracket width the locator converges to, in seconds (default
    ``study.dt * 1e-6``). A crossing already within ``time_tol`` of the checkpoint is handled at
    the checkpoint rather than bisected into a zero-length step.

    The initial point is NOT scanned for events: a zero at ``t0`` is the caller's initial
    condition to declare, not an event to locate.

    ``factor_cache`` (W6) reuses one factorization per reuse key, as in
    :func:`~motronics.gauss.time_integrators.integrate_transient`. No invalidation hook is needed
    here: the key re-derives from the system every solve, so the toggle's changed tangent, the
    history restart's bootstrap weights, and each bisection probe's unique ``dt`` all miss the
    cache on their own -- a stale factor across an event is impossible unless the key itself
    under-reads the system (the hole the module docstring names).
    """
    if factor_cache is None and callable(getattr(problem, 'factor_reuse_key', None)):
        factor_cache = FactorCache()
    integrator = get_integrator(study.integrator)
    u0 = np.asarray(u0, dtype=float)
    state = integrator.start(u0, study.t0, v0, a0)
    dt = study.dt
    bracket_tol = dt * 1.0e-6 if time_tol is None else time_tol
    n_steps = round((study.t1 - study.t0) / dt)
    solution = EventfulTransientSolution(times=[study.t0], states=[u0])
    g_prev = [float(g(u0, study.t0)) for g in event_fns]
    prev_dt: float | None = None
    # The last time each function fired. A function lands on its own zero, so on the pass right
    # after firing its sign at the checkpoint is epsilon-arbitrary and reads as a fresh crossing
    # rooted AT the firing instant -- the same event counted twice (measured: a stateless
    # u-level function double-fired [0, 1, 1]). Suppress re-detection while the march is still
    # within one bracket_tol of the firing. NAMED HOLE: a GENUINE re-crossing of the same
    # function inside that window is suppressed with the spurious one -- physically
    # indistinguishable from chatter, which _MAX_EVENTS_PER_CELL already refuses.
    last_fired: dict[int, float] = {}

    def step_from(
        from_state: IntegratorState, from_prev_dt: float | None, dt_step: float, t_new: float
    ) -> tuple[IntegratorState, NDArray]:
        if (
            integrator.name == 'bdf2'
            and len(from_state.us) >= 2
            and from_prev_dt is not None
            and from_prev_dt != dt_step
        ):
            form = _bdf2_variable_form(from_state, dt_step, from_prev_dt, t_new)
        else:
            form = integrator.prepare(from_state, dt_step, t_new)
        u_new = _solve_step(
            problem, form, from_state.last_u, solve=solve, max_newton=max_newton, tol=tol, factor_cache=factor_cache
        )
        return integrator.advance(from_state, u_new, dt_step, t_new), u_new

    for k in range(n_steps):
        t_target = study.t0 + (k + 1) * dt
        pending_dt = dt  # dt exactly (not t_target - state.t, which is off by ulps from dt)
        events_this_cell = 0
        while True:  # one nominal cell, possibly interrupted by events
            checkpoint = _take_checkpoint(state, prev_dt, solution)
            t_from = state.t
            probe_state, u_probe = step_from(state, prev_dt, pending_dt, t_target)
            g_new = [float(g(u_probe, t_target)) for g in event_fns]
            crossed = [
                i
                for i in range(len(event_fns))
                if _sign(g_new[i]) != _sign(g_prev[i]) and t_from - last_fired.get(i, -np.inf) > bracket_tol
            ]
            if not crossed:
                state = probe_state
                prev_dt = pending_dt
                solution.times.append(t_target)
                solution.states.append(u_probe)
                g_prev = g_new
                break

            # Bisect each crossed function to its root; handle the EARLIEST one only.
            best: tuple[float, int, NDArray[np.floating]] | None = None
            for i in crossed:
                if g_new[i] == 0.0:
                    candidate = (t_target, i, u_probe)
                elif t_target - t_from <= bracket_tol:
                    candidate = (t_from, i, np.array(checkpoint.state.last_u, dtype=float, copy=True))
                else:
                    lo_t, lo_sign = t_from, _sign(g_prev[i])
                    hi_t = t_target
                    u_mid = u_probe
                    while hi_t - lo_t > bracket_tol:
                        mid_t = 0.5 * (lo_t + hi_t)
                        trial_state, trial_prev_dt = _rollback(checkpoint, solution)
                        _ps, u_mid = step_from(trial_state, trial_prev_dt, mid_t - t_from, mid_t)
                        if _sign(float(event_fns[i](u_mid, mid_t))) == lo_sign:
                            lo_t = mid_t
                        else:
                            hi_t = mid_t
                    t_star = 0.5 * (lo_t + hi_t)
                    trial_state, trial_prev_dt = _rollback(checkpoint, solution)
                    _ps, u_star = step_from(trial_state, trial_prev_dt, t_star - t_from, t_star)
                    candidate = (t_star, i, u_star)
                if best is None or candidate[0] < best[0]:
                    best = candidate

            t_star, i_star, u_star = best
            state, prev_dt = _rollback(checkpoint, solution)
            if t_star > t_from:  # land the march exactly on the event
                state, u_star = step_from(state, prev_dt, t_star - t_from, t_star)
                solution.times.append(t_star)
                solution.states.append(u_star)
            else:
                u_star = np.array(state.last_u, dtype=float, copy=True)
            solution.events.append((i_star, t_star))
            # RESTART the history at the event: the toggle invalidates the pre-event derivative
            # history, so the first post-event step must not read it (see the module docstring for
            # the kink-rounding measurement). The landing step's v/a carry over for second-order
            # integrators; for BDF they are None and the restart is a plain bootstrap.
            state = integrator.start(u_star, t_star, state.v, state.a)
            prev_dt = None
            if on_event is not None:
                on_event(i_star, t_star, u_star)
            last_fired[i_star] = t_star
            g_prev = [float(g(u_star, t_star)) for g in event_fns]
            pending_dt = t_target - t_star  # the genuine remainder of this cell
            events_this_cell += 1
            if events_this_cell > _MAX_EVENTS_PER_CELL:
                msg = (
                    f'more than {_MAX_EVENTS_PER_CELL} events inside one nominal cell ending at '
                    f'{t_target:g}: an event function whose sign does not flip when its callback runs '
                    're-triggers at the same instant forever. on_event must change what the crossing '
                    'MEANS (the toggle), so the same zero is not re-located; a function that cannot '
                    'do that is chattering, and the honest answer is to fix its model, not to march on'
                )
                raise RuntimeError(msg)
            if pending_dt <= bracket_tol:  # the event consumed the rest of the cell
                break
        # end of one nominal cell
    return solution
