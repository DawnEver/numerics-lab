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

from motronics.contracts.study_spec import TransientStudy
from motronics.gauss.time_integrators import (
    FactorCache,
    IntegratorState,
    TransientSolution,
    _bdf2_variable_form,
    _default_solve,
    _solve_step,
    get_integrator,
)

__all__ = ['EventfulTransientSolution', 'integrate_transient_with_events']

#: Events tolerated inside ONE nominal cell before the function is declared chattering. A genuine
#: circuit sees a handful of crossings per cell at most (a PWM edge and a state ripple); past this
#: the event function is re-triggering on its own zero, which no march can progress past. The
#: ceiling is sized for the CONTINUOUS-PI reproduction (MANUAL-635: the laplace FmDrive's
#: sampling_divisor = 200 samples the PI at the march's step rate -- one legitimately-located
#: sample per step, ~200 per nominal cell -- the dense-but-genuine event stream the locator
#: resolves, not chatter).
_MAX_EVENTS_PER_CELL = 512


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
    """Retract every point appended past the checkpoint; hand back its history (re-copied) + spacing.

    A retracted point is no longer a step OF the solution, so any exhaustion promoted to
    ``nonconverged_times`` when it landed is DEMOTED back to a discarded probe here -- otherwise a
    rolled-back step would keep clearing :meth:`converged` for a trajectory it is not part of.
    """
    for t_gone in solution.times[checkpoint.n_points :]:
        for j in range(len(solution.nonconverged_times) - 1, -1, -1):
            if solution.nonconverged_times[j] == t_gone:
                del solution.nonconverged_times[j]
                solution.nonconverged_probe_times.append(t_gone)
                break
    del solution.times[checkpoint.n_points :]
    del solution.states[checkpoint.n_points :]
    restored = _take_checkpoint(checkpoint.state, checkpoint.prev_dt, solution)
    return restored.state, restored.prev_dt


def _sign(value: float) -> bool:
    """The bisection's sign bit: ``g >= 0``. A landing exactly on zero reads as the far side."""
    return value >= 0.0


def integrate_transient_with_events(
    problem,
    study: TransientStudy,
    u0: NDArray[np.floating],
    *,
    event_fns: Sequence[Callable[[NDArray[np.floating], float], float]] = (),
    on_event: Callable[[int, float, NDArray[np.floating]], None] | None = None,
    time_breakpoints: Sequence[float] = (),
    v0: NDArray[np.floating] | None = None,
    a0: NDArray[np.floating] | None = None,
    solve=_default_solve,
    max_newton: int = 1,
    tol: float = 1e-12,
    tol_rel: float | None = None,
    time_tol: float | None = None,
    factor_cache: FactorCache | None = None,
    backtrack_band: float | None = None,
    backtrack_cap: int = 8,
    gmin_stages: tuple[float, ...] | None = None,
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

    ``time_breakpoints`` pins KNOWN non-smooth instants (a source edge -- the laplace
    ``SourceSpec.breakpoints_in`` instants) onto the marched grid: a breakpoint inside a nominal
    cell clips the cell's target, so the march lands exactly ON the edge instead of sampling past
    it and integrating a slope that switched mid-cell. A breakpoint is a TIME event, not a
    state event: no bisection, no ``on_event`` callback -- the grid is simply aligned, and the
    variable-step BDF2 form keeps the landing accurate. They must be strictly increasing and lie
    in the open-closed window ``(t0, t1]`` (the same window discipline
    ``SourceSpec.breakpoints_in`` validates). With ``time_breakpoints`` empty the march is
    bit-identical to omitting the argument.

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
    solution = EventfulTransientSolution(times=[study.t0], states=[u0])
    g_prev = [float(g(u0, study.t0)) for g in event_fns]
    prev_dt: float | None = None
    breakpoints = tuple(time_breakpoints)
    for i, bp in enumerate(breakpoints):
        if not study.t0 < bp <= study.t1:
            msg = f'time_breakpoints[{i}]={bp:g} is outside the study window ({study.t0:g}, {study.t1:g}]'
            raise ValueError(msg)
        if i and bp <= breakpoints[i - 1]:
            msg = f'time_breakpoints must be strictly increasing, got {time_breakpoints}'
            raise ValueError(msg)
    # Consumption is a CURSOR advance, not a pop(0): popping shifts the list tail on every
    # consumed edge -- O(n^2) over a PWM march with thousands of edges (SR-014). Breakpoints
    # are only ever consumed front-to-back, so an index into the sorted tuple is equivalent.
    bp_index = 0
    # The last time each function fired. A function lands on its own zero, so on the pass right
    # after firing its sign at the checkpoint is epsilon-arbitrary and reads as a fresh crossing
    # rooted AT the firing instant -- the same event counted twice (measured: a stateless
    # u-level function double-fired [0, 1, 1]). Suppress re-detection while the march is still
    # within one bracket_tol of the firing. The suppression's distance is measured to the CELL
    # TARGET, not the cell start: after an event the march RE-MARCHES from the fired instant, so
    # the cell start sits exactly at last_fired forever -- a pure-time sawtooth event (the
    # continuous-PI's every-step samples, MANUAL-635) with a GENUINE new crossing at every cell
    # target would be suppressed as a duplicate forever (measured: the sampling_divisor = 200's
    # PI fired exactly once). The target distance is the new-instant test: the same-instant
    # re-triggering reads ~0 and stays suppressed; a genuinely later crossing reads its true
    # distance. NAMED HOLE: a GENUINE re-crossing inside one bracket_tol of the firing is
    # suppressed with the spurious one -- physically indistinguishable from chatter, which
    # _MAX_EVENTS_PER_CELL already refuses.
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
        u_new, converged, _iterations, residual = _solve_step(
            problem,
            form,
            from_state.last_u,
            solve=solve,
            max_newton=max_newton,
            tol=tol,
            tol_rel=tol_rel,
            factor_cache=factor_cache,
            backtrack_band=backtrack_band,
            backtrack_cap=backtrack_cap,
            gmin_stages=gmin_stages,
        )
        # The convergence contract rides the eventful march too: an exhausted step is labelled on
        # the solution, never silent. But WHICH list it belongs in is not knowable here -- this
        # helper is called both for steps that will land and for bisection probes the rollback
        # throws away. So it records the failure as PROVISIONALLY a probe, and the commit sites
        # (`_commit`) promote it when the point actually joins the trajectory. Recording it as
        # landed here made `converged` ~9x pessimistic on a switching march and useless.
        solution.worst_step_residual = max(solution.worst_step_residual, residual)
        if not converged:
            solution.nonconverged_probe_times.append(t_new)
        return integrator.advance(from_state, u_new, dt_step, t_new), u_new

    def _commit(t_new: float, u_new: NDArray) -> None:
        """Append a point to the trajectory, promoting its exhaustion label from probe to landed."""
        solution.times.append(t_new)
        solution.states.append(u_new)
        # The step that produced this point recorded any failure against t_new; it landed, so it
        # is a real non-converged STEP. Only the most recent such record is promoted -- an earlier
        # discarded probe at the same instant stays a probe.
        for j in range(len(solution.nonconverged_probe_times) - 1, -1, -1):
            if solution.nonconverged_probe_times[j] == t_new:
                del solution.nonconverged_probe_times[j]
                solution.nonconverged_times.append(t_new)
                break

    # The plain march's fixed step count -- integrate_transient's own n_steps; the eventful
    # march without breakpoints must take exactly these (SR-038), so the count lives here.
    n_steps = round((study.t1 - study.t0) / dt)
    resume_from: float | None = None
    k = 0
    # A clipped cell RE-ANCHORS the nominal grid at the breakpoint (resume_from keeps the
    # uniform dt grid going from there); the plain nominal grid study.t0 + (k+1)*dt applies
    # only while nothing clipped. Without the re-anchor the nominal targets drift ahead of the
    # physics: measured on a linear ramp with a breakpoint at 0.05 (dt=0.1) -- times
    # [0, 0.05, 0.2, ...] carried states [0, 0.05, 0.15, ...], max |u - t| = 0.08 (SR-022,
    # the W1b march-corruption repro). A while loop (not a fixed step count) because
    # clipping inserts grid points: the march ends at t1, not at a fixed step count.
    while state.t < study.t1 - bracket_tol:
        if resume_from is not None:
            # The re-anchored grid continues from the last breakpoint; the t1 clamp makes the
            # FINAL re-anchored cell a PARTIAL step, so its dt is the true remainder -- a full
            # dt there advanced the physics past t1 while labelling it t1 (SR-025, measured
            # max |u - t| = 0.05 on a final-cell clip).
            t_target = min(resume_from, study.t1)
            resume_from = t_target + dt
            pending_dt = t_target - state.t
        # Without breakpoints the grid is the plain nominal one, and it must be BIT-
        # IDENTICAL to integrate_transient's: that march stops after its fixed step count
        # n_steps = round((t1-t0)/dt) -- which TRUNCATES at the last nominal step before
        # t1 (plain [0, .3, .6, .9] for t1=1.0, dt=0.3) -- never a full step past t1
        # (SR-038: the round-2 adjudication wrongly claimed integrate_transient lands
        # past t1; the eventful march did, and the bit-identity promise broke).
        elif k >= n_steps:
            # Preserve the plain march's truncated grid only when there is nothing left
            # to consume.  A breakpoint in the final partial interval makes that interval
            # part of the eventful march, including the remainder from the edge to t1.
            if bp_index >= len(breakpoints):
                break
            t_target = study.t1
            pending_dt = t_target - state.t
        else:
            t_target = study.t0 + (k + 1) * dt
            pending_dt = dt  # dt exactly (not t_target - state.t, which is off by ulps from dt)
        k += 1
        events_this_cell = 0
        while True:  # one nominal cell, possibly interrupted by events
            # W1b: a time breakpoint inside the cell clips the target -- a known non-smooth
            # instant is LANDED on the grid, never sampled past (a landed point is already
            # reached, so it is consumed and the next one examined).
            while bp_index < len(breakpoints) and breakpoints[bp_index] <= state.t + bracket_tol:
                bp_index += 1
            if bp_index < len(breakpoints) and breakpoints[bp_index] <= t_target:
                t_target = breakpoints[bp_index]
                pending_dt = t_target - state.t
                resume_from = t_target + dt  # the uniform grid continues from the breakpoint
            checkpoint = _take_checkpoint(state, prev_dt, solution)
            t_from = state.t
            probe_state, u_probe = step_from(state, prev_dt, pending_dt, t_target)
            g_new = [float(g(u_probe, t_target)) for g in event_fns]
            crossed = [
                i
                for i in range(len(event_fns))
                if _sign(g_new[i]) != _sign(g_prev[i]) and t_target - last_fired.get(i, -np.inf) > bracket_tol
            ]
            if not crossed:
                state = probe_state
                prev_dt = pending_dt
                _commit(t_target, u_probe)
                g_prev = g_new
                break

            # Bisect each crossed function to its root; the EARLIEST instant wins, and every
            # function rooted within ``bracket_tol`` of it fires AT it. The earliest-only rule
            # would drop a SIMULTANEOUS crossing (a dead-time H-bridge's complementary pair
            # roots the SAME pulse spec at the same instant), and the dropped function is
            # STUCK, not just delayed: it reads only its own state, which the fired event's
            # callback does not change, so its sign at the event (``g == 0`` reads as the far
            # side) matches its post-event sign and the re-detection never sees it. Measured
            # on hbridge_deadtime 2026-08-07: the S4's OFF edge at 37.4 us was lost and the
            # switch stayed ON through the whole dead-time, changing the circuit.
            candidates: list[tuple[float, int, NDArray[np.floating]]] = []
            for i in crossed:
                if g_new[i] == 0.0:
                    candidates.append((t_target, i, u_probe))
                elif t_target - t_from <= bracket_tol:
                    candidates.append((t_from, i, np.array(checkpoint.state.last_u, dtype=float, copy=True)))
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
                    t_star_i = 0.5 * (lo_t + hi_t)
                    trial_state, trial_prev_dt = _rollback(checkpoint, solution)
                    _ps, u_star_i = step_from(trial_state, trial_prev_dt, t_star_i - t_from, t_star_i)
                    candidates.append((t_star_i, i, u_star_i))

            # The suppression's SECOND half, where the landing INSTANT is known: a crossing
            # located within the tolerance of the function's last firing is the same instant's
            # re-detection (the re-marched remainder re-finds the just-fired zero on the other
            # side of the bisect's convergence -- measured [0, 1, 1] on a two-level ramp). The
            # crossed pre-filter's t_target proxy cannot tell that from a genuinely new crossing
            # (the continuous-PI's every-step samples land at every cell target, MANUAL-635),
            # so the drop happens HERE. A dropped remainder means the crossing was spurious:
            # the cell is consumed as if nothing crossed.
            candidates = [c for c in candidates if c[0] - last_fired.get(c[1], -np.inf) > bracket_tol]
            if not candidates:
                state = probe_state
                prev_dt = pending_dt
                _commit(t_target, u_probe)
                g_prev = g_new
                break
            t_star = min(c[0] for c in candidates)
            simultaneous = [c for c in candidates if c[0] - t_star <= bracket_tol]
            state, prev_dt = _rollback(checkpoint, solution)
            if t_star > t_from:  # land the march exactly on the event
                state, u_star = step_from(state, prev_dt, t_star - t_from, t_star)
                _commit(t_star, u_star)
            else:
                u_star = np.array(state.last_u, dtype=float, copy=True)
            for _root, i, _u in simultaneous:
                solution.events.append((i, t_star))
                if on_event is not None:
                    on_event(i, t_star, u_star)
                last_fired[i] = t_star
            # RESTART the history at the event: the toggle invalidates the pre-event derivative
            # history, so the first post-event step must not read it (see the module docstring for
            # the kink-rounding measurement). The landing step's v/a carry over for second-order
            # integrators; for BDF they are None and the restart is a plain bootstrap.
            state = integrator.start(u_star, t_star, state.v, state.a)
            prev_dt = None
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
