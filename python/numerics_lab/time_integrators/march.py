"""THE time march -- the one loop every transient in this repo steps through, physics-blind.

It marches a nominal ``window.dt`` grid (the output grid and the largest step), lands KNOWN
non-smooth instants (``time_breakpoints``), locates STATE events (``event_fns``), halves a step
whose Newton failed (``adaptive``) and, with a :class:`StepTolerance`, controls every step's LOCAL
TRUNCATION ERROR per unknown. Without events, breakpoints or step control it is the plain fixed grid.

Event location + checkpoint/rollback, roadmap W2:

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
:func:`~numerics_lab.time_integrators.integrate_transient`.

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

This module imports only numpy and the sibling integrator module; it touches no
circuit, no device, no physics.

THE STEP PLANS, all inside this one loop. A march takes its cell targets from exactly one plan:

* the NOMINAL grid (the default): ``origin + (k + 1) dt`` cells, re-anchored at every landed
  breakpoint or live deadline; its locator cursor is what an event checkpoint persists;
* a LATTICE: an explicit ``step_times`` table, or the absolute parent lattice ``step_origin + k dt``
  (also the EEC cycle clock); a live deadline clips a cell without moving the lattice;
* the CURVATURE controller (``curvature_atol``, BDF2): one proposed step per cell, sized by the
  predictor-curvature indicator on an optional absolute ``timestep_quantum`` lattice.

Whatever the plan, every accepted state goes through the same commit: material history, the
accepted observers (``on_landing``, ``on_accept``) and their right-state projection, the EEC
cycle bookkeeping, the periodic steady-state stop and the checkpoint stream.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from numerics_lab.formulation import DerivativeWeights, TimeWindow
from numerics_lab.time_integrators._fixed_lattice import fixed_lattice_step_size, fixed_lattice_targets
from numerics_lab.time_integrators._forms import (
    IntegratorState,
    TransientSolution,
    _default_solve,
    seeded_bdf2_state,
)
from numerics_lab.time_integrators._integrators import get_integrator
from numerics_lab.time_integrators._landing import (
    AcceptedProjectionResult,
    accepted_landing,
    accepted_observation,
    accepted_projection,
    next_deadline,
    owned_deadline,
)
from numerics_lab.time_integrators._step import HeldFactor, ProblemCapabilities, StepOutcome, take_step
from numerics_lab.time_integrators._time_lattice import adaptive_lattice
from numerics_lab.time_integrators.checkpoint import (
    EECCheckpointState,
    StepControlCheckpointState,
    _checkpoint_of,
    owns_history,
    restore_history_checkpoint,
    validate_history_owner,
)
from numerics_lab.time_integrators.eec_tangent import CycleTangentCSR
from numerics_lab.time_integrators.event_checkpoint import EventCheckpointState
from numerics_lab.time_integrators.factor_reuse import FactorCache
from numerics_lab.time_integrators.instants import same_instant
from numerics_lab.time_integrators.periodic import (
    bdf2_boundary_moment,
    canonical_eec_window,
    explicit_error_correction,
    nonlinear_error_correction,
    periodic_converged,
    validate_eec_restart,
)
from numerics_lab.time_integrators.prepared import PreparedCandidates, PreparedStepProblem, restart_prepared

logger = logging.getLogger(__name__)

__all__ = [
    'BudgetExceeded',
    'EventSurface',
    'RunBudget',
    'StepNotConverged',
    'StepTolerance',
    'integrate_transient',
    'march_log_record',
]

#: Events tolerated inside ONE nominal cell before the function is declared chattering. A genuine
#: circuit sees a handful of crossings per cell at most (a PWM edge and a state ripple); past this
#: the event function is re-triggering on its own zero, which no march can progress past. The
#: ceiling is sized for the CONTINUOUS-PI reproduction (MANUAL-635: the laplace FmDrive's
#: sampling_divisor = 200 samples the PI at the march's step rate -- one legitimately-located
#: sample per step, ~200 per nominal cell -- the dense-but-genuine event stream the locator
#: resolves, not chatter).
_MAX_EVENTS_PER_CELL = 512
# A temporary performance cache, never a solve/acceptance threshold. Oversized trials are simply
# recomputed; large field vectors must not make simultaneous event observers exhaust host memory.
_EVENT_TRIAL_CACHE_BYTES = 64 * 1024**2
#: The integrators a portable checkpoint can carry (their full history is ``us`` + ``v``/``a``).
_CHECKPOINT_INTEGRATORS = ('bdf1', 'bdf2', 'trapezoid', 'tr_bdf2', 'newmark', 'generalized_alpha')


@dataclass(frozen=True)
class EventSurface:
    """A scalar observer with its actual declared crossing orientation.

    Direction zero locates either sign change; +1/-1 admit rising/falling
    crossings. An initial zero leaving into the stay region is not an event.
    """

    observe: Callable[[NDArray[np.floating], float], float]
    direction: int = 0

    def __post_init__(self) -> None:
        if not callable(self.observe) or type(self.direction) is not int or self.direction not in (-1, 0, 1):
            msg = 'event surfaces require a callable and an explicit -1/0/+1 direction'
            raise ValueError(msg)

    def __call__(self, state, time) -> float:
        return float(self.observe(state, time))


class StepNotConverged(RuntimeError):
    """A step the solver could not converge would have JOINED the trajectory: the state reached is here.

    SOLVER HONESTY: a march never returns a trajectory containing a state its own Newton rejected
    unless the caller asked for labels (``on_nonconverged='label'``). ``t`` is the step's instant,
    ``residual`` its last residual norm, ``state`` the iterate it stopped at and ``t_reached`` /
    ``u_reached`` the last state that DID converge. Nothing of the refused step -- trajectory point,
    material history, accepted observation or checkpoint -- is published before this is raised.
    """

    def __init__(self, t: float, residual: float, state, t_reached: float, u_reached, *, worst_row=None) -> None:
        named = '' if worst_row is None else f' Worst {worst_row}.'
        super().__init__(
            f'the step to t={t:g} did not converge: residual {residual:.3e}; the march reached t={t_reached:g}.'
            f'{named} A spurious or absent root (a gate-coupled source chattering between its modes) is the usual '
            'cause; pass on_nonconverged="label" only to RECORD such steps, never to hide them.'
        )
        self.t, self.residual, self.state, self.t_reached, self.u_reached = t, residual, state, t_reached, u_reached
        #: The row the per-row bar refused by the largest ratio (``_step.RowVerdict``), when rows were judged.
        self.worst_row = worst_row


@dataclass(frozen=True)
class RunBudget:
    """Hard limits on one march: wall-clock seconds and resident memory bytes (``None`` = unlimited).

    The engine reads no process memory itself; a memory limit comes with ``resident_bytes``, the
    caller's probe of the resident set (e.g. ``lambda: psutil.Process().memory_info().rss``).

    Checked before and after each trial solve, and after every accepted step. An individual
    Newton/linear solve is not preempted. A march over budget RAISES :class:`BudgetExceeded` with the
    state it reached -- a run reaped by the OS reports nothing, a run that stops itself reports where.
    """

    wall_seconds: float | None = None
    memory_bytes: int | None = None
    resident_bytes: Callable[[], int] | None = None

    def __post_init__(self) -> None:
        if self.wall_seconds is not None and (not math.isfinite(self.wall_seconds) or self.wall_seconds < 0):
            msg = f'wall_seconds must be finite and nonnegative, got {self.wall_seconds}'
            raise ValueError(msg)
        if self.memory_bytes is not None and self.memory_bytes <= 0:
            msg = f'memory_bytes must be positive, got {self.memory_bytes}'
            raise ValueError(msg)
        if self.memory_bytes is not None and self.resident_bytes is None:
            msg = 'a memory_bytes limit needs resident_bytes, the caller-supplied resident-set probe'
            raise ValueError(msg)

    def check(self, started: float, t_reached: float, u_reached, steps: int) -> None:
        """Check one run's budget against its last accepted state, including discarded work."""
        reason = None
        if self.wall_seconds is not None and time.perf_counter() - started > self.wall_seconds:
            reason = f'wall > {self.wall_seconds:g} s'
        elif self.memory_bytes is not None and self.resident_bytes() > self.memory_bytes:
            reason = f'resident > {self.memory_bytes} B'
        if reason is not None:
            raise BudgetExceeded(reason, t_reached, u_reached, steps)


class BudgetExceeded(RuntimeError):
    """The march hit its :class:`RunBudget`; ``t_reached`` / ``u_reached`` / ``steps`` say how far it got."""

    def __init__(self, reason: str, t_reached: float, u_reached, steps: int) -> None:
        super().__init__(f'run budget exceeded ({reason}) at t={t_reached:g} after {steps} accepted steps')
        self.reason, self.t_reached, self.u_reached, self.steps = reason, t_reached, u_reached, steps


def march_log_record(body: str, **attributes) -> dict:
    """THE one march log line: OpenTelemetry log-record field names, attributes namespaced ``march.``."""
    return {
        'timestamp': datetime.now(UTC).isoformat(),
        'severity_text': 'WARN' if body in {'reject', 'nonconverged'} else 'INFO',
        'body': body,
        'attributes': {f'march.{k}': v for k, v in attributes.items()},
    }


class _StepLog:
    """One JSON object per line (:func:`march_log_record`), to a caller-owned text stream."""

    def __init__(self, stream) -> None:
        self.stream = stream

    def __call__(self, body: str, **attributes) -> None:
        if self.stream is None:
            return
        print(json.dumps(march_log_record(body, **attributes)), file=self.stream)


@dataclass(frozen=True)
class StepTolerance:
    """Per-unknown local-error tolerances: a step is accepted when EVERY unknown satisfies
    ``|LTE_i| <= atol_i + rtol_i * max(|u_n,i|, |u_n+1,i|)``.

    ``rtol`` / ``atol`` are scalars or vectors the length of the state; a vector is how a caller
    whose unknowns carry different units says so (a circuit's node volts and branch amperes need
    different absolute floors -- one scalar is right for at most one of them). ``h_min`` is the
    smallest step the controller may take (default ``window.dt * 1e-6``); a step still rejected
    there RAISES with the error reached rather than being accepted.
    """

    rtol: float | NDArray[np.floating] = 1e-3
    atol: float | NDArray[np.floating] = 1e-6
    h_min: float | None = None


def _lte_ratio(
    history: list[tuple[float, NDArray[np.floating]]],
    t_new: float,
    u_new: NDArray[np.floating],
    order: int,
    error_constant: float,
    tolerance: StepTolerance,
) -> float:
    """The worst unknown's ``|LTE| / tol`` for the step to ``(t_new, u_new)``; 0 without enough history.

    The predictor is the degree-``order`` polynomial through the last ``order + 1`` accepted points,
    extrapolated to ``t_new``. Its gap to the corrector is ``u^(p+1) prod(t_new - t_i) / (p+1)!``
    to leading order on ANY spacing, and the scheme's own error is ``C h^(p+1) u^(p+1)``; their
    ratio converts one into the other, so the estimate needs no assumption of uniform steps.
    """
    if len(history) < order + 1:
        return 0.0
    pts = history[-(order + 1) :]
    ts = [t for t, _u in pts]
    pred = np.zeros(u_new.shape, dtype=u_new.dtype)
    for j, (tj, uj) in enumerate(pts):
        w = 1.0
        for m, tm in enumerate(ts):
            if m != j:
                w *= (t_new - tm) / (tj - tm)
        pred = pred + w * uj
    span = float(np.multiply.reduce([t_new - t for t in ts]))
    h = t_new - ts[-1]
    lte = error_constant * math.factorial(order + 1) * h ** (order + 1) / span * (u_new - pred)
    scale = tolerance.atol + tolerance.rtol * np.maximum(np.abs(pts[-1][1]), np.abs(u_new))
    return float((np.abs(lte) / scale).max()) if lte.size else 0.0


#: Step-size controller limits: never grow past 2x or shrink below 0.2x in one decision, and aim
#: 10% inside the tolerance so an accepted step is not immediately followed by a rejected one.
_GROW, _SHRINK, _SAFETY = 2.0, 0.2, 0.9


def _resize(h: float, ratio: float, order: int) -> float:
    if ratio <= 0.0:
        return _GROW * h
    return h * min(_GROW, max(_SHRINK, _SAFETY * ratio ** (-1.0 / (order + 1))))


@dataclass
class _Checkpoint:
    """The march state at an instant, deep enough that probing past it cannot alias into it.

    The history spacing rides :class:`IntegratorState` (``h``), so the copy carries it.
    """

    state: IntegratorState
    n_points: int


def _take_checkpoint(state: IntegratorState, solution: TransientSolution) -> _Checkpoint:
    """Snapshot the integrator history (arrays COPIED -- a rollback must not restore an alias)."""
    return _Checkpoint(state=state.copy(), n_points=len(solution.times))


def _rollback(checkpoint: _Checkpoint, solution: TransientSolution) -> IntegratorState:
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
    if solution.state_rates is not None:
        del solution.state_rates[max(0, checkpoint.n_points - 1) :]
    if solution.residual_times is not None:
        del solution.residual_times[max(0, checkpoint.n_points - 1) :]
    if solution.rate_state_overrides is not None:
        for index in tuple(solution.rate_state_overrides):
            if index >= max(0, checkpoint.n_points - 1):
                del solution.rate_state_overrides[index]
    return checkpoint.state.copy()


def _sign(value: float) -> bool:
    """The bisection's sign bit: ``g >= 0``. A landing exactly on zero reads as the far side."""
    if not np.isfinite(value):
        msg = 'event observations must be finite'
        raise ValueError(msg)
    return value >= 0.0


def _crossed(surface, before: float, after: float) -> bool:
    """Whether ``surface`` crossed between two readings, honouring a declared :class:`EventSurface` direction."""
    before_sign, after_sign = _sign(before), _sign(after)
    direction = surface.direction if isinstance(surface, EventSurface) else 0
    if direction > 0:
        return before < 0.0 and after >= 0.0
    if direction < 0:
        return before > 0.0 and after <= 0.0
    return before_sign != after_sign


def _locate_bracket(
    lo_t: float, g_lo: float, hi_t: float, g_hi: float, probe: Callable[[float], float], *, tol: float
) -> tuple[float, float]:
    """Shrink a sign-change bracket to ``tol`` wide; every ``probe`` is a whole re-step.

    ILLINOIS regula falsi, safeguarded: the secant point is probed, and the end that survives twice
    has its ``g`` halved so neither end stalls. A probe is aimed half a tolerance
    bracket past the secant root, so a secant that has converged straddles the root and closes the
    bracket in one more probe. Two probes that do not together halve the bracket earn a bisection
    (Brent's guard), so a pathological ``g`` still terminates. Same sign convention, same final width, same midpoint
    rule as the plain bisection this replaced (which spent ~21 re-steps per crossing, measured).
    """
    lo_sign = _sign(g_lo)
    side = 0  # which end moved on the last probe: -1 lo, +1 hi
    widths = [hi_t - lo_t]
    while hi_t - lo_t > tol:
        width = hi_t - lo_t
        # Brent's guard: two probes that did not together halve the bracket earn a bisection.
        if len(widths) >= 3 and width > 0.5 * widths[-3]:
            mid_t = 0.5 * (lo_t + hi_t)
            widths.clear()
        elif g_hi == g_lo or not (np.isfinite(g_lo) and np.isfinite(g_hi)):
            mid_t = 0.5 * (lo_t + hi_t)
        else:
            # Aim half a tolerance PAST the secant root, toward the end that did not move: once the
            # secant has converged this probe straddles the root and the bracket closes.
            mid_t = hi_t - g_hi * width / (g_hi - g_lo) + (0.5 * tol if side <= 0 else -0.5 * tol)
            mid_t = min(max(mid_t, lo_t + 0.25 * tol), hi_t - 0.25 * tol)
        g_mid = probe(mid_t)
        if _sign(g_mid) == lo_sign:
            lo_t, g_lo = mid_t, g_mid
            if side == -1:
                g_hi *= 0.5
            side = -1
        else:
            hi_t, g_hi = mid_t, g_mid
            if side == 1:
                g_lo *= 0.5
            side = 1
        widths.append(hi_t - lo_t)
    return lo_t, hi_t


def _positive_finite(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and bool(np.isfinite(value)) and value > 0


def _admit_eec(window, integrator, *, period, sign, affine, nonlinear, step_times, curvature, owner) -> None:
    """Every TP-EEC admission rule, before any history is restored or any step is solved."""
    if nonlinear and (period is None or affine):
        msg = 'nonlinear EEC needs an active interval and its distinct nonlinear operator assertion'
        raise ValueError(msg)
    if (
        isinstance(sign, bool)
        or not isinstance(sign, Integral)
        or sign not in (-1, 1)
        or (sign == -1 and period is None)
    ):
        msg = 'eec_sign must be explicit +1 or -1 with an active correction interval'
        raise ValueError(msg)
    if period is None:
        return
    if nonlinear and (sign != 1 or integrator.name != 'bdf1' or curvature or step_times is not None):
        msg = 'nonlinear EEC requires uniform full-period BDF1 without step tables or adaptive controls'
        raise ValueError(msg)
    if isinstance(period, bool) or not isinstance(period, Real) or not np.isfinite(period) or period <= 0:
        msg = 'eec_period requires a positive finite nonboolean interval'
        raise ValueError(msg)
    if (
        not np.isfinite(window.dt)
        or window.dt <= 0
        or not window.t0 + window.dt > window.t0
        or not window.t1 - window.dt < window.t1
        or not window.t0 + period > window.t0
    ):
        msg = 'EEC requires representable advancing time steps and correction intervals'
        raise ValueError(msg)
    if sign == -1 and integrator.name == 'bdf1' and step_times is not None:
        msg = 'anti-periodic EEC requires fixed uniform BDF1 without step tables'
        raise ValueError(msg)
    if integrator.name == 'bdf2' or nonlinear:
        intervals = (window.t1 - window.t0) / window.dt
        ticks = period / window.dt
        if (
            not np.isfinite(intervals + ticks)
            or ticks < 1
            or not np.isclose(intervals, round(intervals), rtol=0.0, atol=1e-9)
            or not np.isclose(ticks, round(ticks), rtol=0.0, atol=1e-9)
            or (
                step_times is not None
                and not np.allclose(np.diff(step_times), window.dt, rtol=0.0, atol=window.dt * 1e-12)
            )
        ):
            msg = (
                'nonlinear EEC requires a uniform BDF1 grid with positive integer ticks and aligned endpoints'
                if nonlinear
                else 'EEC requires a uniform BDF2 grid with positive integer ticks and aligned final '
                'endpoint/correction period'
            )
            raise ValueError(msg)
    if not (affine or nonlinear):
        msg = 'TP-EEC requires an explicit assertion of a time-invariant affine first-order operator'
        raise ValueError(msg)
    if curvature or integrator.name not in ('bdf1', 'bdf2'):
        msg = 'eec_period requires a positive period and fixed/table first-order BDF march'
        raise ValueError(msg)
    if owns_history(owner):
        msg = 'TP-EEC cannot correct material history'
        raise ValueError(msg)


def integrate_transient(
    problem,
    window: TimeWindow,
    u0: NDArray[np.floating],
    *,
    event_fns: Sequence[Callable[[NDArray[np.floating], float], float]] = (),
    on_event: Callable[[int, float, NDArray[np.floating]], None] | None = None,
    on_breakpoint: Callable[[float, NDArray[np.floating]], AcceptedProjectionResult | NDArray | None] | None = None,
    time_breakpoints: Sequence[float] = (),
    observation_times: Sequence[float] = (),
    v0: NDArray[np.floating] | None = None,
    a0: NDArray[np.floating] | None = None,
    solve=_default_solve,
    max_newton: int = 1,
    tol: float = 1e-12,
    tol_rel: float | None = None,
    row_acceptance: bool = False,
    time_tol: float | None = None,
    factor_cache: FactorCache | None = None,
    backtrack_band: float | None = 1.5,
    backtrack_cap: int = 4,
    line_search: str = 'backtrack',
    gmin_stages: tuple[float, ...] | None = None,
    adaptive: bool = False,
    h_min: float | None = None,
    refine: float | None = None,
    step_control: StepTolerance | None = None,
    curvature_atol: float | None = None,
    h_max: float | None = None,
    timestep_quantum: float | None = None,
    timestep_origin: float | None = None,
    step_times: Sequence[float] | None = None,
    step_origin: float | None = None,
    initial_history: tuple[np.ndarray, ...] | None = None,
    on_nonconverged: str = 'raise',
    step_log=None,
    budget: RunBudget | None = None,
    refactor_every: int | None = None,
    steady_period: float | None = None,
    steady_atol: float = 1e-6,
    steady_rtol: float = 1e-3,
    restart=None,
    checkpoint_key: str | None = None,
    on_checkpoint=None,
    eec_period: float | None = None,
    eec_sign: int = 1,
    eec_affine_first_order: bool = False,
    eec_nonlinear_first_order: bool = False,
    on_landing: Callable[[float, NDArray[np.floating]], Sequence[int]] | None = None,
    on_accept: Callable[[float, NDArray[np.floating]], bool] | None = None,
    next_breakpoint: Callable[[float], float | None] | None = None,
    project_accepted_state: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None = None,
) -> TransientSolution:
    """March ``problem`` over ``window``, stopping the grid at every zero of ``event_fns``.

    With no ``event_fns`` this is the plain fixed-grid march. Otherwise each step is taken under a
    checkpoint; a sign change in any ``g_i`` between the step's ends brackets a crossing, which is
    bisected (by re-stepping from the checkpoint) down to ``time_tol`` and landed on exactly: the
    located point joins the march, ``on_event(i, t*, u*)`` runs -- the caller's whole knowledge of
    what the crossing means -- and the remainder of the nominal cell is re-marched under a fresh
    checkpoint. Steps whose size differs from their history's spacing use the variable-step BDF2
    form, so event-landing costs no accuracy (see the module docstring for the trap this avoids).
    An :class:`EventSurface` declares a crossing orientation; a plain callable locates both.

    ``time_tol`` is the bracket width the locator converges to, in seconds (default
    ``window.dt * 1e-6``). A crossing already within ``time_tol`` of the checkpoint is handled at
    the checkpoint rather than bisected into a zero-length step.

    The initial point is NOT scanned for events: a zero at ``t0`` is the caller's initial
    condition to declare, not an event to locate.

    ``adaptive`` (default ``False``) REJECTS a step whose Newton failed and halves it (down to
    ``h_min``, default ``window.dt / 1000``) until it converges or hits the floor; a shortened step
    lands inside the cell and the march continues to the same nominal target, so the cell's
    endpoint is unchanged. Only a step that ACTUALLY failed is subdivided: blanket subdivision was
    measured WORSE (64 -> 98/148/268 exhausted on srdab_closed_loop).

    ``on_nonconverged`` is SOLVER HONESTY: ``'raise'`` (the default) refuses to let a step whose
    Newton exhausted join the trajectory -- :class:`StepNotConverged` carries the residual and the
    state reached, and nothing of the step is published first. ``'label'`` keeps the march going
    and records the instant in ``nonconverged_times`` (with a warning), for a caller whose
    contract is to REPORT it. Discarded bisection probes never raise.

    ``step_log`` (a writable text stream the caller owns) receives one JSON line per solve,
    accepted step, rejection, event and landed exhaustion, in OpenTelemetry log-record fields.
    ``budget`` (:class:`RunBudget`) aborts the march with :class:`BudgetExceeded` and the state
    reached.

    ``step_control`` is LOCAL-ERROR control for every integrator: each converged step that lands
    without crossing an event has its truncation error estimated per unknown (:func:`_lte_ratio`)
    against the :class:`StepTolerance`; a step over tolerance is rejected and retried smaller, an
    accepted one sets the next step's size. ``window.dt`` stays the largest step and the output
    grid. A step still over tolerance at ``h_min`` RAISES. ``refine`` is the eventful march's
    step-doubling check: a step whose two half-steps differ by more than ``refine`` (relative to
    ``max |u|``) is replaced by them.

    ``curvature_atol`` sizes BDF2 steps by the predictor-curvature indicator
    ``|u_new - u_extrap|`` (an ``O(h^2)`` curvature measure, not a certified local error) between
    ``h_min`` and ``h_max`` (defaults ``dt / 1000`` and ``1000 dt``), optionally on the absolute
    ``timestep_quantum`` lattice from ``timestep_origin``. A step accepted at the floor over the
    bar is recorded in ``indicator_unmet_times``.

    ``time_breakpoints`` pins KNOWN non-smooth instants onto the marched grid: a breakpoint inside
    a nominal cell clips the cell's target, so the march lands exactly ON the edge, its residual is
    evaluated at the left limit, ``on_breakpoint`` may project the right state, and the derivative
    history restarts. They must be strictly increasing in ``(t0, t1]``. ``next_breakpoint(t)`` is
    the owner's LIVE next deadline, re-read after every accepted state: it clips steps without
    changing endpoint laws (those remain the owner's, through ``on_accept`` / ``on_landing``).

    ``observation_times`` requires true solved endpoints on a continuous requested grid. Like
    breakpoints these instants clip a step and re-anchor its nominal grid, but they never take a
    left-limit evaluation or restart integrator/error history. Values must increase strictly in
    ``(t0, t1]``. A coincident actual breakpoint keeps its discontinuity semantics.

    ``step_times`` replaces the nominal grid with an explicit table from ``t0`` to ``t1``;
    ``step_origin`` with the absolute parent lattice ``step_origin + k dt`` (a fresh restart window
    keeps the producing run's products exactly). BESIDE a table, ``step_origin`` has ONE meaning: it
    is the table's PROVENANCE -- the parent lattice the table was cut from -- recorded on every
    checkpoint so a restart re-derives the same table, and it moves no step.

    ACCEPTED OBSERVERS see canonical states only, never a Newton or event probe: ``on_landing(t, u)``
    returns the integer identifiers of discontinuities dispatched at that accepted state,
    ``on_accept(t, u)`` returns whether the owner's law changed there (it also sees ``t0`` on a
    fresh march). Either restarts the derivative history; ``project_accepted_state`` supplies the
    right-side state, while the incoming step's rate keeps its left state in
    ``rate_state_overrides``.

    ``restart`` / ``checkpoint_key`` / ``on_checkpoint`` resume from and stream a portable
    :class:`~numerics_lab.time_integrators.checkpoint.TransientCheckpoint` after every accepted
    state: integrator history, material history (``checkpoint_history`` protocol), and the step
    plan's own clock -- the nominal locator cursor, the parent lattice origin, or the EEC cycle.
    A resumed march reproduces the uninterrupted one.

    ``steady_period`` stops the march once two complete cycles match within ``steady_atol`` /
    ``steady_rtol``. ``eec_period`` enables TP-EEC correction at that interval (``eec_sign=-1``
    anti-periodic); the generic residual protocol cannot prove linearity, so callers must assert
    ``eec_affine_first_order`` (``M u' + K u = f(t)``, constant K and M) or
    ``eec_nonlinear_first_order`` (uniform BDF1, the cycle-integrated tangent). No second-order,
    moving-operator or material-history correction is supported. ``initial_history`` seeds BDF2
    with its two states at ``t0 - dt`` and ``t0``.

    ``factor_cache`` (W6) reuses one factorization per reuse key; the key re-derives from the
    system every solve, so a toggle, a history restart or a bisection probe's unique ``dt`` all
    miss the cache on their own.
    """
    # ---------------------------------------------------------------- admission
    if on_nonconverged not in {'raise', 'label'}:
        msg = f"on_nonconverged must be 'raise' or 'label', got {on_nonconverged!r}"
        raise ValueError(msg)
    if type(adaptive) is not bool:
        msg = 'the step policy shared by the live locator and its event checkpoint requires a boolean adaptive flag'
        raise ValueError(msg)
    if h_min is not None and not _positive_finite(h_min):
        msg = 'the step policy shared by the live locator and its event checkpoint requires a finite positive h_min'
        raise ValueError(msg)
    if refine is not None and not _positive_finite(refine):
        msg = 'the step policy shared by the live locator and its event checkpoint requires a positive refinement'
        raise ValueError(msg)
    held = HeldFactor(refactor_every) if refactor_every is not None else None
    emit = _StepLog(step_log)
    started = time.perf_counter()
    capabilities = ProblemCapabilities.resolve(problem)
    if factor_cache is None and capabilities.factor_reusable:
        factor_cache = FactorCache()
    integrator = get_integrator(window.integrator)
    curvature = curvature_atol is not None
    eec = eec_period is not None
    _admit_eec(
        window,
        integrator,
        period=eec_period,
        sign=eec_sign,
        affine=eec_affine_first_order,
        nonlinear=eec_nonlinear_first_order,
        step_times=step_times,
        curvature=curvature or adaptive or step_control is not None or refine is not None,
        owner=problem,
    )
    window, restart = canonical_eec_window(window, restart, uniform_clock=eec_period is not None and step_times is None)
    if steady_period is not None and (not np.isfinite(steady_period) or steady_period <= 0):
        msg = 'steady_period must be finite and positive'
        raise ValueError(msg)
    if steady_atol < 0 or steady_rtol < 0 or not np.isfinite(steady_atol + steady_rtol):
        msg = 'steady tolerances must be finite and nonnegative'
        raise ValueError(msg)
    if curvature:
        if integrator.name != 'bdf2':
            msg = f'curvature stepping is implemented for BDF2 only, got integrator {integrator.name}'
            raise ValueError(msg)
        if event_fns or step_control is not None or refine is not None or observation_times or step_times is not None:
            msg = 'curvature stepping sizes every step itself: it cannot combine with events, step tables or another controller'
            raise ValueError(msg)
    elif timestep_quantum is not None or h_max is not None:
        msg = 'timestep_quantum and h_max belong to the curvature controller (curvature_atol)'
        raise ValueError(msg)
    if step_control is not None and refine is not None:
        msg = 'step_control and refine are two accuracy controllers for one step; declare one'
        raise ValueError(msg)
    if eec and (event_fns or time_breakpoints or observation_times):
        msg = 'EEC corrects a uniform physical cycle: it cannot combine with events, breakpoints or observations'
        raise ValueError(msg)
    if on_landing is not None and (curvature or eec):
        msg = 'accepted landing currently requires fixed stepping without EEC'
        raise ValueError(msg)
    if on_accept is not None and eec:
        msg = 'accepted observation requires a physical trajectory without EEC state corrections'
        raise ValueError(msg)
    if next_breakpoint is not None and eec:
        msg = 'live breakpoints cannot combine with EEC schedule corrections'
        raise ValueError(msg)
    if project_accepted_state is not None and (
        on_accept is None and on_landing is None and not (event_fns and on_event is not None)
    ):
        msg = 'accepted state projection requires an accepted observer, landing or event owner'
        raise ValueError(msg)
    if on_checkpoint is not None and checkpoint_key is None:
        msg = 'on_checkpoint requires checkpoint_key'
        raise ValueError(msg)
    if checkpoint_key is not None:
        if isinstance(problem, PreparedStepProblem) and not owns_history(problem):
            msg = 'prepared step checkpoint requires an explicit committed hidden-history owner'
            raise ValueError(msg)
        if integrator.name not in _CHECKPOINT_INTEGRATORS:
            msg = f'a portable checkpoint carries {_CHECKPOINT_INTEGRATORS} history, not {integrator.name}'
            raise ValueError(msg)
        validate_history_owner(problem)
    if step_times is not None:
        grid = np.asarray(step_times, dtype=float)
        if (
            grid.ndim != 1
            or len(grid) < 2
            or not np.all(np.isfinite(grid))
            or not np.all(np.diff(grid) > 0)
            or grid[0] != window.t0
            or grid[-1] != window.t1
        ):
            msg = 'step_times must increase from t0 to t1 and cannot combine with adaptive'
            raise ValueError(msg)
    if step_origin is not None and (
        curvature
        or isinstance(step_origin, bool)
        or not isinstance(step_origin, Real)
        or not np.isfinite(step_origin)
        or step_origin > window.t0
        or eec
    ):
        msg = 'step_origin requires a finite fixed parent lattice without EEC'
        raise ValueError(msg)
    if restart is not None and step_origin is not None and restart.step_origin != step_origin:
        msg = 'checkpoint step_origin does not match the explicit schedule'
        raise ValueError(msg)
    if restart is not None and restart.newton_tolerance is not None:
        # THE STRAIGHT MARCH'S BAR: the record's, never the one the caller recomputed at the resume instant
        # (`TransientCheckpoint.newton_tolerance` says why).
        tol = restart.newton_tolerance
    # THE STEP PLAN: exactly one of nominal / lattice (table, parent lattice or EEC clock) / curvature.
    plan = (
        'curvature'
        if curvature
        else ('lattice' if (step_times is not None or step_origin is not None or eec) else 'nominal')
    )
    if plan == 'lattice' and checkpoint_key is not None and event_fns:
        msg = 'a lattice-plan checkpoint carries no event locator history; locate events on the nominal grid'
        raise ValueError(msg)
    dt = window.dt
    if time_tol is not None and (not np.isfinite(time_tol) or time_tol <= 0.0):
        msg = f'time_tol must be finite and positive, got {time_tol!r}'
        raise ValueError(msg)
    bracket_tol = dt * 1.0e-6 if time_tol is None else time_tol
    minimum_step = dt / 1000.0 if h_min is None else h_min
    directions = tuple(surface.direction if isinstance(surface, EventSurface) else 0 for surface in event_fns)
    locator_policy = (
        EventCheckpointState(
            dt, bracket_tol, adaptive, minimum_step, refine, 0, window.t0, None, 0, (), directions, line_search
        )
        if checkpoint_key is not None and plan == 'nominal'
        else None
    )

    # ---------------------------------------------------------------- the initial state
    u0 = np.asarray(u0, dtype=float)
    state = integrator.start(u0, window.t0, v0, a0)
    if initial_history is not None:
        state = seeded_bdf2_state(state, initial_history, integrator, restart, dt)
    origin = window.t0 if step_origin is None else float(step_origin)
    cursor = None
    if restart is not None:
        restart.validate_restart(checkpoint_key, integrator.name, window.t0, u0.size)
        validate_eec_restart(restart, eec_period, eec_sign, dt, eec_nonlinear_first_order)
        if restart.event_state is not None:
            if plan != 'nominal':
                msg = 'event locator history must resume through the nominal eventful grid'
                raise ValueError(msg)
            saved = restart.event_state
            policy = (dt, bracket_tol, adaptive, minimum_step, refine, directions, line_search)
            saved_policy = (
                saved.step_dt,
                saved.bracket_tolerance,
                saved.adaptive,
                saved.minimum_step,
                saved.refine,
                saved.surface_directions,
                saved.line_search,
            )
            if policy != saved_policy:
                msg = 'eventful restart does not match its actual step and surface policies'
                raise ValueError(msg)
            cursor = saved
        if plan == 'nominal':
            if restart.step_origin is None:
                msg = 'fixed checkpoint restart requires its original parent step_origin'
                raise ValueError(msg)
            origin = restart.step_origin
        restore_history_checkpoint(problem, restart)
        states = [np.array(vector, dtype=float) for vector in restart.states]
        state = IntegratorState(
            us=states,
            t=restart.time,
            v=None if restart.velocity is None else np.array(restart.velocity, dtype=float),
            a=None if restart.acceleration is None else np.array(restart.acceleration, dtype=float),
            h=restart.previous_dt,
        )
        u0 = state.last_u
    solution = TransientSolution(
        times=[window.t0], states=[u0], state_rates=[], residual_times=[], rate_state_overrides={}
    )
    # A fresh march's initial state is canonical too: the owner may declare its law there.
    if restart is None and on_accept is not None and accepted_observation(on_accept, state.t, state.last_u):
        projection = accepted_projection(project_accepted_state, state.t, state.last_u)
        u0 = projection.state.copy()
        state = restart_prepared(problem, integrator, state, projection, state.t)
        solution.states[0] = u0
    if eec:
        solution.correction_model = (
            'cycle_integrated_nonlinear_tangent' if eec_nonlinear_first_order else 'affine_cycle_mean'
        )

    # ---------------------------------------------------------------- per-run bookkeeping
    # The accepted points since the last start/restart: the step-error predictor's history.
    lte_history: list[tuple[float, NDArray]] = [(state.t, u0)]
    h_floor = dt * 1e-6 if step_control is None or step_control.h_min is None else step_control.h_min
    h_ctrl = dt  # the local-error controller's next step size (only consulted with step_control)
    if restart is not None and (restart.step_control_state is None) != (step_control is None):
        msg = 'checkpoint step control does not match the march: a record and its resume declare it both or neither'
        raise ValueError(msg)
    if restart is not None and step_control is not None:
        saved_control = restart.step_control_state
        if saved_control.tolerance != StepControlCheckpointState.tolerance_of(step_control):
            msg = 'checkpoint step control was taken under another tolerance; that resume would be another march'
            raise ValueError(msg)
        # The controller resumes where it stood: its predictor points and the step it proposed.
        lte_history = [(t, np.array(u, dtype=float)) for t, u in saved_control.points]
        h_ctrl = saved_control.next_step
    g_prev = [float(g(u0, state.t)) for g in event_fns]
    breakpoints = tuple(time_breakpoints)
    observations = tuple(observation_times)
    for label, instants in (('time_breakpoints', breakpoints), ('observation_times', observations)):
        for i, instant in enumerate(instants):
            if not window.t0 < instant <= window.t1:
                msg = f'{label}[{i}]={instant:g} is outside the march window ({window.t0:g}, {window.t1:g}]'
                raise ValueError(msg)
            if i and instant <= instants[i - 1]:
                msg = f'{label} must be strictly increasing, got {instants}'
                raise ValueError(msg)
    # Consumption is a CURSOR advance, not a pop(0): popping shifts the list tail on every
    # consumed edge -- O(n^2) over a PWM march with thousands of edges (SR-014).
    bp_index = 0
    observation_index = 0
    breakpoint_set = frozenset(breakpoints)
    # The last time each function fired. A function lands on its own zero, so on the pass right
    # after firing its sign at the checkpoint is epsilon-arbitrary and reads as a fresh crossing
    # rooted AT the firing instant -- the same event counted twice (measured: a stateless
    # u-level function double-fired [0, 1, 1]). Suppress re-detection while the march is still
    # within one bracket_tol of the firing, measured to the CELL TARGET (the continuous-PI's
    # every-step samples, MANUAL-635, are genuine new crossings at every target). NAMED HOLE: a
    # GENUINE re-crossing inside one bracket_tol of the firing is suppressed with the spurious one.
    last_fired: dict[int, float] = {} if cursor is None else dict(cursor.last_fired)
    accepted_dt: float | None = None if restart is None else restart.previous_dt
    stop = False  # the periodic steady state was reached

    # EEC cycle bookkeeping (the lattice plan's coarse correction clock).
    saved_eec = None if restart is None else restart.eec_state
    eec_origin = window.t0 if saved_eec is None else saved_eec.origin_time
    next_correction = None
    cycle_start = u0.copy()
    integrated_tangent = None
    cycle_converged = True
    if eec:
        next_correction = window.t0 + eec_period if saved_eec is None else saved_eec.next_correction_time
        if integrator.name == 'bdf2' and len(state.us) >= 2:
            cycle_start = bdf2_boundary_moment(state.us)
        if saved_eec is not None:
            cycle_start = np.asarray(saved_eec.cycle_start_moment, dtype=float)
            if saved_eec.cycle_tangent is not None:
                integrated_tangent = saved_eec.cycle_tangent.matrix()
                cycle_converged = saved_eec.cycle_converged

    # The step plan's own clock.
    nominal_anchor = origin if cursor is None else cursor.nominal_anchor
    k = 0 if cursor is None else cursor.nominal_index
    resumed_target = None if cursor is None else cursor.pending_target
    resumed_events = 0 if cursor is None else cursor.pending_events
    if plan == 'nominal' and cursor is None and restart is not None:
        # A checkpoint without a locator cursor resumes on its absolute parent lattice.
        tick = math.floor((state.t - origin) / dt)
        # `same_instant`: a full final cell is labelled with the declared endpoint, not the tick.
        if same_instant(origin + (tick + 1) * dt, state.t):
            tick += 1
        k = tick
        if not same_instant(origin + tick * dt, state.t):
            # Mid-cell: the parent cell this state lies in is the one still being marched.
            resumed_target = origin + (tick + 1) * dt
            k = tick + 1
    lattice_targets: list[float] = []
    if plan == 'lattice':
        if step_times is not None:
            lattice_targets = [float(t) for t in list(step_times)[1:]]
        else:
            lattice_origin = eec_origin if eec else origin
            lattice_targets = fixed_lattice_targets(lattice_origin, window.t0, window.t1, dt)
        if eec:
            count = max(0.0, (window.t1 - next_correction) / eec_period + 1.0)
            if not np.isfinite(count) or count > len(lattice_targets) + 1e-9:
                msg = 'TP-EEC correction period requires positive integer steps between boundaries'
                raise ValueError(msg)
            for index in range(int(np.floor(count + 1e-12))):
                boundary = next_correction + index * eec_period
                if not any(abs(t - boundary) <= eec_period * 1e-12 for t in lattice_targets):
                    msg = 'TP-EEC requires step times at every correction period'
                    raise ValueError(msg)
    lattice_index = 0
    # The curvature controller's state.
    curve_lattice = None
    if curvature:
        curve_lattice = adaptive_lattice(
            window, True, timestep_quantum, timestep_origin, dt if restart is None else restart.previous_dt
        )
        h_curve = dt
        h_prev_curve = dt if restart is None else restart.previous_dt
        h_lo = dt / 1000.0 if h_min is None else h_min
        h_hi = dt * 1000.0 if h_max is None else h_max
        if (
            not np.isfinite(curvature_atol)
            or curvature_atol <= 0
            or not np.isfinite(h_lo + h_hi)
            or not 0 < h_lo <= h_hi
        ):
            msg = 'curvature stepping requires positive finite curvature_atol and 0 < h_min <= h_max'
            raise ValueError(msg)
        if curve_lattice is not None:
            if h_hi < curve_lattice.quantum:
                msg = 'curvature h_max must permit at least one time lattice tick'
                raise ValueError(msg)
            floor_ticks = max(1, int(np.ceil(h_lo / curve_lattice.quantum)))
            h_lo = floor_ticks * curve_lattice.quantum
            if h_lo > h_hi:
                msg = 'curvature bounds must contain at least one time lattice step'
                raise ValueError(msg)
    elif timestep_origin is not None:
        msg = 'timestep_origin requires timestep_quantum'
        raise ValueError(msg)

    def check_budget() -> None:
        if budget is not None:
            budget.check(started, solution.times[-1], solution.states[-1], len(solution.times) - 1)

    prepared = PreparedCandidates(problem) if isinstance(problem, PreparedStepProblem) else None

    def step_from(from_state: IntegratorState, dt_step: float, t_new: float) -> StepOutcome:
        check_budget()
        out = take_step(
            problem,
            integrator,
            from_state,
            dt_step,
            t_new,
            capabilities=capabilities,
            candidate_registry=prepared,
            endpoint_left=t_new in breakpoint_set,
            solve=solve,
            max_newton=max_newton,
            tol=tol,
            tol_rel=tol_rel,
            row_acceptance=row_acceptance,
            factor_cache=factor_cache,
            backtrack_band=backtrack_band,
            backtrack_cap=backtrack_cap,
            line_search=line_search,
            gmin_stages=gmin_stages,
            held=held,
        )
        check_budget()
        converged, residual, on_floor = out.converged, out.residual, out.on_floor
        # Called for steps that will land AND for bisection probes a rollback discards, so a failure
        # is recorded PROVISIONALLY as a probe; `_commit` promotes it when the point actually lands.
        solution.worst_step_residual = max(solution.worst_step_residual, residual)
        solution.newton_iterations += int(out.iterations)
        if on_floor:
            # The SOLVE diagnostic: like `worst_step_residual` it includes discarded probes.
            solution.floor_converged_times.append(t_new)
        if not converged:
            solution.nonconverged_probe_times.append(t_new)
        emit('solve', t=t_new, dt=dt_step, residual=residual, converged=converged, iterations=out.iterations)
        return out

    def checkpoint_now(target: float | None, event_count: int):
        """The portable record of the march at its last accepted state, in this plan's clock."""
        eec_state = None
        event_state = None
        lattice_origin = None
        if eec:
            tangent = None
            if eec_nonlinear_first_order:
                tangent = (
                    CycleTangentCSR.of(integrated_tangent)
                    if integrated_tangent is not None
                    else CycleTangentCSR((u0.size, u0.size), (0,) * (u0.size + 1), (), ())
                )
            eec_state = EECCheckpointState(
                eec_origin,
                next_correction,
                tuple(float(value) for value in cycle_start),
                float(eec_period),
                int(eec_sign),
                bootstrap_pending=integrator.name == 'bdf2' and len(state.us) == 1,
                cycle_tangent=tangent,
                cycle_converged=cycle_converged if eec_nonlinear_first_order else True,
            )
        elif plan == 'nominal':
            lattice_origin = origin
            pending = target if target is not None and target > state.t else None
            event_state = replace(
                locator_policy,
                nominal_index=k,
                nominal_anchor=nominal_anchor,
                pending_target=pending,
                pending_events=event_count if pending is not None else 0,
                last_fired=tuple(sorted(last_fired.items())),
            )
        elif plan == 'lattice' and (step_times is None or step_origin is not None):
            lattice_origin = origin
        return _checkpoint_of(
            problem,
            state,
            # THE INTEGRATOR'S OWN STEP, not the clock difference: BDF2 forms its coefficients from the step
            # ratio `state.h` carries, and `t_new - t_prev` sits ulps off it (0.0010000000000000009 for a
            # 1 ms step), so a resume restored from the difference marched other coefficients -- MEASURED
            # on the sampled terminal controller, whose ticks restart the history every other step.
            accepted_dt if state.h is None else state.h,
            checkpoint_key,
            integrator.name,
            # A resume whose window is under the end tolerance takes no step (a source corner one ulp
            # before t1): its accepted step is still the restart's, and so is the instant before it.
            previous_time=solution.times[-2] if len(solution.times) > 1 else restart.previous_time,
            eec_state=eec_state,
            step_origin=lattice_origin,
            event_state=event_state,
            step_control_state=None
            if step_control is None
            else StepControlCheckpointState.of(lte_history, h_ctrl, step_control),
            newton_tolerance=float(tol),
        )

    def emit_checkpoint(target: float | None, event_count: int) -> None:
        if checkpoint_key is None or accepted_dt is None:
            return
        record = checkpoint_now(target, event_count)
        solution.checkpoint = record
        if on_checkpoint is not None:
            on_checkpoint(record)

    def _commit(t_new: float, outcome: StepOutcome, *, observe: bool = True) -> bool:
        """Append an accepted point: material history, labels, accepted observers; True on a law change."""
        nonlocal state, accepted_dt
        u_new, rate = outcome.u, outcome.rate
        failed = not outcome.converged
        if failed and on_nonconverged == 'raise':
            emit('nonconverged', t=t_new, residual=outcome.residual)
            raise StepNotConverged(
                t_new, outcome.residual, u_new, solution.times[-1], solution.states[-1], worst_row=outcome.worst_row
            )
        if failed and (on_accept is not None or on_landing is not None):
            msg = (
                f'a nonconverged step cannot advance accepted observations (t={t_new:.9g}, '
                f'dt={t_new - solution.times[-1]:.3g}, residual {outcome.residual:.3e})'
            )
            raise ValueError(msg)
        accepted_dt = t_new - solution.times[-1]
        if prepared is not None:
            prepared.accept(outcome.candidate)
        # THE STEP JOINED THE TRAJECTORY, so its material history is the one the next step starts
        # from -- including a step labelled non-converged: the march continues from `u_new`.
        if capabilities.history:
            problem.commit_history(u_new, t_new)
        solution.times.append(t_new)
        solution.states.append(u_new)
        assert solution.state_rates is not None
        solution.state_rates.append(rate)
        assert solution.residual_times is not None
        solution.residual_times.append(outcome.residual_time)
        emit('step', t=t_new, dt=accepted_dt, index=len(solution.times) - 1)
        check_budget()
        lte_history.append((t_new, u_new))
        del lte_history[: -(integrator.order + 1)]
        if failed:
            emit('nonconverged', t=t_new, residual=outcome.residual)
            for j in range(len(solution.nonconverged_probe_times) - 1, -1, -1):
                if solution.nonconverged_probe_times[j] == t_new:
                    del solution.nonconverged_probe_times[j]
                    solution.nonconverged_times.append(t_new)
                    # The CONVERGENCE CONTRACT, label-don't-block: one warning per landed exhaustion.
                    logger.warning('step at t=%s did not converge in %s Newton iteration(s)', t_new, max_newton)
                    break
        if not observe:
            return False
        changed = False
        if on_landing is not None:
            landed = accepted_landing(on_landing, t_new, u_new)
            if landed:
                solution.landed_events.extend((int(index), t_new) for index in landed)
                changed = True
        if on_accept is not None and accepted_observation(on_accept, t_new, u_new):
            changed = True
        if changed:
            solution.accepted_discontinuities.append(t_new)
            projection = accepted_projection(project_accepted_state, t_new, u_new)
            right_state = projection.state.copy()
            if project_accepted_state is not None:
                assert solution.rate_state_overrides is not None
                solution.rate_state_overrides[len(solution.state_rates) - 1] = u_new.copy()
                solution.states[-1] = right_state
            state = restart_prepared(problem, integrator, state, projection, t_new)
            lte_history[:] = [(t_new, right_state)]
        return changed

    def after_accept(t_new: float, step_dt: float, converged: bool, before: IntegratorState, target) -> None:
        """The per-accepted-state duties every plan shares: EEC cycle, steady stop, checkpoint stream."""
        nonlocal state, cycle_start, integrated_tangent, cycle_converged, next_correction, stop
        u_new = solution.states[-1]
        if eec:
            if integrator.name == 'bdf2' and len(before.us) == 1:
                cycle_start = 0.5 * (before.last_u + u_new)
            if eec_nonlinear_first_order:
                cycle_converged = cycle_converged and converged
                zero = np.zeros_like(u_new)
                tangent = problem.jacobian(u_new, zero, zero, t_new, DerivativeWeights(alpha0=1.0)) * step_dt
                integrated_tangent = tangent if integrated_tangent is None else integrated_tangent + tangent
        steady = (
            steady_period is not None
            and not solution.nonconverged_times
            and periodic_converged(solution.times, solution.states, steady_period, steady_atol, steady_rtol)
        )
        if eec and abs(t_new - next_correction) <= eec_period * 1e-12:
            boundary = bdf2_boundary_moment(state.us) if integrator.name == 'bdf2' else u_new
            if converged and (not eec_nonlinear_first_order or cycle_converged):
                corrected = (
                    nonlinear_error_correction(problem, boundary, cycle_start, t_new, integrated_tangent, solve=solve)
                    if eec_nonlinear_first_order
                    else explicit_error_correction(
                        problem, boundary, cycle_start, eec_period, t_new, solve=solve, sign=eec_sign
                    )
                )
                # The correction is an algorithmic state jump, not a physical time step: shift every
                # stored history state equally, preserving BDF2 rather than bootstrapping BE.
                shift = corrected - boundary
                state = replace(state, us=[value + shift for value in state.us])
                solution.corrected_times.append(t_new)
                cycle_start = corrected.copy()
            else:
                cycle_start = boundary.copy()
            next_correction = eec_origin + (round((next_correction - eec_origin) / eec_period) + 1) * eec_period
            integrated_tangent, cycle_converged = None, True
        emit_checkpoint(target, events_this_cell)
        if steady:
            stop = True

    # ---------------------------------------------------------------- the march
    # A clipped cell RE-ANCHORS the nominal grid at the breakpoint (anchor + integer * dt keeps the grid
    # grid going from there); without the re-anchor the nominal targets drift ahead of the physics
    # (SR-022). A while loop because clipping inserts grid points: the march ends at t1, not at a
    # fixed step count. Only roundoff at the declared end may terminate it -- the SPAN's roundoff and
    # the end clock's own ulps, never a fraction of the clock's MAGNITUDE: `1e-12 * |t0|` is 1 s at a
    # 1e12 s origin, which ended a 4 ms march before its first step (`test_axisymmetric_checkpoint`).
    end_tol = max(1e-12 * (window.t1 - window.t0), 4.0 * float(np.spacing(abs(window.t1))))
    events_this_cell = 0
    try:
        while not stop and (state.t < window.t1 - end_tol or (len(observations) > 0 and state.t < observations[-1])):
            continuing_cell = resumed_target is not None
            lattice_cell = False
            if plan == 'curvature':
                h_curve = min(h_curve, window.t1 - state.t)
                proposed = state.t + h_curve
                t_target = next_deadline(next_breakpoint, state.t, proposed)
                h_curve = t_target - state.t
                if curve_lattice is not None:
                    if t_target < proposed:
                        curve_lattice.tick(t_target)
                    current_tick = curve_lattice.tick(state.t)
                    count = min(curve_lattice.steps(h_curve), curve_lattice.tick(window.t1) - current_tick)
                    h_curve = count * curve_lattice.quantum
                    t_target = curve_lattice.time(current_tick + count)
                    if not t_target > state.t:
                        msg = 'curvature time lattice cannot make representable clock progress'
                        raise ValueError(msg)
                pending_dt = t_target - state.t
            elif plan == 'lattice':
                while lattice_index < len(lattice_targets) and (
                    lattice_targets[lattice_index] <= state.t
                    or (step_times is None and same_instant(lattice_targets[lattice_index], state.t))
                ):
                    lattice_index += 1
                if lattice_index >= len(lattice_targets):
                    break
                t_target = lattice_targets[lattice_index]
                lattice_cell = True
                if step_times is None:
                    lattice_origin = eec_origin if eec else origin
                    pending_dt = fixed_lattice_step_size(lattice_origin, state.t, t_target, dt)
                else:
                    pending_dt = t_target - state.t
            elif continuing_cell:
                t_target = min(resumed_target, window.t1)
                resumed_target = None
                pending_dt = t_target - state.t
            else:
                t_target = nominal_anchor + (k + 1) * dt
                # dt exactly on a canonical cell (not t_target - state.t, which is off by ulps from dt).
                pending_dt = dt if nominal_anchor + k * dt == state.t else t_target - state.t
                if same_instant(t_target, window.t1):
                    # A full nominal cell retains its weights; its time label is the declared endpoint.
                    t_target = window.t1
                elif t_target > window.t1:
                    t_target = window.t1
                    pending_dt = t_target - state.t
            if plan == 'nominal' and not continuing_cell:
                k += 1
            events_this_cell = resumed_events if continuing_cell else 0
            resumed_events = 0
            while True:  # one cell, possibly interrupted by events, rejections and landings
                # W1b: a time breakpoint inside the cell clips the target -- a known non-smooth instant
                # is LANDED on the grid, never sampled past.
                while bp_index < len(breakpoints) and breakpoints[bp_index] <= state.t:
                    bp_index += 1
                if bp_index < len(breakpoints) and breakpoints[bp_index] <= t_target + bracket_tol:
                    # Prefer the exact declared edge to a nearly coincident grid point; land it from the
                    # preceding state, never via a tiny extra step.
                    t_target = breakpoints[bp_index]
                    pending_dt = t_target - state.t
                    if not lattice_cell:
                        nominal_anchor, k = t_target, 0
                if plan != 'curvature':
                    deadline = owned_deadline(next_breakpoint, state.t)
                    if deadline is not None and t_target == window.t1 and same_instant(deadline, window.t1):
                        deadline = window.t1  # The explicitly requested window end is the hard authority.
                    elif deadline is not None and lattice_cell and same_instant(deadline, t_target):
                        deadline = t_target  # This plan's target is itself a declared lattice instant.
                    if (
                        deadline is not None
                        and deadline <= window.t1
                        and (deadline <= t_target or (not lattice_cell and same_instant(deadline, t_target)))
                    ):
                        # The owner's live deadline clips the cell; a nominal grid re-anchors there, a
                        # lattice keeps its absolute targets and resumes with the true remainder.
                        t_target = deadline
                        pending_dt = t_target - state.t
                        if not lattice_cell:
                            nominal_anchor, k = t_target, 0
                while observation_index < len(observations) and observations[observation_index] <= state.t:
                    observation_index += 1
                lands_on_breakpoint = t_target in breakpoint_set
                if observation_index < len(observations) and (
                    observations[observation_index] < t_target
                    or observations[observation_index] == t_target
                    or (not lands_on_breakpoint and same_instant(observations[observation_index], t_target))
                ):
                    # A nominal endpoint and a requested endpoint separated only by roundoff are one cell,
                    # solved at the requested time; a declared breakpoint is exact, so a distinct
                    # neighbour is its own cell.
                    t_target = observations[observation_index]
                    pending_dt = t_target - state.t
                    nominal_anchor, k = t_target, 0
                checkpoint = _take_checkpoint(state, solution)
                t_from = state.t
                step_dt = pending_dt
                t_step = t_target
                # Event-location precision cannot suppress an LTE-requested smaller step.
                if step_control is not None and h_ctrl < pending_dt:
                    step_dt = h_ctrl
                    t_step = t_from + step_dt
                probe = step_from(state, step_dt, t_step)
                probe_state, u_probe, step_ok = probe.state, probe.u, probe.converged
                if plan == 'curvature':
                    if len(checkpoint.state.us) >= 2:
                        u_prev = np.asarray(checkpoint.state.us[-2], dtype=float)
                        u_extrap = checkpoint.state.last_u + (checkpoint.state.last_u - u_prev) * (
                            step_dt / h_prev_curve
                        )
                        err = max(float(np.linalg.norm(u_probe - u_extrap)), 1e-300)
                    else:
                        err = 0.0  # the first step after a start: no history to extrapolate from
                    # An absolute lattice owns integer intervals. Subtracting its float time labels
                    # can put a floor tick just above h_lo and retry the identical tick forever.
                    at_floor = (
                        curve_lattice.tick(t_step) - curve_lattice.tick(t_from) <= floor_ticks
                        if curve_lattice is not None
                        else step_dt <= h_lo
                    )
                    if not (at_floor or (step_ok and (err == 0.0 or err <= curvature_atol))):
                        # Reject: halve and propose again, bounded by h_lo where the step is taken with its
                        # labels rather than spun forever.
                        h_curve = max(h_lo, step_dt / 2.0)
                        emit('reject', t=t_step, dt=step_dt, error_ratio=err / curvature_atol, next_dt=h_curve)
                        break
                    state = probe_state
                    _commit(t_step, probe)
                    if err > curvature_atol:
                        solution.indicator_unmet_times.append(t_step)
                    h_prev_curve = step_dt
                    if err > 0.0:
                        h_curve = min(h_hi, max(h_lo, step_dt * min(2.0, 1.2 * (curvature_atol / err) ** 0.5)))
                    after_accept(t_step, step_dt, step_ok, checkpoint.state, None)
                    break
                if adaptive and not step_ok:
                    # REJECT-HALVE-RETRY: only a step that ACTUALLY failed is subdivided. Each rejected
                    # attempt stays in `nonconverged_probe_times`: an iterate the trajectory excludes.
                    while not step_ok and 0.5 * step_dt >= minimum_step:
                        step_dt *= 0.5
                        t_step = t_from + step_dt
                        probe = step_from(state, step_dt, t_step)
                        probe_state, u_probe, step_ok = probe.state, probe.u, probe.converged
                g_new = [float(g(u_probe, t_step)) for g in event_fns]
                crossed = [
                    i
                    for i in range(len(event_fns))
                    if _crossed(event_fns[i], g_prev[i], g_new[i]) and t_step - last_fired.get(i, -np.inf) > bracket_tol
                ]
                if not crossed:
                    halves = None
                    if step_control is not None and step_ok:
                        order = integrator.order
                        if len(lte_history) >= order + 1:
                            ratio = _lte_ratio(
                                lte_history, t_step, u_probe, order, integrator.error_constant, step_control
                            )
                        else:
                            # BOOTSTRAP (no predictor history after a start or a restart): STEP DOUBLING.
                            # The bootstrap step is BDF1-class whatever the scheme, so the estimate takes order 1.
                            order, ratio = 1, 0.0
                            h2 = 0.5 * step_dt
                            mid = step_from(checkpoint.state.copy(), h2, t_from + h2)
                            end = step_from(mid.state, h2, t_step) if mid.converged else None
                            if end is not None and end.converged:
                                g_mid = [float(g(mid.u, t_from + h2)) for g in event_fns]
                                g_end = [float(g(end.u, t_step)) for g in event_fns]
                                if all(
                                    _sign(a) == _sign(b) == _sign(c)
                                    for a, b, c in zip(g_prev, g_mid, g_end, strict=True)
                                ):
                                    scale = step_control.atol + step_control.rtol * np.maximum(
                                        np.abs(state.last_u), np.abs(end.u)
                                    )
                                    ratio = float(np.max(np.abs(end.u - u_probe) * 2.0 / scale)) if scale.size else 0.0
                                    halves = (mid, end, g_end)
                        if ratio > 1.0:
                            if step_dt <= h_floor * (1.0 + 1e-9):
                                msg = (
                                    f'step control: the step at t={t_from:g} is still {ratio:.3g}x over its '
                                    f'local-error tolerance at h_min={h_floor:g}; the state reached is t={t_from:g}. '
                                    'An undeclared discontinuity (a source edge without a breakpoint, a switch '
                                    'without an event) is the usual cause.'
                                )
                                raise RuntimeError(msg)
                            h_ctrl = max(h_floor, _resize(step_dt, ratio, order))
                            emit('reject', t=t_step, dt=step_dt, error_ratio=ratio, next_dt=h_ctrl)
                            continue
                        grown = _resize(step_dt, ratio, order)
                        h_ctrl = max(h_ctrl, grown) if step_dt < h_ctrl else grown
                    if refine is not None and step_ok and step_dt > 2.0 * minimum_step:
                        # Step-doubling refinement: re-take the step at half size from the same
                        # checkpoint; a large discrepancy means the full step under-resolved the
                        # dynamics, and the two half-steps replace it.
                        h2 = 0.5 * step_dt
                        t_mid = t_from + h2
                        mid = step_from(checkpoint.state.copy(), h2, t_mid)
                        end = step_from(mid.state, h2, t_step) if mid.converged else None
                        if end is not None and end.converged:
                            g_mid = [float(g(mid.u, t_mid)) for g in event_fns]
                            g_end = [float(g(end.u, t_step)) for g in event_fns]
                            crossed_mid = any(
                                _crossed(event_fns[i], g_prev[i], g_mid[i])
                                and t_mid - last_fired.get(i, -np.inf) > bracket_tol
                                for i in range(len(event_fns))
                            )
                            crossed_end = any(
                                _crossed(event_fns[i], g_mid[i], g_end[i])
                                and t_step - last_fired.get(i, -np.inf) > bracket_tol
                                for i in range(len(event_fns))
                            )
                            scale = max(float(np.max(np.abs(u_probe))) if u_probe.size else 1.0, 1e-12)
                            if (
                                not (crossed_mid or crossed_end)
                                and float(np.max(np.abs(end.u - u_probe))) > refine * scale
                            ):
                                if (
                                    on_accept is not None
                                    or on_landing is not None
                                    or capabilities.history
                                    or checkpoint_key is not None
                                ):
                                    # The midpoint can change the physical law: commit it with its own
                                    # derivative state, then re-march -- the speculative second half is
                                    # no longer authoritative.
                                    state = mid.state
                                    _commit(t_mid, mid)
                                    after_accept(t_mid, h2, True, checkpoint.state, t_target)
                                    g_prev = [float(g(state.last_u, t_mid)) for g in event_fns]
                                    if stop:
                                        break
                                    pending_dt = t_target - state.t
                                    continue
                                halves = (mid, end, g_end)
                    if halves is not None and (on_accept is not None or on_landing is not None or capabilities.history):
                        # An accepted observer sees the midpoint, which may change its law: commit it,
                        # then re-march the rest of the cell from the canonical midpoint state.
                        mid, _end, _g = halves
                        state = mid.state
                        t_mid = t_from + 0.5 * step_dt
                        _commit(t_mid, mid)
                        after_accept(t_mid, 0.5 * step_dt, True, checkpoint.state, t_target)
                        g_prev = [float(g(state.last_u, t_mid)) for g in event_fns]
                        if stop:
                            break
                        pending_dt = t_target - state.t
                        continue
                    if halves is not None:  # the doubled pair is the more accurate answer: keep it
                        mid, end, g_new = halves
                        state = end.state
                        before = checkpoint.state
                        _commit(t_from + 0.5 * step_dt, mid)
                        _commit(t_step, end)
                    else:
                        state = probe_state
                        before = checkpoint.state
                        _commit(t_step, probe)
                    if bp_index < len(breakpoints) and t_step == breakpoints[bp_index]:
                        projection = AcceptedProjectionResult(state.last_u)
                        if on_breakpoint is not None:
                            if solution.nonconverged_times:
                                msg = 'a physical breakpoint cannot consume a non-converged history'
                                raise RuntimeError(msg)
                            edge_state = state.last_u.copy()
                            right_state = edge_state.copy()
                            selected = on_breakpoint(t_step, right_state)
                            if (
                                not isinstance(selected, AcceptedProjectionResult)
                                and not np.isfinite(right_state if selected is None else np.asarray(selected)).all()
                            ):
                                msg = 'a breakpoint handler produced a non-finite state'
                                raise ValueError(msg)
                            projection = (
                                selected
                                if isinstance(selected, AcceptedProjectionResult)
                                else AcceptedProjectionResult(right_state if selected is None else selected)
                            )
                            if projection.state.shape != edge_state.shape:
                                msg = 'a breakpoint handler must preserve state shape'
                                raise ValueError(msg)
                            if not np.array_equal(edge_state, projection.state):
                                solution.rate_state_overrides[len(solution.times) - 2] = edge_state
                            solution.states[-1] = projection.state.copy()
                        # A declared source edge is as nonsmooth as a located state event; no
                        # integrator or error predictor may extrapolate its pre-edge history.
                        state = restart_prepared(problem, integrator, state, projection, t_step)
                        lte_history[:] = [(t_step, state.last_u)]
                    g_prev = [float(g(state.last_u, t_step)) for g in event_fns]
                    if t_step < t_target - bracket_tol:
                        # A SHORTENED step landed inside the cell: the march continues toward the same
                        # target with the true remainder.
                        after_accept(t_step, step_dt, step_ok, before, t_target)
                        if stop:
                            break
                        pending_dt = t_target - state.t
                        continue
                    after_accept(t_step, step_dt, step_ok, before, t_target)
                    break

                # Bisect each crossed function to its root; the EARLIEST instant wins, and every function
                # rooted within ``bracket_tol`` of it fires AT it -- the earliest-only rule would drop a
                # SIMULTANEOUS crossing (a dead-time H-bridge's complementary pair, hbridge_deadtime
                # 2026-08-07), and the dropped function would be stuck, not delayed.
                candidates: list[tuple[float, int]] = []
                # Every observer below queries the SAME deep-owned checkpoint/history, spacing, integrator
                # and unchanged problem, so the exact candidate time determines the whole trial form. The
                # cache lives only in this search; failed/nonfinite trials are not answers to reuse.
                trials: dict[float, StepOutcome] = {}
                trial_bytes = 0
                for i in crossed:
                    if g_new[i] == 0.0:
                        candidates.append((t_step, i))
                    elif t_step - t_from <= bracket_tol:
                        candidates.append((t_from, i))
                    else:

                        def _probe(mid_t: float, i: int = i, t_from: float = t_from, checkpoint=checkpoint) -> float:
                            nonlocal trial_bytes
                            if mid_t in trials:
                                trial = trials[mid_t]
                                u_mid, ok, _rate = trial.u, trial.converged, trial.rate
                            else:
                                trial_state = _rollback(checkpoint, solution)
                                trial = step_from(trial_state, mid_t - t_from, mid_t)
                                u_mid, ok, _rate = trial.u, trial.converged, trial.rate
                                size = u_mid.nbytes + _rate.nbytes
                                if (
                                    ok
                                    and np.isfinite(u_mid).all()
                                    and np.isfinite(_rate).all()
                                    and size <= _EVENT_TRIAL_CACHE_BYTES
                                ):
                                    while trials and trial_bytes + size > _EVENT_TRIAL_CACHE_BYTES:
                                        old_trial = trials.pop(next(iter(trials)))
                                        trial_bytes -= old_trial.u.nbytes + old_trial.rate.nbytes
                                    snapshot, rate = u_mid.copy(), _rate.copy()
                                    snapshot.setflags(write=False)
                                    rate.setflags(write=False)
                                    trials[mid_t] = replace(trial, state=trial.state.copy(), u=snapshot, rate=rate)
                                    trial_bytes += size
                            if adaptive and not ok:
                                msg = (
                                    f'event probe at t={mid_t:g} did not converge; its event sign cannot '
                                    'locate a physical switch. Reduce the step or repair the nonlinear solve.'
                                )
                                raise RuntimeError(msg)
                            return float(event_fns[i](u_mid.copy(), mid_t))

                        lo_t, hi_t = _locate_bracket(
                            t_from, float(g_prev[i]), t_step, float(g_new[i]), _probe, tol=bracket_tol
                        )
                        candidates.append((0.5 * (lo_t + hi_t), i))
                # No observer may retain a trial across an accepted state or physical reset.
                trials.clear()
                # The suppression's SECOND half, where the landing INSTANT is known: a crossing located
                # within the tolerance of the function's last firing is that instant's re-detection.
                candidates = [c for c in candidates if c[0] - last_fired.get(c[1], -np.inf) > bracket_tol]
                if not candidates:
                    state = probe_state
                    before = checkpoint.state
                    _commit(t_step, probe)
                    g_prev = [float(g(state.last_u, t_step)) for g in event_fns]
                    after_accept(t_step, step_dt, step_ok, before, t_target)
                    if stop:
                        break
                    if t_step < t_target - bracket_tol:
                        pending_dt = t_target - state.t
                        continue
                    break
                t_star = min(c[0] for c in candidates)
                simultaneous = [c for c in candidates if c[0] - t_star <= bracket_tol]
                state = _rollback(checkpoint, solution)
                if t_star > t_from:  # land the march exactly on the event
                    landing = step_from(state, t_star - t_from, t_star)
                    state, u_star, landed = landing.state, landing.u, landing.converged
                    if not landed:
                        msg = (
                            f'event landing at t={t_star:g} did not converge; its state cannot '
                            'trigger a physical switch. Reduce the step or repair the nonlinear solve.'
                        )
                        raise RuntimeError(msg)
                    _commit(t_star, landing, observe=False)
                else:
                    u_star = np.array(state.last_u, dtype=float, copy=True)
                if on_event is not None and solution.nonconverged_times:
                    msg = (
                        f'physical event at t={t_star:g} cannot consume a history containing '
                        f'non-converged steps {solution.nonconverged_times}. Repair the failed solve '
                        'before advancing a physical handler.'
                    )
                    raise RuntimeError(msg)
                residual_state = u_star.copy() if on_event is not None and t_star > t_from else None
                for _root, i in simultaneous:
                    solution.events.append((i, t_star))
                    emit('event', t=t_star, function=i)
                    if on_event is not None:
                        on_event(i, t_star, u_star)
                    last_fired[i] = t_star
                # Located state events own the incoming state first; exact scheduled clocks then consume
                # that same canonical landing, never a probe.
                if on_landing is not None:
                    landed_ids = accepted_landing(on_landing, t_star, u_star)
                    if landed_ids:
                        solution.landed_events.extend((int(index), t_star) for index in landed_ids)
                        solution.accepted_discontinuities.append(t_star)
                if residual_state is not None and not np.array_equal(residual_state, u_star, equal_nan=True):
                    assert solution.state_rates is not None
                    assert solution.rate_state_overrides is not None
                    rate_index = len(solution.state_rates) - 1
                    if rate_index >= 0 and rate_index not in solution.rate_state_overrides:
                        residual_state.flags.writeable = False
                        solution.rate_state_overrides[rate_index] = residual_state
                # Select the actual outgoing pair before restarting either numerical owner.
                projection = AcceptedProjectionResult(u_star)
                if on_accept is not None and accepted_observation(on_accept, t_star, u_star):
                    solution.accepted_discontinuities.append(t_star)
                if project_accepted_state is not None:
                    projection = accepted_projection(project_accepted_state, t_star, u_star)
                    right_state = projection.state.copy()
                    rate_index = len(solution.state_rates) - 1
                    if t_star > t_from and rate_index not in solution.rate_state_overrides:
                        solution.rate_state_overrides[rate_index] = u_star.copy()
                    solution.states[-1] = right_state
                    u_star = right_state
                state = restart_prepared(problem, integrator, state, projection, t_star)
                lte_history[:] = [(t_star, u_star)]
                g_prev = [float(g(u_star, t_star)) for g in event_fns]
                pending_dt = t_target - t_star  # the genuine remainder of this cell
                events_this_cell += 1
                emit_checkpoint(t_target, events_this_cell)
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
            # end of one cell
        if held is not None:
            logger.info('held-factor march: factorizations=%d solves=%d', held.factorizations, held.solves)
        if checkpoint_key is not None and accepted_dt is not None:
            solution.checkpoint = checkpoint_now(None, 0)
        solution.final_integrator_state = state.copy()
        return solution
    finally:
        if prepared is not None:
            prepared.close()
