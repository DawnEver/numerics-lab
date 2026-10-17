"""Periodic steady state, physics-blind: convergence on accepted states, and the period-map fixed point.

TWO MECHANISMS, ONE SUBJECT, joined at an integration merge (2026-10-07), where two branches had each
created this module for a different half of "reach the periodic steady state":

* EXPLICIT ERROR CORRECTION on a time-marched field solve (including nonuniform grids):
  :func:`periodic_converged`, :func:`explicit_error_correction`, :func:`nonlinear_error_correction`
  and the restart identity checks :func:`canonical_eec_window` / :func:`validate_eec_restart`.
* THE PERIOD MAP'S FIXED POINT for a forced circuit: :func:`periodic_fixed_point`, by reduced rank
  extrapolation, returning a :class:`PeriodicFixedPoint`.

The second half's own description follows.
Periodic steady state as a fixed point of the one-period map -- physics-blind.

A circuit driven at period ``T`` reaches a periodic steady state when its state at the start of a
period equals the state one period later: ``x* = P(x*)`` where ``P`` marches one period. Marching
until that happens costs the slowest time constant over ``T`` periods -- a filter capacitor that
settles in 10^4 switching periods costs 10^4 marches. SIMBA's steady-state finder exists for that
cost; this is ours, and it is a statement about ``P`` only: what a period IS, and how to march one,
is the caller's.

The accelerator is REDUCED RANK EXTRAPOLATION (RRE, Skelboe's choice for exactly this problem):
from ``k + 1`` successive map values ``x_0 .. x_{k+1}`` with differences ``d_j = x_{j+1} - x_j``,
the extrapolated fixed point is ``sum_j gamma_j x_j`` with ``gamma`` minimising ``|sum_j gamma_j
d_j|`` under ``sum_j gamma_j = 1``. For an affine ``P`` it is EXACT once ``k`` reaches the number of
slow modes; for a switched (piecewise-affine) ``P`` it is a Newton-like cycle whose rate is set by
how affine ``P`` is near the orbit. No Jacobian, no state-variable bookkeeping -- the map is opaque.

The answer carries its own residual ``|P(x*) - x*|``: a non-converged search RAISES, by the same
rule every solve here follows -- a periodic state that is not periodic is not an answer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._forms import TransientSolution
from numerics_lab.time_integrators.instants import same_instant

__all__ = [
    'PeriodicFixedPoint',
    'bdf2_boundary_moment',
    'canonical_eec_window',
    'explicit_error_correction',
    'nonlinear_error_correction',
    'periodic_converged',
    'periodic_fixed_point',
    'periodic_steady_state',
    'smooth_phase',
    'validate_eec_restart',
]


def canonical_eec_window(window, restart, *, uniform_clock=False):
    """Resolve a declared uniform tick to its one original-clock representation: ``(window, restart)``.

    THE IDENTITY IS THE INTEGER TICK, not a float time. The checkpoint's own ``time`` (a first run
    stopped at its ``t1``) and the producing lattice's ``origin + k dt`` can differ in the last ULP
    for the SAME tick ``k``; an exact float comparison refused every mid-cycle JSON restart on that
    representation alone. Both are therefore returned on the lattice representation, so the resumed
    grid is the uninterrupted run's grid and the checkpoint identity check compares like with like.
    """
    origin = window.t0
    if restart is not None and restart.eec_state is not None:
        origin = restart.eec_state.origin_time
        phase = (window.t0 - origin) / window.dt
        if not np.isfinite(phase):
            msg = 'EEC restart time must align with the original uniform integration clock'
            raise ValueError(msg)
        time = origin + round(phase) * window.dt
        if not same_instant(window.t0, time):
            msg = 'EEC restart time must align with the original uniform integration clock'
            raise ValueError(msg)
        checkpoint_phase = (restart.time - origin) / window.dt
        if not np.isfinite(checkpoint_phase) or round(checkpoint_phase) != round(phase):
            msg = 'EEC restart original-clock tick does not match the checkpoint time'
            raise ValueError(msg)
        window, restart = replace(window, t0=time), replace(restart, time=time)
    if uniform_clock and np.isfinite(window.dt) and window.dt > 0:
        phase = (window.t1 - origin) / window.dt
        if not np.isfinite(phase):
            msg = 'EEC endpoint must align with its declared uniform integration clock'
            raise ValueError(msg)
        endpoint = origin + round(phase) * window.dt
        if not same_instant(window.t1, endpoint):
            msg = (
                f'EEC endpoint must align with its declared uniform {window.integrator.value.upper()} integration clock'
            )
            raise ValueError(msg)
        if endpoint != window.t1:
            window = replace(window, t1=endpoint)
    return window, restart


def validate_eec_restart(restart, period, sign, dt, nonlinear=False):
    """Reject missing or changed algorithm controls before restoring physical history."""
    if restart is None:
        return
    if (restart.eec_state is None) != (period is None):
        msg = 'EEC restart requires its saved cycle-origin history and active correction controls'
        raise ValueError(msg)
    if restart.eec_state is not None:
        restart.eec_state.validate_controls(period, sign, nonlinear)
        phase = (restart.time - restart.eec_state.origin_time) / dt
        if not np.isfinite(phase) or not np.isclose(phase, round(phase), rtol=0.0, atol=1e-9):
            msg = 'EEC restart time must align with the original uniform integration clock'
            raise ValueError(msg)


def bdf2_boundary_moment(history):
    """Uniform BDF2 telescopes to z_end-z_start, z=1.5u_n-0.5u_(n-1).

    With a backward-Euler first step, the initial boundary moment instead is
    0.5*(u_0+u_1); the driver captures it after that physical bootstrap solve.
    A constant coarse shift p of BOTH history states shifts z by p without
    destroying the BDF2 derivative offsets. This affine symmetric K/M coarse
    specialization is not a derivation of JMAG's proprietary implementation.
    """
    return 1.5 * history[-1] - 0.5 * history[-2]


def nonlinear_error_correction(problem, current, previous, time, integrated_tangent, *, solve):
    """Uniform BE coarse linearization using the actual cycle-integrated Jacobian.

    COMPUMAG 2019 PA-A5-9 II.B equation (5), single process/full period:
    ``dt sum J_g(u_i) p = M (u_end-u_start)``. This local coarse approximation
    has no global contraction guarantee and is not proprietary HTP-EEC.
    """
    zero = np.zeros_like(current)
    mass = problem.jacobian(current, zero, zero, time, DerivativeWeights(alpha0=0.0, alpha1=1.0))
    return current + np.asarray(solve(integrated_tangent, mass @ (current - previous)), dtype=float)


def explicit_error_correction(problem, current, previous, period, time, *, solve, sign=1):
    """First-order TP-EEC: remove the estimated cycle-mean transient error.

    For ``M u' + K u = f(t)``, integration over one supply period gives
    ``K mean(error) = -M (u_end-u_start)/T``. Adding ``K^-1 M delta/T``
    removes that mean error. This is EEC, not endpoint extrapolation or an
    exact periodic shooting solve; it is most effective for long time constants.
    The caller must supply a time-invariant affine first-order problem.

    Native half-period specialization (sign=-1) of the uniform backward-Euler
    coarse correction in Takahashi et al., COMPUMAG 2019, II.B, equations (3)-(5):
    https://www.compumag.org/Proceedings/2019_Paris/files/papers/PA-A5-9.pdf
    ``(n K + 2 M/dt) p = -(M/dt)(u_end+u_start)``. The correction is
    injected at the ENDPOINT'S advanced phase: ``u_end+p``, never ``-u_end+p``.
    This scholarly affine specialization is not proof of JMAG's proprietary
    implementation and implements neither reverse rotor correction nor HTP-EEC.
    """
    if isinstance(sign, bool) or not isinstance(sign, Integral) or sign not in (-1, 1):
        msg = 'EEC sign must be the explicit integer +1 or -1'
        raise ValueError(msg)
    if not isinstance(period, Real) or not np.isfinite(period) or period <= 0:
        msg = 'EEC correction interval must be a positive finite scalar'
        raise ValueError(msg)
    zero = np.zeros_like(current)
    stiffness = problem.jacobian(current, zero, zero, time, DerivativeWeights(alpha0=1.0))
    mass = problem.jacobian(current, zero, zero, time, DerivativeWeights(alpha0=0.0, alpha1=1.0))
    if sign == -1:
        correction = solve(stiffness + 2 * mass / period, -mass @ ((current + previous) / period))
        return current + np.asarray(correction, dtype=float)
    return current + np.asarray(solve(stiffness, mass @ ((current - previous) / period)), dtype=float)


def periodic_converged(times, states, period: float, atol: float, rtol: float) -> bool:
    """Compare every sample in the last cycle with interpolated previous-cycle states.

    Two complete cycles are mandatory. Comparing endpoints alone would mislabel a
    changing waveform whose endpoints happen to agree as steady.
    """
    if times[-1] - times[0] < 2 * period - 1e-12 * period:
        return False
    ts = np.asarray(times)
    us = np.asarray(states)
    recent = ts >= ts[-1] - period
    targets = ts[recent] - period
    previous = np.stack([np.interp(targets, ts, col) for col in us.reshape(len(ts), -1).T], axis=1)
    current = us[recent].reshape(len(targets), -1)
    return bool(np.all(np.abs(current - previous) <= atol + rtol * np.maximum(np.abs(current), np.abs(previous))))


@dataclass(frozen=True)
class PeriodicFixedPoint:
    """The periodic start state, its relative residual, and how many one-period maps it cost."""

    state: NDArray[np.floating]
    residual: float
    map_calls: int


def _rre(xs: list[NDArray[np.floating]]) -> NDArray[np.floating]:
    """Reduced rank extrapolation from successive map values (``len(xs) >= 3``).

    The constraint ``sum gamma = 1`` is eliminated against the LAST difference, so the solve is an
    ordinary least squares on the differences themselves -- never on their Gram matrix, whose
    squared condition number stalls exactly the slow modes this exists to extrapolate.
    """
    x = np.asarray(xs)
    d = np.diff(x, axis=0)  # (k+1, n)
    basis = (d[:-1] - d[-1]).T  # gamma = e_last + sum_j c_j (e_j - e_last)
    c, *_ = np.linalg.lstsq(basis, -d[-1], rcond=None)
    gamma = np.append(c, 1.0 - c.sum())
    return gamma @ x[:-1]


def periodic_fixed_point(
    period_map: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    *,
    tol: float = 1e-9,
    rank: int = 6,
    max_cycles: int = 50,
    atol: float = 0.0,
    admissible: Callable[[NDArray[np.floating]], bool] | None = None,
) -> PeriodicFixedPoint:
    """Find ``x`` with ``|P(x) - x| <= tol * max(|x|, |P(x)|)`` (or ``<= atol``) by cycled RRE.

    ``atol`` is in the STATE's units and exists for an orbit that starts where the state is near
    zero (a sine-fed resistive circuit at its zero crossing), where the relative test compares
    round-off with round-off.

    ``admissible(x)`` lets the caller refuse an extrapolated start the map is not near-affine at
    (a controller output past its saturation): the cycle then continues from its last plain period
    instead. RRE assumes an affine map; a clipped actuator is the measured case where it is not.

    Each cycle marches ``rank + 1`` periods from the current estimate and extrapolates; the cycle
    stops as soon as one plain period already meets ``tol``. ``max_cycles`` exhausted RAISES with the
    residual reached, never returns a state that is not periodic.
    """
    if rank < 1 or max_cycles < 1:
        msg = f'rank and max_cycles must be positive, got rank={rank}, max_cycles={max_cycles}'
        raise ValueError(msg)
    if not np.isfinite(tol) or tol <= 0:
        msg = f'tol must be finite and positive, got {tol}'
        raise ValueError(msg)
    x = np.asarray(x0, dtype=float).copy()
    calls = 0
    residual = np.inf
    for _ in range(max_cycles):
        xs = [x]
        for _ in range(rank + 1):
            # A caller may march in place or reuse its output buffer; neither may alter history.
            xs.append(np.asarray(period_map(xs[-1].copy()), dtype=float).copy())
            calls += 1
            scale = max(float(np.max(np.abs(xs[-1]))), float(np.max(np.abs(xs[-2]))), np.finfo(float).tiny)
            difference = float(np.max(np.abs(xs[-1] - xs[-2])))
            residual = difference / scale
            if residual <= tol or difference <= atol:
                return PeriodicFixedPoint(state=xs[-2], residual=residual, map_calls=calls)
        candidate = _rre(xs)
        x = candidate if admissible is None or admissible(candidate) else xs[-1]
    msg = (
        f'no periodic steady state within {max_cycles} RRE cycles ({calls} periods): relative '
        f'residual {residual:.3e} > {tol:g}. The map may not be contractive (an unstable or '
        'subharmonic orbit), or the period is not the forcing period.'
    )
    raise RuntimeError(msg)


def smooth_phase(period: float, edges: Sequence[float]) -> float:
    """The middle of the widest edge-free gap in one period: where a period map may restart its history.

    Restarting the integrator ON a source edge integrates the edge differently from a march that
    spans it (measured on a pulse-driven LC: the restarted map settled at 4.018 V against the
    continuous march's 3.993 V). Away from every edge the restart is one smooth bootstrap step.
    """
    if not np.isfinite(period) or period <= 0:
        msg = f'period must be finite and positive, got {period}'
        raise ValueError(msg)
    marks = sorted({0.0, *(e % period for e in edges)})
    gaps = [((marks[(i + 1) % len(marks)] - m) % period or period, m) for i, m in enumerate(marks)]
    width, at = max(gaps)
    return (at + 0.5 * width) % period


def periodic_steady_state(
    march: Callable[[float, float, NDArray[np.floating], tuple[float, ...], bool], TransientSolution],
    period: float,
    x0: Callable[[float], NDArray[np.floating]] | NDArray[np.floating],
    *,
    edges: Sequence[float] = (),
    tol: float = 1e-9,
    rank: int = 6,
    max_cycles: int = 50,
    atol: float = 0.0,
) -> tuple[PeriodicFixedPoint, TransientSolution]:
    """THE forced periodic steady state: the period map's fixed point and one marched period of it.

    ``march(t0, t1, x, breakpoints, first)`` marches one window (``first`` is True only for the very
    first map, so a caller may reset its own switch state once). Successive maps march successive
    periods ``[phase + kT, phase + (k+1)T]``, so the returned orbit lies in the period after the last
    map, not in the first. The map starts at
    :func:`smooth_phase` -- never on a forcing edge -- and lands every edge of the period as a
    breakpoint. ``x0`` is the start state, or a callable returning it at the chosen phase. ``period``
    must be the forcing period of every source; a free-running orbit has no such clock.
    """
    phase = smooth_phase(period, edges)
    wrapped = sorted({e % period + (period if e % period <= phase else 0.0) for e in edges})
    bps = tuple(phase + period if abs(e - phase - period) < 1e-12 * period else e for e in wrapped)
    window = [0]

    def advance(x: NDArray[np.floating]) -> TransientSolution:
        # EACH MAP MARCHES THE NEXT PERIOD IN ABSOLUTE TIME. The problem is T-periodic, so the map is
        # the same one for every k -- and a device or controller that keeps an accepted-time history
        # (a switch's last event, a sampled controller's clock) never sees time run backwards.
        k = window[0]
        shift = k * period
        sol = march(phase + shift, phase + shift + period, x, tuple(b + shift for b in bps), k == 0)
        window[0] += 1
        return sol

    def period_map(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.asarray(advance(x).states[-1], dtype=float)

    start = x0(phase) if callable(x0) else np.asarray(x0, dtype=float)
    fixed = periodic_fixed_point(period_map, start, tol=tol, rank=rank, max_cycles=max_cycles, atol=atol)
    return fixed, advance(fixed.state)
