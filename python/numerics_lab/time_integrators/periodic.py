"""Periodic steady state as a fixed point of the one-period map -- physics-blind.

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
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from motronics.gauss.time_integrators._forms import TransientSolution

__all__ = ['PeriodicFixedPoint', 'periodic_fixed_point', 'periodic_steady_state', 'smooth_phase']


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
) -> PeriodicFixedPoint:
    """Find ``x`` with ``|P(x) - x| <= tol * max(|x|, |P(x)|)`` by cycled RRE.

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
            residual = float(np.max(np.abs(xs[-1] - xs[-2]))) / scale
            if residual <= tol:
                return PeriodicFixedPoint(state=xs[-2], residual=residual, map_calls=calls)
        x = _rre(xs)
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
) -> tuple[PeriodicFixedPoint, TransientSolution]:
    """THE forced periodic steady state: the period map's fixed point and one marched period of it.

    ``march(t0, t1, x, breakpoints, first)`` marches one window (``first`` is True only for the very
    first map, so a caller may reset its own switch state once). The map starts at
    :func:`smooth_phase` -- never on a forcing edge -- and lands every edge of the period as a
    breakpoint. ``x0`` is the start state, or a callable returning it at the chosen phase. ``period``
    must be the forcing period of every source; a free-running orbit has no such clock.
    """
    phase = smooth_phase(period, edges)
    wrapped = sorted({e % period + (period if e % period <= phase else 0.0) for e in edges})
    bps = tuple(phase + period if abs(e - phase - period) < 1e-12 * period else e for e in wrapped)
    first = [True]

    def period_map(x: NDArray[np.floating]) -> NDArray[np.floating]:
        sol = march(phase, phase + period, x, bps, first[0])
        first[0] = False
        return np.asarray(sol.states[-1], dtype=float)

    start = x0(phase) if callable(x0) else np.asarray(x0, dtype=float)
    fixed = periodic_fixed_point(period_map, start, tol=tol, rank=rank, max_cycles=max_cycles)
    return fixed, march(phase, phase + period, fixed.state, bps, False)
