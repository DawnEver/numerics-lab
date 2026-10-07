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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ['PeriodicFixedPoint', 'periodic_fixed_point']


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
    x = np.asarray(x0, dtype=float).copy()
    calls = 0
    residual = np.inf
    for _ in range(max_cycles):
        xs = [x]
        for _ in range(rank + 1):
            xs.append(np.asarray(period_map(xs[-1]), dtype=float))
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
