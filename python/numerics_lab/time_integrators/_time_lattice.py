"""Optional absolute integer time lattice for a physics-free adaptive march."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

__all__ = ['TimeLattice', 'adaptive_lattice']


@dataclass(frozen=True)
class TimeLattice:
    quantum: float
    origin: float

    def tick(self, time):
        coordinate = (time - self.origin) / self.quantum
        if not np.isfinite(coordinate):
            msg = 'adaptive time lattice requires finite representable coordinates'
            raise ValueError(msg)
        # Beyond this range a float coordinate cannot distinguish adjacent integers.
        # A relative alignment tolerance would silently admit fractional mesh pitches.
        if abs(coordinate) >= 2**53:
            msg = 'adaptive time lattice requires representable neighboring ticks'
            raise ValueError(msg)
        if not np.isclose(coordinate, round(coordinate), rtol=0.0, atol=1e-9):
            msg = 'adaptive time lattice requires aligned finite times'
            raise ValueError(msg)
        tick = round(coordinate)
        self.time(tick)
        return tick

    def time(self, tick):
        if abs(tick) >= 2**53:
            msg = 'adaptive time lattice requires representable neighboring ticks'
            raise ValueError(msg)
        previous = self.origin + (tick - 1) * self.quantum
        current = self.origin + tick * self.quantum
        following = self.origin + (tick + 1) * self.quantum
        if not np.all(np.isfinite([previous, current, following])) or not previous < current < following:
            msg = 'adaptive time lattice quantum cannot make representable clock progress'
            raise ValueError(msg)
        return current

    def steps(self, interval):
        return max(1, int(np.floor(interval / self.quantum + 1e-9)))


def adaptive_lattice(window, adaptive, quantum, origin, previous_dt):
    if quantum is None:
        if origin is not None:
            msg = 'timestep_origin requires timestep_quantum'
            raise ValueError(msg)
        return None
    if (
        not adaptive
        or isinstance(quantum, bool)
        or not isinstance(quantum, Real)
        or not np.isfinite(quantum)
        or quantum <= 0
    ):
        msg = 'adaptive time lattice requires positive finite timestep_quantum'
        raise ValueError(msg)
    origin = window.t0 if origin is None else origin
    if isinstance(origin, bool) or not isinstance(origin, Real) or not np.isfinite(origin):
        msg = 'adaptive time lattice requires finite timestep_origin'
        raise ValueError(msg)
    lattice = TimeLattice(float(quantum), float(origin))
    for time in (window.t0, window.t1, window.t0 - previous_dt, window.t0 + window.dt):
        lattice.tick(time)
    return lattice
