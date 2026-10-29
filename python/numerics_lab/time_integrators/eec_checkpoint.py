"""Explicit safe restart state for the native coarse TP-EEC cycle algorithms."""

import math
from dataclasses import dataclass

from numerics_lab.time_integrators.eec_tangent import CycleTangentCSR

__all__ = ['EECCheckpointState']


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        msg = 'EEC checkpoint values must be finite JSON numbers'
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class EECCheckpointState:
    """Cycle clock and coarse boundary moment, separate from physical BDF history."""

    origin_time: float
    next_correction_time: float
    cycle_start_moment: tuple[float, ...]
    period: float
    sign: int
    bootstrap_pending: bool = False
    cycle_tangent: CycleTangentCSR | None = None
    cycle_converged: bool = True

    def __post_init__(self) -> None:
        for value in (self.origin_time, self.next_correction_time, self.period):
            _finite(value)
        if self.period <= 0 or self.next_correction_time <= self.origin_time:
            msg = 'EEC checkpoint requires a positive period and advancing cycle clock'
            raise ValueError(msg)
        if type(self.sign) is not int or self.sign not in (-1, 1) or type(self.bootstrap_pending) is not bool:
            msg = 'EEC checkpoint sign and bootstrap state must have explicit declared types'
            raise ValueError(msg)
        if not isinstance(self.cycle_start_moment, tuple) or not self.cycle_start_moment:
            msg = 'EEC checkpoint requires a nonempty finite cycle-start moment'
            raise ValueError(msg)
        for value in self.cycle_start_moment:
            _finite(value)
        if (
            type(self.cycle_converged) is not bool
            or (self.cycle_tangent is None and not self.cycle_converged)
            or (
                self.cycle_tangent is not None
                and (
                    type(self.cycle_tangent) is not CycleTangentCSR
                    or self.cycle_tangent.shape != (len(self.cycle_start_moment),) * 2
                )
            )
        ):
            msg = 'EEC cycle tangent dimension and convergence state must match its cycle moment'
            raise ValueError(msg)
        phase = (self.next_correction_time - self.origin_time) / self.period
        if not math.isfinite(phase) or phase < 1.0 or phase > 2**52 or abs(phase - round(phase)) > 64 * math.ulp(phase):
            msg = 'EEC checkpoint next boundary must lie on its original cycle clock'
            raise ValueError(msg)

    def validate_checkpoint(self, time, integrator, states):
        tolerance = 64 * max(math.ulp(time), math.ulp(self.next_correction_time), math.ulp(self.period))
        if (
            integrator not in ('bdf1', 'bdf2')
            or len(self.cycle_start_moment) != len(states[-1])
            or self.origin_time > time
            or self.next_correction_time <= time
            or self.next_correction_time - time > self.period + tolerance
            or self.bootstrap_pending != (integrator == 'bdf2' and len(states) == 1)
            or (self.bootstrap_pending and time != self.origin_time)
        ):
            msg = 'EEC checkpoint cycle moment, bootstrap or clock does not match numerical history'
            raise ValueError(msg)
        if self.cycle_tangent is not None and (integrator != 'bdf1' or self.sign != 1):
            msg = 'nonlinear EEC tangent checkpoint requires full-period BDF1'
            raise ValueError(msg)

    def validate_controls(self, period, sign, nonlinear=False):
        if nonlinear != (self.cycle_tangent is not None):
            msg = 'EEC restart nonlinear correction model must match its saved mode'
            raise ValueError(msg)
        if period != self.period or sign != self.sign:
            msg = 'EEC restart period and sign must match the saved correction cycle'
            raise ValueError(msg)

    def to_mapping(self):
        return {
            'version': 1,
            'origin_time': self.origin_time,
            'next_correction_time': self.next_correction_time,
            'cycle_start_moment': list(self.cycle_start_moment),
            'period': self.period,
            'sign': self.sign,
            'bootstrap_pending': self.bootstrap_pending,
            **(
                {}
                if self.cycle_tangent is None
                else {'cycle_tangent': self.cycle_tangent.to_mapping(), 'cycle_converged': self.cycle_converged}
            ),
        }

    @classmethod
    def from_mapping(cls, data):
        expected = {
            'version',
            'origin_time',
            'next_correction_time',
            'cycle_start_moment',
            'period',
            'sign',
            'bootstrap_pending',
        }
        if (
            not isinstance(data, dict)
            or set(data) not in (expected, expected | {'cycle_tangent', 'cycle_converged'})
            or type(data['version']) is not int
            or data['version'] != 1
            or not isinstance(data['cycle_start_moment'], list)
        ):
            msg = 'EEC checkpoint requires exactly the declared version-one cycle state'
            raise ValueError(msg)
        return cls(
            data['origin_time'],
            data['next_correction_time'],
            tuple(data['cycle_start_moment']),
            data['period'],
            data['sign'],
            data['bootstrap_pending'],
            None if 'cycle_tangent' not in data else CycleTangentCSR.from_mapping(data['cycle_tangent']),
            data.get('cycle_converged', True),
        )
