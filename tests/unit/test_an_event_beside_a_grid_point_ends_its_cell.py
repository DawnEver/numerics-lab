"""An event located just short of a NOMINAL grid point ends its cell; it leaves no sliver step.

A sliver remainder ``r`` after an event becomes the previous step of the next multistep step, whose
ratio ``dt / r`` is then unbounded: variable-step BDF2 weighs its history by ``dt / (r (dt + r))``,
so a rounding floor in the state is divided by ``r`` (measured in motronics' distributed chopper:
1.6 us before a 1 ms grid point, ratio 166, terminal balance 8.8e-9). Variable-step BDF2 is
zero-stable for step ratios up to ``1 + sqrt(2)`` (Grigorieff), so a remainder shorter than
``dt / (1 + sqrt(2))`` -- seconds, the unit of ``dt`` -- re-anchors the nominal grid at the event,
exactly as a breakpoint does. A longer remainder is a legitimate step and is kept.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.unit._helpers import _RampWithToggle

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient

DT = 0.3
_RATIO = 1.0 + np.sqrt(2.0)


def _times(level: float, **options) -> np.ndarray:
    problem = _RampWithToggle(level=level)
    study = TimeWindow(t0=0.0, t1=2.1, dt=DT, integrator=IntegratorKind.BDF2)
    solution = integrate_transient(
        problem,
        study,
        np.array([0.0]),
        event_fns=(problem.event_fn,),
        on_event=lambda _i, _t, _u: problem.toggle(),
        **options,
    )
    assert problem.toggles == 1
    return np.asarray(solution.times, dtype=float)


def test_a_sliver_remainder_re_anchors_the_grid_at_the_event() -> None:
    times = _times(0.89)  # remainder 0.01 s to the 0.9 s grid point, below DT / (1 + sqrt 2)
    after = times[times > 0.89 + 1e-9]
    assert after[0] == pytest.approx(0.89 + DT, abs=1e-9), f'steps after the event: {after[:3]}'
    assert np.all(np.diff(times[times >= 0.89 - 1e-9])[:-1] >= DT / _RATIO)


def test_a_remainder_the_ratio_admits_is_still_stepped() -> None:
    # Planted control: remainder 0.2 s exceeds DT / (1 + sqrt 2) = 0.124 s, so 0.9 s stays a grid point.
    times = _times(0.7)
    assert np.any(np.isclose(times, 0.9, rtol=0.0, atol=1e-12))


def test_a_breakpoint_is_never_absorbed_by_a_nearby_event() -> None:
    # A declared breakpoint is a physical instant: it is landed however close the event before it.
    times = _times(0.89, time_breakpoints=(0.9,))
    assert np.any(times == 0.9)
