"""The multirate scheduler: zero-order holds, rate transitions and the auto sample time.

The planted control for the ZOH is the same RC driven by the UNSAMPLED sine: if the hold were not
applied (the task's value leaking continuously into the plant), the two would agree; they differ by
far more than the ZOH answer's own error against its exact piecewise solution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from motronics.contracts.spec.study_spec import IntegratorKind, TransientStudy
from motronics.gauss.time_integrators import (
    Hold,
    RateTransition,
    SampledTask,
    StepTolerance,
    base_sample_time,
    integrate_sampled,
    integrate_transient,
    sample_hits,
)

_TAU, _W, _TS = 0.2, 3.0, 0.05


class _RC:
    """``tau u' + u = drive(t)``."""

    def __init__(self, drive) -> None:
        self.drive = drive

    def residual(self, u, u_dot, u_ddot, t):
        return _TAU * u_dot + u - self.drive(t)

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.atleast_2d(weights.alpha0.real + _TAU * weights.alpha1.real)


def _zoh_exact(t_end: float) -> float:
    u, t = 0.0, 0.0
    while t < t_end - 1e-12:
        h = math.sin(_W * t)
        step = min(_TS, t_end - t)
        u = h + (u - h) * math.exp(-step / _TAU)
        t += step
    return u


def test_the_plant_sees_the_held_value_and_matches_the_exact_zoh_answer() -> None:
    hold = Hold(0.0)

    def sample(t, _u):
        hold.value = math.sin(_W * t)

    study = TransientStudy(t0=0.0, t1=1.0, dt=_TS, integrator=IntegratorKind.TR_BDF2)
    tol = StepTolerance(rtol=1e-9, atol=1e-11)
    sol = integrate_sampled(
        _RC(lambda _t: np.array([hold.value])), study, np.zeros(1), [SampledTask(_TS, sample)], step_control=tol
    )
    zoh_err = abs(float(sol.states[-1][0]) - _zoh_exact(1.0))
    continuous = integrate_transient(_RC(lambda t: np.array([math.sin(_W * t)])), study, np.zeros(1), step_control=tol)
    leak = abs(float(continuous.states[-1][0]) - _zoh_exact(1.0))
    assert zoh_err < 1e-6
    assert leak > 1e3 * zoh_err  # planted control: the unheld drive is a different answer


def test_an_integer_ratio_rate_transition_reads_the_fast_value_of_its_own_instant() -> None:
    fast_out, seen = Hold(0.0), []
    rt = RateTransition(fast_out, period=3.0)

    def fast(t, _u):
        fast_out.value = t

    def slow(t, _u):
        seen.append((t, rt.target.value))

    tasks = [rt.task(), SampledTask(3.0, slow), SampledTask(1.0, fast)]
    integrate_sampled(_RC(lambda _t: np.zeros(1)), TransientStudy(t0=0.0, t1=9.0, dt=1.0), np.zeros(1), tasks)
    # fastest clock first; same-clock tasks in declaration order (the latch, then its reader)
    assert seen == [(0.0, 0.0), (3.0, 3.0), (6.0, 6.0)]


def test_a_non_integer_ratio_shares_one_grid() -> None:
    fired: dict[float, list[float]] = {2.0: [], 3.0: []}
    tasks = [SampledTask(p, lambda t, _u, p=p: fired[p].append(t)) for p in (2.0, 3.0)]
    sol = integrate_sampled(_RC(lambda _t: np.zeros(1)), TransientStudy(t0=0.0, t1=7.0, dt=1.0), np.zeros(1), tasks)
    assert fired[2.0] == [0.0, 2.0, 4.0, 6.0]
    assert fired[3.0] == [0.0, 3.0, 6.0]
    assert {0.0, 2.0, 3.0, 4.0, 6.0, 7.0} <= set(np.round(sol.times, 12))


@pytest.mark.parametrize(
    ('periods', 'base'), [((1.0, 3.0), 1.0), ((2e-4, 3e-4), 1e-4), ((1e-4, 1.5e-4), 5e-5), ((0.05,), 0.05)]
)
def test_the_auto_sample_time_is_the_rational_gcd(periods, base) -> None:
    assert base_sample_time(periods) == pytest.approx(base, rel=1e-12)


def test_the_hit_grid_reads_a_near_integer_count_as_an_integer() -> None:
    hits = sample_hits(0.0, 8e-3, 2e-6)
    assert len(hits) == 4000
    assert hits[-1] < 8e-3
