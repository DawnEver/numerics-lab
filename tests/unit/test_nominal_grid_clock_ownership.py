"""Numerical targets never replace declared clocks or the checkpoint's anchored grid."""

import json
from itertools import pairwise

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint
from numerics_lab.time_integrators.instants import is_future

_DT = 1e-6
_CORNERS = [k / 144_000.0 for k in range(128, 146)]


class _Ramp:
    def residual(self, state, rate, acceleration, time):
        return rate - 1.0

    def jacobian(self, state, rate, acceleration, time, weights):
        return np.eye(1) * weights.alpha1.real


def _run(start, end, deadlines, restart=None):
    return integrate_transient(
        _Ramp(),
        TimeWindow(start, end, _DT, IntegratorKind.BDF2),
        np.zeros(1),
        next_breakpoint=lambda time: next((value for value in deadlines if is_future(value, time)), None),
        checkpoint_key='owned-clock',
        restart=restart,
    )


def test_reanchored_targets_are_products_from_one_anchor_not_chained_sums():
    solution = _run(0.000889, 0.00101, sorted([*_CORNERS, 1006 * _DT]))
    anchor, index = solution.times[0], 0
    for previous, time in pairwise(solution.times):
        if abs((time - previous) - _DT) > 1e-6 * _DT:
            anchor, index = time, 0
        else:
            index += 1
            assert abs(time - (anchor + index * _DT)) <= abs(np.spacing(time))


def test_each_owner_tick_keeps_its_exact_label_without_roundoff_slivers():
    ticks = np.arange(1011) * _DT
    solution = _run(0.000896, 0.00101, sorted([*ticks, *_CORNERS]))
    assert all(tick in solution.times for tick in ticks[897:])
    assert min(np.diff(solution.times)) > _DT / 1000.0


@pytest.mark.parametrize('direction', [-np.inf, np.inf])
def test_the_explicit_window_end_owns_a_coincident_owner_landing(direction):
    tick = 1e-3
    end = float(np.nextafter(tick, direction))
    solution = integrate_transient(
        _Ramp(),
        TimeWindow(0.0, end, tick, IntegratorKind.BDF1),
        np.zeros(1),
        next_breakpoint=lambda time: tick if is_future(tick, time) else None,
    )
    assert solution.times[-1] == end


@pytest.mark.parametrize('split_index', [1, 7, 19])
def test_json_restart_retains_the_actual_reanchored_clock(split_index):
    deadlines = sorted([*_CORNERS, 1006 * _DT])
    whole = _run(0.000889, 0.00101, deadlines)
    split = whole.times[split_index]
    first = _run(0.000889, split, deadlines)
    saved = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping())))
    resumed = _run(split, 0.00101, deadlines, saved)
    assert resumed.times == whole.times[split_index:]
    np.testing.assert_array_equal(resumed.states, whole.states[split_index:])
    assert resumed.checkpoint.to_mapping() == whole.checkpoint.to_mapping()
