"""Output compression preserves the exact evaluation clock of retained rates."""

import numpy as np
import pytest

from numerics_lab.time_integrators._forms import TransientSolution
from numerics_lab.time_integrators.decimate import decimate


def test_decimation_reindexes_residual_times_with_retained_rates():
    edge_left = float(np.nextafter(3.0, -np.inf))
    solution = TransientSolution(
        times=[0.0, 1.0, 2.0, 3.0],
        states=[np.array([t]) for t in range(4)],
        state_rates=[np.array([1.0]) for _ in range(3)],
        residual_times=[1.0, 2.0, edge_left],
    )
    compressed = decimate(solution, 0.0)
    assert compressed.times == [0.0, 3.0]
    assert compressed.residual_times == [edge_left]
    assert len(compressed.residual_times) == len(compressed.state_rates)


@pytest.mark.parametrize('evaluation_times', [[], [1.0], [1.0, 2.0, 3.0]])
def test_decimation_refuses_misaligned_residual_time_records(evaluation_times):
    solution = TransientSolution(
        times=[0.0, 1.0, 2.0],
        states=[np.array([t]) for t in range(3)],
        residual_times=evaluation_times,
    )
    with pytest.raises(ValueError, match='residual.*time'):
        decimate(solution, 0.0)


def test_decimation_does_not_invent_unrecorded_evaluation_times():
    solution = TransientSolution(
        times=[0.0, 1.0, 2.0],
        states=[np.array([t]) for t in range(3)],
    )
    assert decimate(solution, 0.0).residual_times is None
