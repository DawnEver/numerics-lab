"""Reject unrepresentable clocks before entering an adaptive marching loop."""

from types import SimpleNamespace

import pytest

from numerics_lab.time_integrators._time_lattice import adaptive_lattice


@pytest.mark.parametrize(
    ('start', 'end', 'dt', 'quantum', 'origin'),
    [
        (1e16, 1e16 + 4.0, 0.1, 0.1, 1e16),
        (0.0, float(2**53), 1.0, 1.0, 0.0),
        (0.0, float(2**53) + 2.0, 2.0, 1.0, 0.0),
    ],
)
def test_admission_rejects_nonrepresentable_neighbor_ticks(start, end, dt, quantum, origin):
    study = SimpleNamespace(t0=start, t1=end, dt=dt)
    # Deliberately call admission only: the faulty implementation must not hang a march.
    with pytest.raises(ValueError, match='representable|progress'):
        adaptive_lattice(study, True, quantum, origin, dt)


def test_admission_rejects_an_overflowed_absolute_tick_coordinate():
    study = SimpleNamespace(t0=1e308, t1=1.1e308, dt=1e307)
    with pytest.raises(ValueError, match='finite|representable'):
        adaptive_lattice(study, True, 1e-308, -1e308, study.dt)


def test_admission_does_not_round_a_fractional_mesh_pitch_into_alignment():
    study = SimpleNamespace(t0=0.37, t1=0.37 + 0.15, dt=0.1)
    with pytest.raises(ValueError, match='aligned'):
        adaptive_lattice(study, True, 0.1, 0.37, study.dt)
