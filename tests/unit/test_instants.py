"""One instant proposed by two owners through different arithmetic is one landing."""

from types import SimpleNamespace

import pytest

from motronics.gauss.time_integrators._landing import live_targets, next_deadline
from motronics.gauss.time_integrators.instants import canonical, is_future, same_instant
from motronics.laplace.circuit.core.transient.clock_schedule import sampled_schedule

DT = 1 / 86400
CARRIER = 1 / 2400


def test_the_lattice_and_carrier_spell_one_instant_two_ways():
    assert 36 * DT != CARRIER
    assert same_instant(36 * DT, CARRIER)
    assert canonical(CARRIER, 36 * DT) == 36 * DT


def test_a_carrier_corner_on_the_lattice_is_exactly_one_landing():
    landings, now = [], [35 * DT]

    def carrier(time):
        return CARRIER if is_future(CARRIER, time) else None

    for target in live_targets([36 * DT], lambda: now[0], carrier):
        landings.append(target)
        now[0] = target
    assert landings == [36 * DT]


def test_a_strictly_interior_corner_still_lands_where_it_is():
    edge = 35.4 * DT
    assert next_deadline(lambda _time: edge, 35 * DT, 36 * DT) == edge


def test_an_owner_proposing_the_current_instant_again_is_refused():
    with pytest.raises(ValueError, match='strictly future'):
        next_deadline(lambda _time: CARRIER, 36 * DT, 37 * DT)


def test_sampled_schedule_holds_one_instant_where_a_tick_meets_the_lattice():
    controller = SimpleNamespace(sample_periods=(CARRIER,))
    study = SimpleNamespace(t0=0.0, t1=72 * DT, dt=DT)
    _origin, times = sampled_schedule([controller], study, None)
    assert len(times) == 73
    assert 36 * DT in times
