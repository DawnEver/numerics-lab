"""EEC cycle moments and numerical BDF history restart together, including mid-cycle splits."""

import json
from copy import deepcopy

import numpy as np
import pytest
from tests.unit.test_antiperiodic_eec import Affine

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.checkpoint import EECCheckpointState, TransientCheckpoint


def _run(t0, t1, *, integrator=IntegratorKind.BDF2, sign=1, restart=None, callback=None, period=None):
    return integrate_transient(
        Affine(),
        TimeWindow(t0, t1, 0.05, integrator),
        np.array([1.0]),
        eec_period=(1.0 if sign == 1 else 0.5) if period is None else period,
        eec_sign=sign,
        eec_affine_first_order=True,
        checkpoint_key='eec-native',
        restart=restart,
        on_checkpoint=callback,
        max_newton=20,
    )


@pytest.mark.parametrize('integrator', [IntegratorKind.BDF1, IntegratorKind.BDF2])
@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('split', [0.05, 0.35, 1.0])
def test_eec_midcycle_bootstrap_and_boundary_json_restarts_match_uninterrupted(integrator, sign, split):
    whole = _run(0.0, 2.1, integrator=integrator, sign=sign)
    first = _run(0.0, split, integrator=integrator, sign=sign)
    saved = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping(), allow_nan=False)))
    resumed = _run(saved.time, 2.1, integrator=integrator, sign=sign, restart=saved)
    index = round(split / 0.05)
    # The boundary checkpoint contains corrected history; the physical endpoint
    # reported in the first trajectory intentionally precedes that coarse jump.
    np.testing.assert_allclose(resumed.states[1:], whole.states[index + 1 :], rtol=1e-11, atol=1e-13)
    assert resumed.checkpoint == whole.checkpoint
    assert saved.eec_state is not None
    assert saved.eec_state.origin_time == 0.0


def test_streamed_eec_boundary_checkpoint_is_canonical_postcorrection_state():
    records = []
    whole = _run(0.0, 1.1, callback=records.append)
    boundary = records[19]
    assert boundary.time == 1.0
    assert boundary.eec_state.next_correction_time == 2.0
    assert boundary.eec_state.period == 1.0
    resumed = _run(boundary.time, 1.1, restart=boundary)
    assert resumed.checkpoint == whole.checkpoint


@pytest.mark.parametrize('integrator', [IntegratorKind.BDF1, IntegratorKind.BDF2])
@pytest.mark.parametrize('stop_time', [1.0, 0.7])
def test_steady_stop_streams_canonical_eec_checkpoint_and_resumes_future_states(integrator, stop_time):
    records = []
    stopped = integrate_transient(
        Affine(),
        TimeWindow(0.0, 2.1, 0.05, integrator),
        np.ones(1),
        eec_period=1.0,
        eec_sign=1,
        eec_affine_first_order=True,
        checkpoint_key='eec-native',
        on_checkpoint=records.append,
        max_newton=20,
        steady_period=stop_time / 2,
        steady_atol=1e6,
        steady_rtol=0.0,
    )
    # Loose declared tolerances intentionally select the first eligible pair of
    # cycles, exercising both a correction boundary and an ordinary endpoint.
    assert stopped.times[-1] == pytest.approx(stop_time)
    assert records[-1] == stopped.checkpoint
    assert len(records) == len(stopped.times) - 1
    assert stopped.checkpoint.eec_state.next_correction_time > stopped.checkpoint.time
    if stop_time == 1.0:
        assert stopped.corrected_times == [1.0]
        assert stopped.checkpoint.eec_state.next_correction_time == 2.0
    else:
        assert stopped.corrected_times == []
        assert stopped.checkpoint.eec_state.next_correction_time == 1.0
    saved = TransientCheckpoint.from_mapping(json.loads(json.dumps(stopped.checkpoint.to_mapping(), allow_nan=False)))
    resumed = _run(saved.time, 2.1, integrator=integrator, restart=saved)
    whole = _run(0.0, 2.1, integrator=integrator)
    index = round(stop_time / 0.05)
    np.testing.assert_allclose(resumed.states[1:], whole.states[index + 1 :], rtol=1e-11, atol=1e-13)
    assert resumed.checkpoint == whole.checkpoint


@pytest.mark.parametrize(
    'change',
    [
        {'period': float('nan')},
        {'sign': True},
        {'origin_time': 2.0},
        {'next_correction_time': 0.1},
        {'cycle_start_moment': [1.0, 2.0]},
        {'unknown': 1.0},
        {'period': True},
        {'cycle_start_moment': [float('inf')]},
        {'version': True},
        {'version': 2},
        {'bootstrap_pending': True},
    ],
)
def test_malformed_eec_cycle_payload_is_refused_before_restore(change):
    saved = _run(0.0, 0.35).checkpoint
    mapping = deepcopy(saved.to_mapping())
    mapping['eec_state'].update(change)
    with pytest.raises(ValueError, match='EEC|eec|checkpoint|cycle'):
        malformed = TransientCheckpoint.from_mapping(mapping)
        _run(malformed.time, 1.1, restart=malformed)


def test_eec_restart_cannot_change_correction_controls_or_drop_algorithm_state():
    saved = _run(0.0, 0.35).checkpoint
    with pytest.raises(ValueError, match='EEC|eec|period|cycle'):
        _run(saved.time, 1.1, restart=saved, period=0.5)
    with pytest.raises(ValueError, match='EEC|eec|sign|cycle'):
        _run(saved.time, 1.1, restart=saved, sign=-1)
    with pytest.raises(ValueError, match='EEC|eec|cycle'):
        integrate_transient(
            Affine(), TimeWindow(saved.time, 1.1, 0.05), np.ones(1), restart=saved, checkpoint_key='eec-native'
        )
    mapping = saved.to_mapping()
    del mapping['eec_state']
    missing = TransientCheckpoint.from_mapping(mapping)
    with pytest.raises(ValueError, match='EEC|eec|cycle'):
        _run(missing.time, 1.1, restart=missing)


def test_zero_step_bdf2_checkpoint_resumes_the_original_be_bootstrap_moment():
    initial = TransientCheckpoint(
        'eec-native',
        'bdf2',
        0.0,
        0.05,
        ((1.0,),),
        eec_state=EECCheckpointState(0.0, 1.0, (1.0,), 1.0, 1, bootstrap_pending=True),
    )
    saved = TransientCheckpoint.from_mapping(json.loads(json.dumps(initial.to_mapping(), allow_nan=False)))
    whole = _run(0.0, 2.1)
    resumed = _run(0.0, 2.1, restart=saved)
    np.testing.assert_allclose(resumed.states, whole.states, rtol=1e-11, atol=1e-13)
    assert resumed.checkpoint == whole.checkpoint


def test_zero_step_bdf2_checkpoint_cannot_claim_bootstrap_already_completed():
    with pytest.raises(ValueError, match='bootstrap|EEC|checkpoint'):
        TransientCheckpoint(
            'eec-native',
            'bdf2',
            0.0,
            0.05,
            ((1.0,),),
            eec_state=EECCheckpointState(0.0, 1.0, (1.0,), 1.0, 1, bootstrap_pending=False),
        )


def test_existing_stateless_checkpoint_mapping_remains_exactly_version_three_without_eec_slot():
    checkpoint = TransientCheckpoint('plain', 'bdf1', 0.1, 0.1, ((1.0,),))
    mapping = checkpoint.to_mapping()
    assert mapping['schema_version'] == 3
    assert 'eec_state' not in mapping
    assert TransientCheckpoint.from_mapping(mapping) == checkpoint
