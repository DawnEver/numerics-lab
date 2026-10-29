"""FORMATS-UPGRADE-ON-READ for the sealed transient checkpoint (user ruling 2026-10-18).

A checkpoint is the live state a running march resumes from, so a version change must not strand it:
every older envelope upgrades through ``_UPGRADES`` and resumes; only an unknown or newer one is refused.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import checkpoint as module
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint, read_checkpoint_file, write_checkpoint_file

pytestmark = pytest.mark.unit


class _Cubic:
    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + u + u**3 - np.cos(5.0 * t)

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.diag(weights.alpha0.real + weights.alpha1.real * (1.0 + 3.0 * u**2))


def _march(t0, t1, **kw):
    study = TimeWindow(t0, t1, 0.1, IntegratorKind.BDF2)
    return integrate_transient(_Cubic(), study, np.zeros(1), tol=1e-4, max_newton=30, checkpoint_key='cubic', **kw)


def _tiny():
    return TransientCheckpoint('k', 'bdf1', 0.5, 0.25, ((0.5, 1.0),))


def _v1_mapping(checkpoint: TransientCheckpoint, next_reanchor=None) -> dict:
    """The v1 mapping: its event locator carried the chained ``next_reanchor``, not v2's ``nominal_anchor``."""
    mapping = checkpoint.to_mapping()
    if 'event_state' in mapping:
        event = dict(mapping['event_state'])
        del event['nominal_anchor']
        mapping['event_state'] = {**event, 'next_reanchor': next_reanchor}
    return mapping


def _write_v1(path, checkpoint: TransientCheckpoint, schema: str, mapping: dict | None = None):
    """The v1 writer's shape exactly (the predecessor package before the engine split, and numerics-lab until v2)."""
    mapping = _v1_mapping(checkpoint) if mapping is None else mapping
    canonical = json.dumps(mapping, sort_keys=True, separators=(',', ':'), allow_nan=False)
    seal = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    payload = json.dumps({'schema': schema, 'sha256': seal, 'checkpoint': mapping}, allow_nan=False)
    path.write_text(payload, encoding='utf-8')
    return path


@pytest.mark.parametrize('schema', ['motronics-transient-checkpoint/v1', 'numerics-lab-transient-checkpoint/v1'])
def test_a_v1_checkpoint_resumes_bit_identical_to_the_straight_march(tmp_path, schema) -> None:
    whole = _march(0.0, 1.0)
    first = _march(0.0, 0.5)
    saved = read_checkpoint_file(_write_v1(tmp_path / 'c.json', first.checkpoint, schema))
    resumed = _march(saved.time, 1.0, restart=saved)
    np.testing.assert_array_equal(np.asarray(resumed.states), np.asarray(whole.states[5:]))


def test_the_current_envelope_round_trips(tmp_path) -> None:
    """Planted control: the refusals below are about the envelope, not about reading at all."""
    path = write_checkpoint_file(tmp_path / 'c.json', _tiny())
    assert json.loads(path.read_text(encoding='utf-8'))['schema'] == module.CHECKPOINT_FILE_SCHEMA
    assert read_checkpoint_file(path).time == 0.5


@pytest.mark.parametrize(
    'schema',
    [
        'numerics-lab-transient-checkpoint/v0',
        f'numerics-lab-transient-checkpoint/v{module.CHECKPOINT_FILE_SCHEMA_VERSION + 1}',
        'numerics-lab-transient-checkpoint/vx',
        'predecessor-transient-checkpoint/v2',
        'x/v1',
    ],
)
def test_an_unknown_or_newer_envelope_is_refused_by_name(tmp_path, schema) -> None:
    path = _write_v1(tmp_path / 'c.json', _tiny(), schema)
    with pytest.raises(ValueError, match='unknown to this reader') as refusal:
        read_checkpoint_file(path)
    assert repr(schema) in str(refusal.value)


def test_an_older_envelope_is_sealed_before_it_is_upgraded(tmp_path) -> None:
    path = _write_v1(tmp_path / 'c.json', _tiny(), 'motronics-transient-checkpoint/v1')
    data = json.loads(path.read_text(encoding='utf-8'))
    data['checkpoint']['time'] = 0.75
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='sha256 seal'):
        read_checkpoint_file(path)


def test_a_v1_event_state_is_anchored_where_its_chained_grid_started(tmp_path) -> None:
    """PLANTED: the v1 file really carries ``next_reanchor`` -- the v2 reader without its upgrade refuses it."""
    first = _march(0.0, 0.5)
    assert first.checkpoint.event_state is not None
    path = _write_v1(tmp_path / 'c.json', first.checkpoint, 'numerics-lab-transient-checkpoint/v1')
    assert 'next_reanchor' in json.loads(path.read_text(encoding='utf-8'))['checkpoint']['event_state']
    saved = read_checkpoint_file(path)
    k = saved.event_state.nominal_index
    assert saved.event_state.nominal_anchor == saved.time - k * saved.event_state.step_dt


def test_a_v1_re_anchor_off_its_pending_target_is_refused_by_name(tmp_path) -> None:
    first = _march(0.0, 0.5)
    mapping = _v1_mapping(first.checkpoint, next_reanchor=0.55)
    path = _write_v1(tmp_path / 'c.json', first.checkpoint, 'numerics-lab-transient-checkpoint/v1', mapping)
    with pytest.raises(ValueError, match='no v2 nominal grid'):
        read_checkpoint_file(path)
