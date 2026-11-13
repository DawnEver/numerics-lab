"""FORMATS-MIGRATE-ONCE for the sealed transient checkpoint (user ruling D51, 2026-11-11).

The reader keeps only the current envelope. An older one is refused BY NAME with its migration instruction
(re-run the march), never upgraded on read; an unknown or newer one is refused as unknown.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from numerics_lab.time_integrators import checkpoint as module
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint, read_checkpoint_file, write_checkpoint_file

pytestmark = pytest.mark.unit


def _tiny():
    return TransientCheckpoint('k', 'bdf1', 0.5, 0.25, ((0.5, 1.0),))


def _write(path, schema: str):
    mapping = _tiny().to_mapping()
    canonical = json.dumps(mapping, sort_keys=True, separators=(',', ':'), allow_nan=False)
    seal = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    path.write_text(json.dumps({'schema': schema, 'sha256': seal, 'checkpoint': mapping}), encoding='utf-8')
    return path


def test_the_current_envelope_round_trips(tmp_path) -> None:
    """PLANTED CONTROL: the refusals below are about the envelope, not about reading at all."""
    path = write_checkpoint_file(tmp_path / 'c.json', _tiny())
    assert json.loads(path.read_text(encoding='utf-8'))['schema'] == module.CHECKPOINT_FILE_SCHEMA
    assert read_checkpoint_file(path).time == 0.5
    assert read_checkpoint_file(_write(tmp_path / 'd.json', module.CHECKPOINT_FILE_SCHEMA)).time == 0.5


@pytest.mark.parametrize('schema', ['motronics-transient-checkpoint/v1', 'numerics-lab-transient-checkpoint/v1'])
def test_an_older_envelope_is_refused_by_name_with_its_migration(tmp_path, schema) -> None:
    with pytest.raises(ValueError, match='older envelope') as refusal:
        read_checkpoint_file(_write(tmp_path / 'c.json', schema))
    assert repr(schema) in str(refusal.value)
    assert 're-run the march from its start' in str(refusal.value)


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
    with pytest.raises(ValueError, match='unknown to this reader') as refusal:
        read_checkpoint_file(_write(tmp_path / 'c.json', schema))
    assert repr(schema) in str(refusal.value)


def test_an_edited_file_is_refused_before_its_envelope_is_read(tmp_path) -> None:
    path = _write(tmp_path / 'c.json', 'motronics-transient-checkpoint/v1')
    data = json.loads(path.read_text(encoding='utf-8'))
    data['checkpoint']['time'] = 0.75
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='sha256 seal'):
        read_checkpoint_file(path)
