"""FORMATS-UPGRADE-ON-READ for the sealed transient checkpoint: an older envelope upgrades, an unknown one refuses."""

from __future__ import annotations

import json

import pytest

from numerics_lab.time_integrators import checkpoint as module
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint, read_checkpoint_file, write_checkpoint_file


def _written(tmp_path):
    return write_checkpoint_file(tmp_path / 'c.json', TransientCheckpoint('k', 'bdf1', 0.5, 0.25, ((0.5, 1.0),)))


def test_an_older_envelope_reads_through_its_declared_step(tmp_path, monkeypatch) -> None:
    path = _written(tmp_path)
    steps = []
    monkeypatch.setattr(module, 'CHECKPOINT_FILE_SCHEMA_VERSION', 2)
    monkeypatch.setattr(module, '_UPGRADES', {1: lambda mapping: steps.append(1) or mapping})
    assert read_checkpoint_file(path).time == 0.5
    assert steps == [1]


@pytest.mark.parametrize(
    'schema', ['numerics-lab-transient-checkpoint/v2', 'numerics-lab-transient-checkpoint/vx', 'x/v1']
)
def test_an_unknown_envelope_is_refused(tmp_path, schema) -> None:
    path = _written(tmp_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    path.write_text(json.dumps({**data, 'schema': schema}), encoding='utf-8')
    with pytest.raises(ValueError, match='checkpoint'):
        read_checkpoint_file(path)
