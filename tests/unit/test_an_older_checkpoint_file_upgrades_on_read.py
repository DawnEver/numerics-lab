"""A sealed transient checkpoint reads only in this reader's one envelope; any other is refused by name, never upgraded."""

from __future__ import annotations

import json

import pytest
from tests.architecture.test_retired_names_stay_retired import RETIRED
from tests.bound_names import binds

from numerics_lab.time_integrators import checkpoint as module
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint, read_checkpoint_file, write_checkpoint_file


def _written(tmp_path):
    return write_checkpoint_file(tmp_path / 'c.json', TransientCheckpoint('k', 'bdf1', 0.5, 0.25, ((0.5, 1.0),)))


def _with_schema(path, schema):
    data = json.loads(path.read_text(encoding='utf-8'))
    path.write_text(json.dumps({**data, 'schema': schema}), encoding='utf-8')
    return path


def test_the_current_envelope_reads(tmp_path) -> None:
    """Planted control: the refusals below are about the envelope, not about reading at all."""
    assert read_checkpoint_file(_written(tmp_path)).time == 0.5


@pytest.mark.parametrize(
    'schema',
    [
        'numerics-lab-transient-checkpoint/v0',
        'numerics-lab-transient-checkpoint/v2',
        'numerics-lab-transient-checkpoint/vx',
        'motronics-transient-checkpoint/v1',
        'x/v1',
    ],
)
def test_any_other_envelope_is_refused_naming_both_schemas(tmp_path, schema) -> None:
    path = _with_schema(_written(tmp_path), schema)
    with pytest.raises(ValueError, match='never upgraded') as refusal:
        read_checkpoint_file(path)
    assert repr(schema) in str(refusal.value)
    assert module.CHECKPOINT_FILE_SCHEMA in str(refusal.value)


def test_the_reader_binds_no_retired_upgrade_name() -> None:
    retired = {name for name in RETIRED if name.isidentifier()}
    assert retired  # planted control: the registry does hold the upgrade table's names
    assert not {name for name in retired if binds('numerics_lab.time_integrators.checkpoint', name)}
