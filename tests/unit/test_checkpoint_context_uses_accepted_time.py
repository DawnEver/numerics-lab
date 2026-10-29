"""The numerical checkpoint owns its accepted time, not a physical metadata field."""

import pytest

from numerics_lab.time_integrators.checkpoint import TransientCheckpoint, restore_history_checkpoint


class SelectedTimeContext:
    def __init__(self) -> None:
        self.restored = None
        self.received_times = []

    def checkpoint_context(self, time):
        return {'origin': 0.0, 'time': time}

    def restore_checkpoint_context(self, data):
        pytest.fail('selected-time authority must not be replaced by its constructor-clock contract')

    def restore_checkpoint_context_at(self, data, time):
        self.received_times.append(time)
        if data != self.checkpoint_context(time):
            msg = 'physical context differs from the numerical accepted time'
            raise ValueError(msg)
        self.restored = data.copy()


def _saved(context):
    return TransientCheckpoint('physical', 'bdf1', 0.5, 0.1, ((1.0,),), problem_context=context)


def test_selected_context_restore_receives_the_actual_checkpoint_accepted_time():
    owner = SelectedTimeContext()
    record = _saved({'origin': 0.0, 'time': 0.5})
    restore_history_checkpoint(owner, record)
    assert owner.received_times == [record.time]
    assert owner.restored == record.problem_context


def test_metadata_time_cannot_select_its_own_validation_clock():
    owner = SelectedTimeContext()
    record = _saved({'origin': 0.0, 'time': 0.75})
    with pytest.raises(ValueError, match='numerical accepted time'):
        restore_history_checkpoint(owner, record)
    assert owner.received_times == [0.5]
    assert owner.restored is None


def test_owner_without_selected_time_authority_keeps_its_actual_context_contract():
    class Context:
        def checkpoint_context(self, time):
            return {'origin': 0.0}

        def restore_checkpoint_context(self, data):
            self.restored = data.copy()

    owner = Context()
    record = _saved({'origin': 0.0})
    restore_history_checkpoint(owner, record)
    assert owner.restored == {'origin': 0.0}
