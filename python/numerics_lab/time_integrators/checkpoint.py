"""Portable numerical history for restarting a native transient, without pickle execution."""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from motronics.contracts.converted import Converted
from motronics.contracts.spec.analysis_formulation import HistoryProblem
from motronics.gauss.time_integrators.eec_checkpoint import EECCheckpointState
from motronics.gauss.time_integrators.event_checkpoint import EventCheckpointState

__all__ = [
    'CheckpointContext',
    'CheckpointHistory',
    'EECCheckpointState',
    'HistoryPresence',
    'TransientCheckpoint',
    'capture_history',
    'model_digest',
    'restore_history_checkpoint',
    'validate_context_json',
    'validate_history_json',
    'validate_history_owner',
]


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        msg = 'checkpoint numeric values must be finite JSON numbers, not booleans or strings'
        raise ValueError(msg)
    return float(value)


def model_digest(parts):
    """Hash explicit numerical model data; refuse opaque objects instead of hashing their repr."""

    def plain(value):
        if isinstance(value, np.ndarray):
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'values': plain(value.tolist())}
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if isinstance(value, complex):
            return {'real': value.real, 'imag': value.imag}
        if isinstance(value, np.generic):
            return plain(value.item())
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        msg = f'checkpoint model contains opaque {type(value).__name__}'
        raise TypeError(msg)

    payload = json.dumps(plain(parts), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


@runtime_checkable
class HistoryPresence(Protocol):
    """A generic problem may expose a history seam even when its current laws are stateless."""

    @property
    def has_history(self) -> bool: ...


@runtime_checkable
class CheckpointHistory(Protocol):
    """A physics owner explicitly exports/restores finite plain-JSON material history."""

    def checkpoint_history(self) -> dict: ...

    def restore_checkpoint_history(self, data: dict) -> None: ...


@runtime_checkable
class CheckpointContext(Protocol):
    """Explicit finite-JSON physical context independent of material memory."""

    def checkpoint_context(self, time: float) -> dict: ...

    def restore_checkpoint_context(self, data: dict) -> None: ...


@dataclass(frozen=True)
class TransientCheckpoint:
    """Integrator history plus caller-owned problem identity and last accepted step size.

    A checkpoint does not invent a mesh/material fingerprint: the producer supplies the
    authoritative problem key and the consumer must match it before restoring any state.
    Optional material history is plain JSON supplied and restored by the physics
    owner's explicit protocol, alongside the numerical history in the same record.
    """

    problem_key: str
    integrator: str
    time: float
    previous_dt: float
    states: tuple[tuple[float, ...], ...]
    velocity: tuple[float, ...] | None = None
    acceleration: tuple[float, ...] | None = None
    previous_time: Converted[float | None] = Converted(
        lambda value: value,
        default=None,
        resolve=lambda checkpoint, value: checkpoint.time - checkpoint.previous_dt if value is None else value,
    )
    material_history: dict | None = None
    problem_context: dict | None = None
    eec_state: EECCheckpointState | None = None
    step_origin: float | None = None
    event_state: EventCheckpointState | None = None

    def __post_init__(self) -> None:
        if self.problem_context is not None:
            if not isinstance(self.problem_context, dict):
                msg = 'checkpoint problem_context must be a JSON object'
                raise ValueError(msg)
            validate_context_json(self.problem_context)
        if self.material_history is not None:
            if not isinstance(self.material_history, dict):
                msg = 'checkpoint material_history must be a JSON object'
                raise ValueError(msg)
            validate_history_json(self.material_history)
        if (
            not isinstance(self.problem_key, str)
            or not self.problem_key.strip()
            or self.integrator
            not in (
                'bdf1',
                'bdf2',
                'tr_bdf2',
                'newmark',
                'generalized_alpha',
            )
        ):
            msg = 'checkpoint requires problem identity and integrator'
            raise ValueError(msg)
        _number(self.time)
        _number(self.previous_dt)
        if not math.isfinite(self.time) or not math.isfinite(self.previous_dt) or self.previous_dt <= 0:
            msg = 'checkpoint requires finite time and positive previous_dt'
            raise ValueError(msg)
        history = np.asarray(self.states, dtype=float)
        if history.ndim != 2 or not history.size or not np.all(np.isfinite(history)):
            msg = 'checkpoint requires finite, nonempty, equal-sized history states'
            raise ValueError(msg)
        for state in self.states:
            for value in state:
                _number(value)
        second_order = self.integrator in ('newmark', 'generalized_alpha')
        carried_rate = self.integrator == 'tr_bdf2'
        if (
            (second_order and (self.velocity is None or self.acceleration is None))
            or (carried_rate and (self.velocity is None or self.acceleration is not None))
            or (not second_order and not carried_rate and (self.velocity is not None or self.acceleration is not None))
        ):
            msg = 'checkpoint derivative history must match the integrator order'
            raise ValueError(msg)
        if len(self.states) > (2 if self.integrator == 'bdf2' else 1):
            msg = 'checkpoint history length does not match the integrator'
            raise ValueError(msg)
        if self.previous_time is not None:
            _number(self.previous_time)
            if not math.isclose(self.time - self.previous_time, self.previous_dt, rel_tol=1e-12, abs_tol=1e-15):
                msg = 'checkpoint history time and previous_dt are inconsistent'
                raise ValueError(msg)
        for vector in (self.velocity, self.acceleration):
            if vector is not None and (len(vector) != history.shape[1] or not np.all(np.isfinite(vector))):
                msg = 'checkpoint derivative history must match the state dimension'
                raise ValueError(msg)
            if vector is not None:
                for value in vector:
                    _number(value)
        if self.integrator in ('newmark', 'generalized_alpha') and (self.velocity is None or self.acceleration is None):
            msg = 'checkpoint second-order integrator requires derivative history'
            raise ValueError(msg)
        if self.step_origin is not None:
            _number(self.step_origin)
            if self.step_origin > self.time:
                msg = 'checkpoint step_origin must not follow its accepted time'
                raise ValueError(msg)
        if self.eec_state is not None:
            if type(self.eec_state) is not EECCheckpointState:
                msg = 'checkpoint eec_state must be an explicit native EECCheckpointState'
                raise ValueError(msg)
            self.eec_state.validate_checkpoint(self.time, self.integrator, self.states)
        if self.event_state is not None:
            if type(self.event_state) is not EventCheckpointState or self.eec_state is not None:
                msg_0 = 'event checkpoint requires its actual locator state without EEC'
                raise ValueError(msg_0)
            self.event_state.validate_clock(self.time, self.step_origin)

    def validate_restart(self, problem_key, integrator, time, size):
        if (problem_key, integrator, time, size) != (
            self.problem_key,
            self.integrator,
            self.time,
            len(self.states[-1]),
        ):
            msg = 'checkpoint does not match problem, integrator, restart time or state dimension'
            raise ValueError(msg)

    def to_mapping(self):
        mapping = {
            'schema_version': 3,
            'problem_key': self.problem_key,
            'integrator': self.integrator,
            'time': self.time,
            'previous_dt': self.previous_dt,
            'previous_time': self.previous_time,
            'states': [list(state) for state in self.states],
            'velocity': None if self.velocity is None else list(self.velocity),
            'acceleration': None if self.acceleration is None else list(self.acceleration),
            'material_history': self.material_history,
            'problem_context': self.problem_context,
        }
        if self.eec_state is not None:
            mapping['eec_state'] = self.eec_state.to_mapping()
        if self.step_origin is not None:
            mapping['step_origin'] = self.step_origin
        if self.event_state is not None:
            mapping['event_state'] = self.event_state.to_mapping()
        return mapping

    @classmethod
    def from_mapping(cls, data):
        expected = {
            'schema_version',
            'problem_key',
            'integrator',
            'time',
            'previous_dt',
            'previous_time',
            'states',
            'velocity',
            'acceleration',
            'material_history',
            'problem_context',
        }
        if (
            not isinstance(data, dict)
            or not expected <= set(data)
            or set(data) - expected - {'eec_state', 'step_origin', 'event_state'}
            or type(data['schema_version']) is not int
            or data['schema_version'] != 3
        ):
            msg = 'checkpoint schema must be version 3 with exactly the declared fields'
            raise ValueError(msg)
        if not isinstance(data['states'], list) or any(not isinstance(state, list) for state in data['states']):
            msg = 'checkpoint states must be a JSON array of arrays'
            raise ValueError(msg)
        return cls(
            problem_key=data['problem_key'],
            integrator=data['integrator'],
            time=_number(data['time']),
            previous_dt=_number(data['previous_dt']),
            previous_time=_number(data['previous_time']),
            material_history=data['material_history'],
            problem_context=data['problem_context'],
            eec_state=None if 'eec_state' not in data else EECCheckpointState.from_mapping(data['eec_state']),
            step_origin=None if 'step_origin' not in data else _number(data['step_origin']),
            event_state=None if 'event_state' not in data else EventCheckpointState.from_mapping(data['event_state']),
            states=tuple(tuple(_number(value) for value in state) for state in data['states']),
            velocity=None if data['velocity'] is None else tuple(_number(value) for value in data['velocity']),
            acceleration=(
                None if data['acceleration'] is None else tuple(_number(value) for value in data['acceleration'])
            ),
        )


def validate_history_json(value):
    """Validate plain JSON; physics owners validate their metadata and numeric schemas."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            msg = 'checkpoint history object keys must be strings'
            raise ValueError(msg)
        for item in value.values():
            validate_history_json(item)
    elif isinstance(value, list):
        for item in value:
            validate_history_json(item)
    elif value is not None and not isinstance(value, (str, bool)):
        _number(value)


def validate_context_json(value):
    """Physical clock/index metadata is bounded; large field histories belong in states."""
    remaining = 4096

    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 16 or (isinstance(item, str) and len(item) > 4096):
            msg = 'checkpoint physical context exceeds its size or nesting bound'
            raise ValueError(msg)
        if isinstance(item, dict):
            if any(not isinstance(key, str) or len(key) > 4096 for key in item):
                msg = 'checkpoint context requires bounded string object keys'
                raise ValueError(msg)
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, bool):
            _number(item)
        else:
            validate_history_json(item)

    visit(value, 0)


def validate_history_owner(problem):
    if owns_history(problem) and not isinstance(problem, CheckpointHistory):
        msg = 'checkpoint requires a material-history export/restore protocol'
        raise ValueError(msg)


def restore_history_checkpoint(problem, checkpoint):
    contextual = isinstance(problem, CheckpointContext)
    if contextual != (checkpoint.problem_context is not None):
        msg = 'checkpoint physical context does not match the problem context requirement'
        raise ValueError(msg)
    validate_history_owner(problem)
    stateful = owns_history(problem)
    if stateful != (checkpoint.material_history is not None):
        msg = 'checkpoint material history does not match the problem history requirement'
        raise ValueError(msg)
    if contextual:
        problem.restore_checkpoint_context(checkpoint.problem_context)
    if stateful:
        problem.restore_checkpoint_history(checkpoint.material_history)


def capture_history(problem):
    return problem.checkpoint_history() if owns_history(problem) else None


def owns_history(problem) -> bool:
    """Whether a march over ``problem`` must commit a material history -- THE one predicate.

    ``HistoryPresence`` is optional: a ``HistoryProblem`` that does not spell it always commits, and
    a problem that is not a ``HistoryProblem`` never does, whatever else it carries.
    """
    return isinstance(problem, HistoryProblem) and (not isinstance(problem, HistoryPresence) or problem.has_history)


def _checkpoint_of(problem, state, dt, key, name, *, eec_state=None, step_origin=None, event_state=None):
    if state.stage != 0:
        msg = 'portable checkpoint requires an accepted macrostep, not an intermediate stage'
        raise ValueError(msg)
    return TransientCheckpoint(
        key,
        name,
        state.t,
        dt,
        tuple(tuple(float(x) for x in u) for u in state.us),
        None if state.v is None else tuple(float(x) for x in state.v),
        None if state.a is None else tuple(float(x) for x in state.a),
        material_history=capture_history(problem),
        problem_context=problem.checkpoint_context(state.t) if isinstance(problem, CheckpointContext) else None,
        eec_state=eec_state,
        step_origin=step_origin,
        event_state=event_state,
    )
