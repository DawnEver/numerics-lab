"""Portable numerical history for restarting a native transient, without pickle execution."""

import dataclasses
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from numerics_lab.formulation import HistoryProblem
from numerics_lab.time_integrators.eec_checkpoint import EECCheckpointState
from numerics_lab.time_integrators.event_checkpoint import EventCheckpointState

__all__ = [
    'CHECKPOINTED_MARCHES',
    'CHECKPOINT_FILE_SCHEMA',
    'FIXED_POINTS_OVER_A_MARCH',
    'CheckpointContext',
    'CheckpointContextAt',
    'CheckpointHistory',
    'EECCheckpointState',
    'HistoryPresence',
    'ScheduleCheckpointState',
    'StepControlCheckpointState',
    'TransientCheckpoint',
    'capture_history',
    'declared_identity',
    'model_digest',
    'read_checkpoint_file',
    'restore_history_checkpoint',
    'validate_context_json',
    'validate_history_json',
    'validate_history_owner',
    'write_checkpoint_file',
]


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        msg = 'checkpoint numeric values must be finite JSON numbers, not booleans or strings'
        raise ValueError(msg)
    return float(value)


def declared_identity(value):
    """A declared model object as plain values for :func:`model_digest`; nothing is copied.

    Dataclasses by field, mappings (a frozen ``mappingproxy`` included) by item, enums by value, a
    non-finite float spelled as data (JSON cannot). Anything else is handed on as it is, so
    :func:`model_digest` refuses what it cannot read -- an opaque callable never becomes an identity.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: declared_identity(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): declared_identity(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [declared_identity(v) for v in value]
    if isinstance(value, Enum):
        return declared_identity(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    return value


def _array_identity(value: np.ndarray) -> dict:
    """A numeric array digested by its BUFFER, not walked element by element as Python objects.

    2026-10-25: the element walk was 5.3M calls, 12 s of a case_a replay's model identity. JSON
    refused a non-finite value; so does this.
    """
    if value.dtype.kind in 'fc' and not np.isfinite(value).all():
        msg = 'checkpoint model contains a non-finite array value'
        raise ValueError(msg)
    buffer = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder('<'))
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(buffer).hexdigest()}


def model_digest(parts):
    """Hash explicit numerical model data; refuse opaque objects instead of hashing their repr."""

    def plain(value):
        if isinstance(value, np.ndarray):
            return (
                _array_identity(value)
                if value.dtype.kind in 'biufc'
                else {'shape': list(value.shape), 'dtype': str(value.dtype), 'values': plain(value.tolist())}
            )
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


@runtime_checkable
class CheckpointContextAt(CheckpointContext, Protocol):
    """Physical context with restoration at an explicit numerical accepted time."""

    def restore_checkpoint_context_at(self, data: dict, time: float) -> None: ...


@dataclass(frozen=True)
class ScheduleCheckpointState:
    """The sampled-schedule cursor: where the segment holding ``time`` started, and the tasks' history.

    A sampled march (``integrate_sampled``, ``integrate_explicit_sampled``) fires its due tasks at a
    segment's START, before any step in it, so a record taken inside or at the end of a segment has
    already fired that segment's tasks and not yet the next one's. ``segment_start`` is that segment's
    first bound: the resume re-derives the SAME schedule from it (the steps inside a segment are
    measured from its start, so a resume anchored elsewhere would land other instants). ``tasks`` is
    the plain-JSON history of the caller's task owner (held outputs, integrators, latches) through
    the :class:`CheckpointHistory` protocol, ``None`` when the march has no task.
    """

    segment_start: float
    tasks: dict | None = None

    def __post_init__(self) -> None:
        _number(self.segment_start)
        if self.tasks is not None:
            if not isinstance(self.tasks, dict):
                msg = 'checkpoint schedule tasks history must be a JSON object'
                raise ValueError(msg)
            validate_history_json(self.tasks)

    def to_mapping(self) -> dict:
        return {'segment_start': self.segment_start, 'tasks': self.tasks}

    @classmethod
    def from_mapping(cls, data) -> 'ScheduleCheckpointState':
        if not isinstance(data, dict) or set(data) != {'segment_start', 'tasks'}:
            msg = 'checkpoint schedule_state must carry exactly segment_start and tasks'
            raise ValueError(msg)
        return cls(_number(data['segment_start']), data['tasks'])


def _plain_tolerance(value) -> float | list[float]:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        msg = 'checkpoint step-control tolerance must be finite'
        raise ValueError(msg)
    return array.tolist()


@dataclass(frozen=True)
class StepControlCheckpointState:
    """The local-error controller's history: what an adaptive march decides its next step FROM.

    ``points`` are the last accepted ``(t, u)`` the predictor extrapolates through (at most ``order + 1``,
    the newest at the record's time; ONE after a start, a restart or an event, which is the controller's
    own bootstrap), ``next_step`` the step it proposed after the record's accepted step. ``tolerance`` is
    the ``{rtol, atol, h_min}`` it was taken under: a resume under another tolerance would be another march,
    so it refuses rather than reusing a history sized for different bars.
    """

    points: tuple[tuple[float, tuple[float, ...]], ...]
    next_step: float
    tolerance: dict

    def __post_init__(self) -> None:
        _number(self.next_step)
        if self.next_step <= 0.0 or not self.points:
            msg = 'checkpoint step-control state needs accepted points and a positive next step'
            raise ValueError(msg)
        times = [_number(t) for t, _u in self.points]
        if any(b <= a for a, b in itertools.pairwise(times)) or len({len(u) for _t, u in self.points}) != 1:
            msg = 'checkpoint step-control points must increase in time and share one state size'
            raise ValueError(msg)
        for _t, u in self.points:
            for value in u:
                _number(value)
        if set(self.tolerance) != {'rtol', 'atol', 'h_min'}:
            msg = 'checkpoint step-control tolerance must carry exactly rtol, atol and h_min'
            raise ValueError(msg)
        validate_history_json(self.tolerance)

    @classmethod
    def of(cls, points, next_step: float, tolerance) -> 'StepControlCheckpointState':
        return cls(
            tuple((float(t), tuple(float(x) for x in np.asarray(u, dtype=float))) for t, u in points),
            float(next_step),
            cls.tolerance_of(tolerance),
        )

    @staticmethod
    def tolerance_of(tolerance) -> dict:
        """The canonical plain form of a ``StepTolerance`` -- the one the record and a resume compare."""
        return {
            'rtol': _plain_tolerance(tolerance.rtol),
            'atol': _plain_tolerance(tolerance.atol),
            'h_min': None if tolerance.h_min is None else float(tolerance.h_min),
        }

    def to_mapping(self) -> dict:
        return {
            'points': [[t, list(u)] for t, u in self.points],
            'next_step': self.next_step,
            'tolerance': self.tolerance,
        }

    @classmethod
    def from_mapping(cls, data) -> 'StepControlCheckpointState':
        if (
            not isinstance(data, dict)
            or set(data) != {'points', 'next_step', 'tolerance'}
            or not isinstance(data['points'], list)
            or any(not isinstance(p, list) or len(p) != 2 or not isinstance(p[1], list) for p in data['points'])
        ):
            msg = 'checkpoint step_control_state must carry exactly points, next_step and tolerance'
            raise ValueError(msg)
        return cls(
            tuple((_number(t), tuple(_number(x) for x in u)) for t, u in data['points']),
            _number(data['next_step']),
            data['tolerance'],
        )


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
    #: Absent means one nominal step back, ``time - previous_dt``; resolved at construction.
    previous_time: float | None = None
    material_history: dict | None = None
    problem_context: dict | None = None
    eec_state: EECCheckpointState | None = None
    step_origin: float | None = None
    event_state: EventCheckpointState | None = None
    schedule_state: ScheduleCheckpointState | None = None
    step_control_state: StepControlCheckpointState | None = None
    #: The per-step Newton bar the producing march stopped on. A resume marches on it: a caller can only
    #: recompute its bar at the resume instant and through the restored material history, where a cosine drive's
    #: load or a hysteretic law's ``r(0)`` is another number -- and another bar is another trajectory.
    newton_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.previous_time is None:
            object.__setattr__(self, 'previous_time', self.time - self.previous_dt)
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
                'trapezoid',
                'newmark',
                'generalized_alpha',
                'rk4',
                'zoh',
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
        # A one-state scheme whose history is the state AND the rate it carries (trapezoid, TR-BDF2's stage).
        carried_rate = self.integrator in ('tr_bdf2', 'trapezoid')
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
        if self.newton_tolerance is not None and (
            not math.isfinite(_number(self.newton_tolerance)) or self.newton_tolerance <= 0.0
        ):
            msg = 'checkpoint newton_tolerance must be a finite positive bar'
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
        if self.schedule_state is not None and (
            type(self.schedule_state) is not ScheduleCheckpointState or self.schedule_state.segment_start > self.time
        ):
            msg = 'checkpoint schedule_state must be a ScheduleCheckpointState whose segment starts by its time'
            raise ValueError(msg)
        if self.step_control_state is not None and (
            type(self.step_control_state) is not StepControlCheckpointState
            or self.step_control_state.points[-1][0] != self.time
            or len(self.step_control_state.points[-1][1]) != history.shape[1]
        ):
            msg = 'checkpoint step-control points must end at the record time with its state size'
            raise ValueError(msg)

    def validate_restart(self, problem_key, integrator, time, size):
        asked = {'problem': problem_key, 'integrator': integrator, 'restart time': time, 'state dimension': size}
        held = {
            'problem': self.problem_key,
            'integrator': self.integrator,
            'restart time': self.time,
            'state dimension': len(self.states[-1]),
        }
        differ = [f'{name} {held[name]!r} != {asked[name]!r}' for name in asked if asked[name] != held[name]]
        if differ:
            msg = f'checkpoint does not match problem, integrator, restart time or state dimension: {"; ".join(differ)}'
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
        if self.schedule_state is not None:
            mapping['schedule_state'] = self.schedule_state.to_mapping()
        if self.step_control_state is not None:
            mapping['step_control_state'] = self.step_control_state.to_mapping()
        if self.newton_tolerance is not None:
            mapping['newton_tolerance'] = self.newton_tolerance
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
            or set(data)
            - expected
            - {'eec_state', 'step_origin', 'event_state', 'schedule_state', 'step_control_state', 'newton_tolerance'}
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
            newton_tolerance=None if 'newton_tolerance' not in data else _number(data['newton_tolerance']),
            event_state=None if 'event_state' not in data else EventCheckpointState.from_mapping(data['event_state']),
            schedule_state=(
                None if 'schedule_state' not in data else ScheduleCheckpointState.from_mapping(data['schedule_state'])
            ),
            step_control_state=(
                None
                if 'step_control_state' not in data
                else StepControlCheckpointState.from_mapping(data['step_control_state'])
            ),
            states=tuple(tuple(_number(value) for value in state) for state in data['states']),
            velocity=None if data['velocity'] is None else tuple(_number(value) for value in data['velocity']),
            acceleration=(
                None if data['acceleration'] is None else tuple(_number(value) for value in data['acceleration'])
            ),
        )


#: EVERY TIME MARCH, ``module:function``, and each takes ``checkpoint_key`` / ``on_checkpoint`` / ``restart``
#: and streams and resumes this module's :class:`TransientCheckpoint` (user ruling 2026-10-13: checkpoint and
#: resume are a general capability of every march). Every layer above marches THROUGH these (plan R2); the
#: one private march left outside gauss is the layering test's named ``_DEBT``, and it is not on the contract.``tests/architecture/layering/test_every_march_is_on_the_checkpoint_contract.py``
#: derives the march set from source and holds it equal to this one, both sides.
CHECKPOINTED_MARCHES: frozenset[str] = frozenset({
    'numerics_lab.time_integrators.march:integrate_transient',
    'numerics_lab.time_integrators.multirate:integrate_sampled',
    'numerics_lab.time_integrators.explicit:integrate_explicit_sampled',
    'numerics_lab.time_integrators.zoh:march_zoh',
})

#: FIXED POINTS OVER A MARCH, not marches: each iterates WHOLE windows or periods of a checkpointed march,
#: so there is no instant inside one to resume at. Its resumable state is its own iterate, which it already
#: takes back as its seed (a periodic state, a relaxed waveform).
FIXED_POINTS_OVER_A_MARCH: frozenset[str] = frozenset({
    'numerics_lab.time_integrators.waveform_relaxation:waveform_relaxation',
    'numerics_lab.time_integrators.shooting:periodic_shooting',
    'numerics_lab.time_integrators.periodic:periodic_fixed_point',
    'numerics_lab.time_integrators.periodic:periodic_steady_state',
    'numerics_lab.time_integrators.periodic_march:shoot_transient_cycle',
})

#: The sealed file envelope's version: the ONE on-disk form a :class:`TransientCheckpoint` is WRITTEN in.
CHECKPOINT_FILE_SCHEMA_VERSION = 2
_CHECKPOINT_FILE_PREFIX = 'numerics-lab-transient-checkpoint/v'
CHECKPOINT_FILE_SCHEMA = f'{_CHECKPOINT_FILE_PREFIX}{CHECKPOINT_FILE_SCHEMA_VERSION}'

#: FORMATS-UPGRADE-ON-READ (user ruling 2026-10-18): a checkpoint is the LIVE state a running march
#: resumes from, so an older envelope is read once into the current form; none is ever written.
#: Older foreign envelope -> the version of this lineage its sealed mapping already is.
#: ``motronics-transient-checkpoint/v1`` is what the predecessor package wrote before the engine moved here.
_PREDECESSOR_ENVELOPES: dict[str, int] = {'motronics-transient-checkpoint/v1': 1}


def _anchored_event_state(mapping: dict) -> dict:
    """v1 -> v2: the event locator's chained ``next_reanchor`` becomes the re-anchored grid's ``nominal_anchor``.

    v1 marched ``t_k = t_(k-1) + dt``; v2 marches ``anchor + k*dt``. A v1 state with no pending re-anchor is
    on the grid that started ``nominal_index`` steps before its accepted time, so that start IS the anchor.
    A pending re-anchor is v2's planned landing (anchor = pending target, index 0); one that is not the
    pending target has no v2 form and is refused by name, never guessed.
    """
    event = mapping.get('event_state')
    if event is None:
        return mapping
    event = dict(event)
    pending, k = event.pop('next_reanchor'), event['nominal_index']
    if pending is None:
        event['nominal_anchor'] = mapping['time'] - k * event['step_dt']
    elif pending == event['pending_target']:
        event.update(nominal_anchor=pending, nominal_index=0)
    else:
        msg = f'a v1 event re-anchor at {pending!r} that is not its pending target has no v2 nominal grid'
        raise ValueError(msg)
    return {**mapping, 'event_state': event}


#: Version -> the step taking its sealed ``checkpoint`` mapping one version on.
_UPGRADES: dict[int, Callable[[dict], dict]] = {1: _anchored_event_state}


def _file_version(schema: object) -> int | None:
    if not isinstance(schema, str):
        return None
    if schema in _PREDECESSOR_ENVELOPES:
        return _PREDECESSOR_ENVELOPES[schema]
    digits = schema.removeprefix(_CHECKPOINT_FILE_PREFIX)
    return int(digits) if digits != schema and digits.isdecimal() else None


def _seal(mapping) -> str:
    canonical = json.dumps(mapping, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def write_checkpoint_file(path, checkpoint: TransientCheckpoint) -> Path:
    """Write ``checkpoint`` sealed by its sha256, atomically: a kill mid-write leaves the previous file whole.

    JSON's shortest float repr round-trips every finite double exactly, so a resume from this file is
    bit-identical to the uninterrupted march.
    """
    path = Path(path)
    mapping = checkpoint.to_mapping()
    payload = json.dumps(
        {'schema': CHECKPOINT_FILE_SCHEMA, 'sha256': _seal(mapping), 'checkpoint': mapping}, allow_nan=False
    )
    staging = path.with_name(f'{path.name}.tmp')
    staging.write_text(payload, encoding='utf-8')
    staging.replace(path)
    return path


def read_checkpoint_file(path) -> TransientCheckpoint:
    """Read a sealed checkpoint; an edited, truncated or unsealed file is REFUSED, never resumed from.

    An OLDER envelope is upgraded on read through :data:`_UPGRADES` after its seal is checked; an
    envelope this reader does not know (newer, never issued, or foreign) is refused by name.
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or set(data) != {'schema', 'sha256', 'checkpoint'}:
        msg = f'{path}: not a sealed {CHECKPOINT_FILE_SCHEMA} checkpoint file'
        raise ValueError(msg)
    if _seal(data['checkpoint']) != data['sha256']:
        msg = f'{path}: checkpoint content does not match its sha256 seal; the file was edited or corrupted'
        raise ValueError(msg)
    version, mapping = _file_version(data['schema']), data['checkpoint']
    while version in _UPGRADES:
        mapping, version = _UPGRADES[version](mapping), version + 1
    if version != CHECKPOINT_FILE_SCHEMA_VERSION:
        msg = f'{path}: checkpoint schema {data["schema"]!r} is unknown to this reader ({CHECKPOINT_FILE_SCHEMA})'
        raise ValueError(msg)
    return TransientCheckpoint.from_mapping(mapping)


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
        if isinstance(problem, CheckpointContextAt):
            # The numerical record supplies the clock. Metadata cannot choose
            # its own validation time, nor is a producer's construction time it.
            problem.restore_checkpoint_context_at(checkpoint.problem_context, checkpoint.time)
        else:
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


def _checkpoint_of(
    problem,
    state,
    dt,
    key,
    name,
    *,
    previous_time=None,
    eec_state=None,
    step_origin=None,
    event_state=None,
    step_control_state=None,
    newton_tolerance=None,
):
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
        previous_time=previous_time,
        material_history=capture_history(problem),
        problem_context=problem.checkpoint_context(state.t) if isinstance(problem, CheckpointContext) else None,
        eec_state=eec_state,
        step_origin=step_origin,
        event_state=event_state,
        step_control_state=step_control_state,
        newton_tolerance=newton_tolerance,
    )
