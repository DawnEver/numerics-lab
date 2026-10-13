"""Numerical accepted-lattice callback; no event location or physical owner knowledge."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

__all__ = [
    'AcceptedProjectionResult',
    'accepted_landing',
    'accepted_observation',
    'accepted_projection',
    'live_targets',
    'next_deadline',
]


@dataclass(frozen=True)
class AcceptedProjectionResult:
    """An owner's actual outgoing state and explicitly available derivative pair."""

    state: np.ndarray
    rate: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    prepared_restart: object | None = None

    def __post_init__(self) -> None:
        state = np.asarray(self.state)
        for name in ('state', 'rate', 'acceleration'):
            value = getattr(self, name)
            if value is None:
                continue
            primitives = value if isinstance(value, np.ndarray) else np.asarray(value, dtype=object)
            if primitives.dtype.kind == 'b' or (
                primitives.dtype.kind == 'O' and any(isinstance(item, (bool, np.bool_)) for item in primitives.flat)
            ):
                msg = 'accepted projection requires aligned finite real state and declared derivatives'
                raise ValueError(msg)
            array = np.asarray(value)
            if state.ndim != 1 or array.shape != state.shape or not np.isrealobj(array) or not np.isfinite(array).all():
                msg = 'accepted projection requires aligned finite real state and declared derivatives'
                raise ValueError(msg)
            owned = np.array(array, dtype=float, copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)


def next_deadline(callback, time, target):
    """Clip a numerical target to an owner's live, strictly future breakpoint."""
    if callback is None:
        return target
    deadline = callback(time)
    if deadline is None:
        return target
    if (
        isinstance(deadline, (bool, np.bool_))
        or not isinstance(deadline, Real)
        or not np.isfinite(deadline)
        or deadline <= time
    ):
        msg = 'next breakpoint must be a finite, strictly future time or None'
        raise ValueError(msg)
    return min(target, float(deadline))


def live_targets(targets, current_time, callback):
    """Re-read the owner's schedule after each acceptance, not during Newton probes."""
    for target in targets:
        while current_time() < target:
            yield next_deadline(callback, current_time(), target)


def accepted_projection(callback, time, state):
    """Validate the owner's right-side state without assigning any physical rate."""
    if callback is None:
        return AcceptedProjectionResult(state)
    observation = state.copy()
    observation.flags.writeable = False
    selected = callback(time, observation)
    projected = selected if isinstance(selected, AcceptedProjectionResult) else AcceptedProjectionResult(selected)
    if projected.state.shape != state.shape:
        msg = 'accepted projection must return a finite real state with unchanged shape'
        raise ValueError(msg)
    return projected


def accepted_observation(callback, time, state):
    """Observe canonical state, never a Newton/event probe; report a genuine law change."""
    observation = state.copy()
    observation.flags.writeable = False
    changed = callback(time, observation)
    if not isinstance(changed, (bool, np.bool_)):
        msg = 'accepted observation must return a boolean discontinuity declaration'
        raise TypeError(msg)
    return bool(changed)


def accepted_landing(callback, time, state):
    """Observe an immutable accepted state and return unique discontinuity identifiers."""
    observation = state.copy()
    observation.flags.writeable = False
    indices = tuple(callback(time, observation))
    if any(isinstance(index, bool) or not isinstance(index, Integral) or index < 0 for index in indices):
        msg = 'accepted landing requires nonnegative integer event identifiers'
        raise ValueError(msg)
    if len(set(indices)) != len(indices):
        msg = 'accepted landing event identifiers must be unique'
        raise ValueError(msg)
    return indices
