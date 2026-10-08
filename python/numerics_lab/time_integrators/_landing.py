"""Numerical accepted-lattice callback; no event location or physical owner knowledge."""

from numbers import Integral, Real

import numpy as np


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
        return state
    observation = state.copy()
    observation.flags.writeable = False
    projected = np.asarray(callback(time, observation))
    if projected.shape != state.shape or not np.isrealobj(projected) or not np.all(np.isfinite(projected)):
        msg = 'accepted projection must return a finite real state with unchanged shape'
        raise ValueError(msg)
    return np.array(projected, dtype=float, copy=True)


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
