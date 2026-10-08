"""Numerical accepted-lattice callback; no event location or physical owner knowledge."""

from numbers import Integral

import numpy as np


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
