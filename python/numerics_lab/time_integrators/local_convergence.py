"""Optional independent-field Newton changes, never a mixed-unit state norm."""

from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np

from motronics.gauss.nonlinear.local import local_change
from motronics.gauss.time_integrators._forms import _FLOOR_SAFETY

__all__ = [
    'LocalIterationCheck',
    'validate_local_atols',
]


@runtime_checkable
class _LocalFieldsOwner(Protocol):
    """A problem that names its independently converging fields."""

    def local_convergence_fields(self, u, t): ...


@runtime_checkable
class _LocalAtolsOwner(Protocol):
    """A problem that declares per-field absolute physical thresholds."""

    @property
    def local_convergence_atols(self): ...


def validate_local_atols(values):
    """Absolute bars are explicitly declared positive physical thresholds, never defaults."""
    for value in values:
        if value is not None and (
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0
        ):
            msg = 'local variation threshold must be finite positive and nonboolean'
            raise ValueError(msg)


class LocalIterationCheck:
    """Every independent field must stabilize alongside residual convergence.

    The owner exposes ``local_convergence_fields(u, t)``; the existing requested
    relative tolerance is reused. Numerical zero protection is ``local_change``'s
    per-field machine floor, not an absolute physical acceptance threshold.
    Optional owner ``local_convergence_atols`` replaces the relative criterion
    only for its corresponding explicitly declared physical quantity.
    """

    def __init__(self, problem, tolerance) -> None:
        self.fields = problem.local_convergence_fields if isinstance(problem, _LocalFieldsOwner) else None
        self.previous = None
        self.finite = True
        self.tolerance = tolerance
        self.atols = problem.local_convergence_atols if isinstance(problem, _LocalAtolsOwner) else None
        if self.atols is not None:
            validate_local_atols(self.atols)
        if self.fields is not None and (tolerance is None or not np.isfinite(tolerance) or tolerance <= 0):
            msg = 'local convergence fields require a finite positive relative tolerance'
            raise ValueError(msg)

    def observe(self, state, time):
        if self.fields is None:
            return True
        current = tuple(np.asarray(value).copy() for value in self.fields(state, time))
        previous, self.previous = self.previous, current
        self.finite = all(np.all(np.isfinite(value)) for value in current)
        if self.atols is not None and len(self.atols) != len(current):
            msg = 'local variation thresholds must match independent field groups'
            raise ValueError(msg)
        if not self.finite or previous is None:
            return False
        atols = (None,) * len(current) if self.atols is None else self.atols
        for old, new, atol in zip(previous, current, atols, strict=True):
            if old.shape != new.shape or not np.all(np.isfinite(old)):
                return False
            if atol is not None:
                if np.max(np.abs(new - old), initial=0.0) > atol:
                    return False
            else:
                # Independently scaled floating-point uncertainty must not make
                # a numerically zero V/I/A coordinate iterate forever. A large
                # voltage cannot set this floor for a small current field.
                scale = max(np.max(np.abs(old), initial=0.0), np.max(np.abs(new), initial=0.0))
                floor = _FLOOR_SAFETY * np.finfo(float).eps * scale
                rounded = np.where(np.abs(new - old) <= floor, old, new)
                if local_change((old,), (rounded,)) > self.tolerance:
                    return False
        return True

    def accepts(self, stable, residual, *, initial=False):
        # An exact seed needs no factorization (its Jacobian may be singular).
        return stable or (initial and residual == 0.0 and self.finite)
