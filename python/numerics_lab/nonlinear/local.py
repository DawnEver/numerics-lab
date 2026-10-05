"""Local convergence without mixing magnetic, voltage, or current units."""

import numpy as np


def local_change(previous, current) -> float:
    """Worst relative change over independent field arrays (element axis first).

    A zero-to-nonzero element must not disappear in a global energy norm. The
    machine-precision floor only protects zeros relative to that field's scale.
    """
    if len(previous) != len(current):
        msg = 'local convergence fields changed during iteration'
        raise ValueError(msg)
    maximum = 0.0
    for old, new in zip(previous, current, strict=True):
        old, new = np.asarray(old), np.asarray(new)
        if old.shape != new.shape:
            msg = 'local convergence field shape changed during iteration'
            raise ValueError(msg)
        if old.size == 0:
            continue
        if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
            return float('inf')
        old = old.reshape(old.shape[0], -1)
        new = new.reshape(new.shape[0], -1)
        change = np.hypot.reduce(np.abs(new - old), axis=1)
        scale = np.maximum(np.hypot.reduce(np.abs(old), axis=1), np.hypot.reduce(np.abs(new), axis=1))
        if not np.all(np.isfinite(change)) or not np.all(np.isfinite(scale)):
            return float('inf')
        floor = np.finfo(float).eps * float(scale.max())
        ratio = np.divide(change, np.maximum(scale, floor), out=np.zeros_like(change), where=scale > 0)
        maximum = max(maximum, float(ratio.max()))
    return maximum


def newton_linear_tolerance(residual_ratio: float, *, initial=1e-2, minimum=1e-10) -> float:
    """Adaptive absolute tolerance relative to the original RHS reference.

    The caller multiplies by ||original RHS||, matching JMAG Automatic's
    documented scaling rather than resetting the reference at every Newton step.
    """
    if not np.isfinite(residual_ratio) or residual_ratio < 0 or not 0 < minimum <= initial < 1:
        msg = 'invalid Newton tolerance policy'
        raise ValueError(msg)
    return max(minimum, min(initial, 0.1 * residual_ratio**1.5))
