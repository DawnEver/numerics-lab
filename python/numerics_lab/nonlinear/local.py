"""Local convergence without mixing magnetic, voltage, or current units."""

import numpy as np


def free_residual_norm(system, x, state) -> float:
    """Free-DOF residual norm, or the protocol default for duck-typed equations."""
    try:
        norm_of = system.residual_norm
    except AttributeError:
        return float(np.linalg.norm(system.residual(x, state)))
    return float(norm_of(x, state))


def residual_reference(system, state, seed_norm: float) -> float:
    """``max(||r(x_0)||, ||rhs||)``: a good warm seed must not tighten the target.

    Without a source-norm hook the seed residual is the reference, exact for a
    zero seed. The reference remains fixed across Newton iterations.
    """
    try:
        rhs_norm = system.rhs_norm
    except AttributeError:
        return seed_norm
    return max(seed_norm, float(rhs_norm(state)))


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
