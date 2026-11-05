"""Pure physical-state trial projection and limiting, shared by Newton and continuation."""

from __future__ import annotations

import numpy as np


def _physical(value, shape, owner):
    value = np.asarray(value)
    if value.shape != shape or not np.isrealobj(value) or not np.isfinite(value).all():
        msg = f'{owner} must return a same-shape finite real state'
        raise ValueError(msg)
    return value


def _lift(form, endpoint, physical, argument):
    alpha = form.weights.alpha0
    if alpha.imag != 0.0 or alpha.real == 0.0:
        msg = 'physical trial transformations require a nonzero real state weight'
        raise ValueError(msg)
    if np.array_equal(physical, argument):
        return endpoint
    # Do not cancel the small bound against the discarded, possibly enormous proposal.
    # A huge history offset can still make a physical value unrepresentable in endpoint
    # coordinates; all residual/admission checks use the reconstructed state, not this target.
    with np.errstate(over='ignore', invalid='ignore'):
        lifted = (physical - form.h_u) / alpha.real
    if not np.isfinite(lifted).all():
        msg = 'the bounded physical trial cannot be lifted to a finite endpoint'
        raise ValueError(msg)
    return lifted


def project_newton_trial(problem, form, previous, proposed, *, affine_projection, newton_limit, limit_changed=None):
    """Project, then limit, at the stage's physical time; neither operation advances history."""
    value = np.asarray(proposed, dtype=float)
    if limit_changed is not None:
        limit_changed[0] = False
    if affine_projection:
        argument = form.u_arg(value)
        projected = _physical(
            problem.project_affine_state(argument.copy(), form.t_eval), argument.shape, 'affine constraint owner'
        )
        value = _lift(form, value, projected, argument)
    if newton_limit:
        argument = form.u_arg(value)
        limited = _physical(
            problem.limit_newton_trial(form.u_arg(previous).copy(), argument.copy(), form.t_eval),
            argument.shape,
            'Newton trial limit',
        )
        if limit_changed is not None:
            limit_changed[0] = not np.array_equal(limited, argument)
        value = _lift(form, value, limited, argument)
    return value


def projection_fixed_point(problem, form, endpoint):
    """Conservative trial-ray eligibility without an admission certificate, NOT a convergence bar."""
    argument = form.u_arg(endpoint)
    projected = _physical(
        problem.project_affine_state(argument.copy(), form.t_eval), argument.shape, 'affine constraint owner'
    )
    return bool(np.array_equal(projected, argument))
