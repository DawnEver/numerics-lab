"""Pure physical-state trial projection and limiting, shared by Newton and continuation."""

from __future__ import annotations

import numpy as np


def _physical(value, shape, owner):
    value = np.asarray(value)
    if value.shape != shape or not np.isrealobj(value) or not np.isfinite(value).all():
        msg = f'{owner} must return a same-shape finite real state'
        raise ValueError(msg)
    return value


def _lift(form, endpoint, physical, argument, *, incremental=False):
    alpha = form.weights.alpha0
    if alpha.imag != 0.0 or alpha.real == 0.0:
        msg = 'physical trial transformations require a nonzero real state weight'
        raise ValueError(msg)
    if np.array_equal(physical, argument):
        return endpoint
    # Affine projection corrects the existing endpoint; rebuilding it from a
    # large history offset needlessly loses precision. A limiter instead owns
    # the small bounded physical point, not the discarded enormous proposal.
    with np.errstate(over='ignore', invalid='ignore'):
        lifted = endpoint + (physical - argument) / alpha.real if incremental else (physical - form.h_u) / alpha.real
    # A changed row grants no authority to perturb untouched endpoint rows.
    lifted = np.where(physical == argument, endpoint, lifted)
    if not np.isfinite(lifted).all():
        msg = 'the bounded physical trial cannot be lifted to a finite endpoint'
        raise ValueError(msg)
    return lifted


def project_newton_trial(problem, form, previous, proposed, *, affine_projection, newton_limit):
    """Return the endpoint and whether its owner limited it; do not advance history."""
    value = np.asarray(proposed, dtype=float)
    limited_trial = False
    if affine_projection:
        argument = form.u_arg(value)
        projected = _physical(
            problem.project_affine_state(argument.copy(), form.t_eval), argument.shape, 'affine constraint owner'
        )
        value = _lift(form, value, projected, argument, incremental=True)
    if newton_limit:
        argument = form.u_arg(value)
        limited = _physical(
            problem.limit_newton_trial(form.u_arg(previous).copy(), argument.copy(), form.t_eval),
            argument.shape,
            'Newton trial limit',
        )
        limited_trial = not np.array_equal(limited, argument)
        value = _lift(form, value, limited, argument)
    return value, limited_trial


def projection_fixed_point(problem, form, endpoint):
    """Conservative trial-ray eligibility without an admission certificate, NOT a convergence bar."""
    argument = form.u_arg(endpoint)
    projected = _physical(
        problem.project_affine_state(argument.copy(), form.t_eval), argument.shape, 'affine constraint owner'
    )
    return bool(np.array_equal(projected, argument))
