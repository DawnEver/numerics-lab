"""Physics-blind time-integration runtime -- BDF / Newmark / generalized-alpha (plan R5).

This is the temporal counterpart of ``gauss/nonlinear`` (iteration strategies) and
``gauss/linear`` (linear-solve backends): a registry of pluggable TIME INTEGRATORS that a
transient study marches with, decoupled from any physics. The orthogonality hinge is
:class:`~motronics.contracts.spec.analysis_formulation.DerivativeWeights`: an analysis formulation
supplies ONLY the triple ``(alpha0, alpha1, alpha2)`` weighting the ``u`` / ``u_dot`` /
``u_ddot`` operators; a time integrator is exactly the machinery that PRODUCES those weights
per step (plus the history-dependent affine offsets) and advances a
:class:`~motronics.contracts.spec.analysis_formulation.SemiDiscreteProblem` one step.

The load-bearing idea: an implicit integrator writes each derivative at the new step as an
AFFINE function of the unknown ``u_{n+1}``,

    u_dot   = alpha1 * u_{n+1} + h_dot        (history)
    u_ddot  = alpha2 * u_{n+1} + h_ddot

so the residual root-find for ``u_{n+1}`` is driven purely through ``residual`` /
``jacobian(weights)`` -- the integrator never sees the physics, only these numbers. The
``alpha1`` / ``alpha2`` slopes ARE the :class:`DerivativeWeights` a study would carry; the
``h_dot`` / ``h_ddot`` offsets are the history the backend cannot know. First-order schemes
(BDF1/BDF2) leave ``alpha2 = 0`` (the eddy-current / diffusion form ``u_dot``); second-order
schemes (Newmark, generalized-alpha) carry ``alpha2`` (the structural ``u_ddot``).

This module imports ONLY contracts + numpy/scipy (enforced by
``test_import_purity.py::test_importing_gauss_pulls_no_physics_into_sys_modules``); it touches no
mesh, no material, no solver.
"""

from __future__ import annotations

import logging
from functools import wraps

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from motronics.contracts.spec.analysis_formulation import (
    DerivativeWeights,
    FactorReusableProblem,
    SemiDiscreteProblem,
)
from motronics.contracts.spec.study_spec import TransientStudy
from motronics.gauss.linear import solve_linear_system
from motronics.gauss.time_integrators import _step
from motronics.gauss.time_integrators._forms import (
    IntegratorState,
    ITimeIntegrator,
    StepForm,
    TransientSolution,
    residual_floor,
    residual_floor_of_magnitude,
)
from motronics.gauss.time_integrators._integrators import (
    INTEGRATOR_REGISTRY,
    BDF1Integrator,
    BDF2Integrator,
    GeneralizedAlphaIntegrator,
    NewmarkIntegrator,
    get_integrator,
)
from motronics.gauss.time_integrators._step import _GMIN_RETRY_STAGES as _GMIN_RETRY_STAGES
from motronics.gauss.time_integrators._step import HELD_STALL_RATIO, HeldFactor
from motronics.gauss.time_integrators._step import _gmin_continue as _gmin_continue
from motronics.gauss.time_integrators._step import _roundoff_floor as _roundoff_floor
from motronics.gauss.time_integrators.factor_reuse import FactorCache

logger = logging.getLogger(__name__)


__all__ = [
    'HELD_STALL_RATIO',
    'INTEGRATOR_REGISTRY',
    'BDF1Integrator',
    'BDF2Integrator',
    'FactorCache',
    'GeneralizedAlphaIntegrator',
    'HeldFactor',
    'ITimeIntegrator',
    'IntegratorState',
    'NewmarkIntegrator',
    'StepForm',
    'TransientSolution',
    'bdf2_variable_form',
    'get_integrator',
    'integrate_transient',
    'residual_floor',
    'residual_floor_of_magnitude',
]


def _full_fixed_steps(study: TransientStudy) -> int:
    """Count full cells, treating a floating-point near-integer as an integer."""
    ratio = (study.t1 - study.t0) / study.dt
    nearest = round(ratio)
    return nearest if abs(ratio - nearest) <= 1e-10 else int(np.floor(ratio))


def _bdf2_variable_form(state: IntegratorState, dt: float, h: float, t_new: float) -> StepForm:
    """The variable-spacing BDF2 step form; ``h`` is the previous step's size.

    Fixed-weight BDF2 assumes uniform history spacing; a step whose size differs from its
    history's spacing needs the derivative of the quadratic through the unevenly spaced
    points (the events module docstring names the O(1) wrong-equation trap this closes).
    """
    u_n, u_nm1 = state.us[-1], state.us[-2]
    return StepForm(
        weights=DerivativeWeights(alpha0=1.0, alpha1=(2.0 * dt + h) / (dt * (dt + h))),
        h_u=np.zeros_like(u_n),
        h_dot=-(dt + h) / (dt * h) * u_n + dt / (h * (dt + h)) * u_nm1,
        h_ddot=np.zeros_like(u_n),
        t_eval=t_new,
    )


#: The public spelling of the variable-step BDF2 form, for a CHECK that must replay the march's own
#: weights (`hamilton.coupling.terminal_balance`). One function, two names, never two formulas.
bdf2_variable_form = _bdf2_variable_form


def _default_solve(matrix: object, rhs: NDArray[np.floating]) -> NDArray[np.floating]:
    """Solve ``matrix @ x = rhs`` for a dense ndarray or a scipy-sparse jacobian (physics-blind)."""
    b = np.asarray(rhs, dtype=float)
    if sp.issparse(matrix):
        return np.asarray(solve_linear_system(sp.csc_matrix(matrix), b))
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 0:  # scalar jacobian (1-DOF problem written as a plain number)
        return b / float(m)
    return np.linalg.solve(m, b)


@wraps(_step._solve_step)
def _solve_step(*args, **kwargs):
    """Retain the public patch seam; the implicit algorithm has one owner in ``_step``."""
    return _step._solve_step(*args, **kwargs)


def integrate_transient(
    problem: SemiDiscreteProblem,
    study: TransientStudy,
    u0: NDArray[np.floating],
    *,
    v0: NDArray[np.floating] | None = None,
    a0: NDArray[np.floating] | None = None,
    solve=_default_solve,
    max_newton: int = 1,
    tol: float = 1e-12,
    tol_rel: float | None = None,
    factor_cache: FactorCache | None = None,
    backtrack_band: float = 1.5,
    backtrack_cap: int = 4,
    gmin_stages: tuple[float, ...] | None = None,
    adaptive: bool = False,
    atol: float = 1e-6,
    h_min: float | None = None,
    h_max: float | None = None,
    refactor_every: int | None = None,
) -> TransientSolution:
    """March ``problem`` from ``study.t0`` to ``study.t1`` by ``study.dt`` with ``study.integrator``.

    Steps the selected :class:`ITimeIntegrator`: each step ``prepare`` yields the affine
    :class:`StepForm`, :func:`_solve_step` advances the residual/jacobian seam for ``u_{n+1}``, and
    ``advance`` folds it back into the history. ``v0`` / ``a0`` seed the second-order schemes
    (default zero). ``max_newton = 1`` is the exact linear step; raise it (or swap ``solve``) for a
    nonlinear problem. Returns the full march including the initial condition.

    ``factor_cache`` (W6) reuses one factorization per reuse key across the march (see
    :mod:`~motronics.gauss.time_integrators.factor_reuse`); the default ``None`` builds a fresh
    cache when the problem exposes ``factor_reuse_key`` and none otherwise, so a keyed problem
    reuses by default and an unkeyed one is untouched. Pass a cache in to READ the counters.
    """
    # The MODIFIED Newton (ruling 3A): one held factor across the whole march, or None (full Newton).
    held = HeldFactor(refactor_every) if refactor_every is not None else None
    if factor_cache is None and isinstance(problem, FactorReusableProblem):
        factor_cache = FactorCache()
    integrator = get_integrator(study.integrator)
    if adaptive and integrator.name != 'bdf2':
        msg = f'adaptive stepping is implemented for BDF2 only, got integrator {integrator.name}'
        raise ValueError(msg)
    u0 = np.asarray(u0, dtype=float)
    state = integrator.start(u0, study.t0, v0, a0)
    dt = study.dt
    if adaptive:
        # The ADAPTIVE march (W6, OTSF from the simba notes): the BDF2's local truncation
        # error is estimated by the difference between the BDF2 step and the linear
        # extrapolation from the history (the second-order difference is O(h^3), so the
        # controller h *= (atol/err)^(1/3) drives the LTE to atol). A rejected step halves
        # and retries; an accepted step's size is grown by the controller's own gain. The
        # event locator's bracket-and-bisect machinery is step-agnostic, so the adaptive
        # steps and the eventful march compose -- the events' own adaptive wiring is a
        # follow-up (the events.py cells are fixed-grid by construction).
        h = dt
        h_prev = dt  # the last ACCEPTED step's size (the history's spacing, for the LTE)
        h_lo = dt / 1000.0 if h_min is None else h_min
        h_hi = dt * 1000.0 if h_max is None else h_max
        t_end = study.t1
        solution = TransientSolution(times=[study.t0], states=[u0])
        while state.t < t_end - 1e-14:
            h = min(h, t_end - state.t)
            t_new = state.t + h
            # The variable-spacing BDF2 form whenever the step differs from the history's
            # spacing: the uniform prepare would differentiate against the WRONG history
            # (measured 2026-08-07: the first retry after a rejection was off by |du|, and
            # the controller spun at the floor chasing a phantom error).
            if len(state.us) >= 2 and h != h_prev:
                form = _bdf2_variable_form(state, h, h_prev, t_new)
            else:
                form = integrator.prepare(state, h, t_new)
            # RETIRED 2026-09-12: this call used to raise the bar to ``max(tol, 1e-10)`` for the
            # adaptive march ALONE, on the correct diagnosis that "at the tiny steps the variable
            # form's history offsets cancel at the 1/h scale, so the residual's noise floor is
            # ~eps*u/h". The diagnosis was right and the remedy was a NUMBER -- right at h=1e-6,
            # wrong at h=1e-9, and blind to the supply besides. :func:`residual_floor` IS that
            # noise floor computed, and ``|J|`` carries the 1/h weights, so the same statement now
            # holds at every h and every voltage AND on the fixed-grid path, which never had it.
            u_new, converged, _it, residual, on_floor = _solve_step(
                problem,
                form,
                state.last_u,
                solve=solve,
                max_newton=max_newton,
                tol=tol,
                tol_rel=tol_rel,
                factor_cache=factor_cache,
                backtrack_band=backtrack_band,
                backtrack_cap=backtrack_cap,
                gmin_stages=gmin_stages,
                held=held,
            )
            # The LTE: |u_new - u_extrap| where u_extrap is the linear continuation from the
            # last two states, SCALED by the spacing ratio h/h_prev -- the naive 2u1-u0 form
            # assumes h == h_prev and overstates the error by |du|*|1 - h/h_prev| on the
            # first retry after a rejection (measured 2026-08-07: the controller then spun
            # at the floor).
            if len(state.us) >= 2:
                u_prev = np.asarray(state.us[-2], dtype=float)
                u_extrap = state.last_u + (state.last_u - u_prev) * (h / h_prev)
                err = float(np.linalg.norm(u_new - u_extrap))
                err = max(err, 1e-300)
            else:
                err = 0.0  # the first step: no history to extrapolate from -- accept it
            if h <= h_lo or (converged and (err == 0.0 or err <= atol)):
                # accept the step -- at the h_lo floor REGARDLESS of the convergence (the
                # step is taken with the contract's label: the nonconverged_times + the
                # warning -- a step that cannot converge at the floor must not spin forever,
                # measured 2026-08-07: the retry loop at h_lo with conv=False never exited)
                solution.worst_step_residual = max(solution.worst_step_residual, residual)
                if not converged:
                    solution.nonconverged_times.append(t_new)
                elif on_floor:
                    # ONE spelling of this rule, shared with the fixed-grid march below: a step is
                    # recorded against exactly one of the two lists, because `on_floor` is only
                    # ever set where the floor MADE the verdict converged.
                    solution.floor_converged_times.append(t_new)
                state = integrator.advance(state, u_new, h, t_new)
                solution.times.append(t_new)
                solution.states.append(u_new)
                h_prev = h
                if err > 0.0:
                    h = min(h_hi, max(h_lo, h * min(2.0, 1.2 * (atol / err) ** (1.0 / 3.0))))
            else:
                # reject: halve and retry (bounded by h_lo -- at the floor the step is taken
                # with the contract's label rather than spun forever)
                h = max(h_lo, h / 2.0)
        return solution
    n_full = _full_fixed_steps(study)
    solution = TransientSolution(times=[study.t0], states=[u0])
    has_remainder = study.t1 - (study.t0 + n_full * dt) > 1e-12 * max(dt, abs(study.t1 - study.t0))
    for k in range(n_full + int(has_remainder)):
        final_partial = k == n_full
        t_new = study.t1 if final_partial else study.t0 + (k + 1) * dt
        step_dt = t_new - state.t if final_partial else dt
        if final_partial and integrator.name == 'bdf2' and len(state.us) >= 2:
            form = _bdf2_variable_form(state, step_dt, dt, t_new)
        else:
            form = integrator.prepare(state, step_dt, t_new)
        u_new, converged, _iterations, residual, on_floor = _solve_step(
            problem,
            form,
            state.last_u,
            solve=solve,
            max_newton=max_newton,
            tol=tol,
            tol_rel=tol_rel,
            factor_cache=factor_cache,
            backtrack_band=backtrack_band,
            backtrack_cap=backtrack_cap,
            gmin_stages=gmin_stages,
            held=held,
        )
        solution.worst_step_residual = max(solution.worst_step_residual, residual)
        if on_floor:
            solution.floor_converged_times.append(t_new)
        elif not converged:
            solution.nonconverged_times.append(t_new)
            # The CONVERGENCE CONTRACT, label-don't-block (AGENTS.md first principle): the
            # march continues, the result carries the measured deviation, and the caller
            # reads the tag -- a silent accept is the forbidden shape. One warning per
            # exhaustion event, carrying the measured residual.
            logger.warning(
                'step at t=%s did not converge in %s Newton iteration(s): residual %s exceeds '
                'both the absolute tol %s and the arithmetic floor of this system',
                t_new,
                max_newton,
                f'{residual:.3e}',
                f'{tol:.3e}',
            )
        state = integrator.advance(state, u_new, step_dt, t_new)
        solution.times.append(t_new)
        solution.states.append(u_new)
    if held is not None:
        logger.info('held-factor march: factorizations=%d solves=%d', held.factorizations, held.solves)
    return solution
