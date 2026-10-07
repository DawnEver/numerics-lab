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
from collections.abc import Callable
from functools import wraps
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from motronics.contracts.spec.analysis_formulation import (
    DerivativeWeights,
    FactorReusableProblem,
    HistoryProblem,
    SemiDiscreteProblem,
)
from motronics.contracts.spec.study_spec import TransientStudy
from motronics.gauss.linear import solve_linear_system
from motronics.gauss.time_integrators import _step
from motronics.gauss.time_integrators._forms import IntegratorState, StepForm, TransientSolution, seeded_bdf2_state
from motronics.gauss.time_integrators._integrators import (
    INTEGRATOR_REGISTRY,
    BDF1Integrator,
    BDF2Integrator,
    GeneralizedAlphaIntegrator,
    NewmarkIntegrator,
    get_integrator,
)
from motronics.gauss.time_integrators._step import HeldFactor, directional_line_search
from motronics.gauss.time_integrators._time_lattice import adaptive_lattice
from motronics.gauss.time_integrators.checkpoint import (
    EECCheckpointState,
    TransientCheckpoint,
    _checkpoint_of,
    owns_history,
    restore_history_checkpoint,
    validate_history_owner,
)
from motronics.gauss.time_integrators.eec_tangent import CycleTangentCSR
from motronics.gauss.time_integrators.factor_reuse import FactorCache
from motronics.gauss.time_integrators.periodic import (
    bdf2_boundary_moment,
    canonical_eec_study,
    explicit_error_correction,
    nonlinear_error_correction,
    periodic_converged,
    validate_eec_restart,
)

logger = logging.getLogger(__name__)


__all__ = [
    'INTEGRATOR_REGISTRY',
    'BDF1Integrator',
    'BDF2Integrator',
    'FactorCache',
    'GeneralizedAlphaIntegrator',
    'NewmarkIntegrator',
    'bdf2_variable_form',
    'directional_line_search',
    'get_integrator',
    'integrate_transient',
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
    """The package's step seam; the implicit algorithm has one owner in ``_step``."""
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
    backtrack_band: float | None = 1.5,
    backtrack_cap: int = 4,
    line_search: str = 'backtrack',
    gmin_stages: tuple[float, ...] | None = None,
    adaptive: bool = False,
    atol: float = 1e-6,
    h_min: float | None = None,
    h_max: float | None = None,
    refactor_every: int | None = None,
    step_times: tuple[float, ...] | None = None,
    step_origin: float | None = None,
    steady_period: float | None = None,
    steady_atol: float = 1e-6,
    steady_rtol: float = 1e-3,
    restart: TransientCheckpoint | None = None,
    checkpoint_key: str | None = None,
    on_checkpoint: Callable[[TransientCheckpoint], None] | None = None,
    eec_period: float | None = None,
    eec_sign: int = 1,
    eec_affine_first_order: bool = False,
    eec_nonlinear_first_order: bool = False,
    initial_history: tuple[np.ndarray, ...] | None = None,
    timestep_quantum: float | None = None,
    timestep_origin: float | None = None,
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

    ``eec_period`` enables correction at the declared interval. ``eec_sign=-1``
    requires anti-periodic forcing over that interval and uniform BDF1/BDF2 steps.
    The generic
    residual protocol cannot prove linearity or time invariance: callers must
    explicitly assert ``eec_affine_first_order=True`` only for
    ``M u_dot + K u = f(t)`` with constant K and M and periodic f. No nonlinear,
    second-order, moving-operator, or material-history correction is supported.
    """
    # The MODIFIED Newton (ruling 3A): one held factor across the whole march, or None (full Newton).
    held = HeldFactor(refactor_every) if refactor_every is not None else None
    if factor_cache is None and isinstance(problem, FactorReusableProblem):
        factor_cache = FactorCache()
    study = canonical_eec_study(study, restart)
    integrator = get_integrator(study.integrator)
    if eec_nonlinear_first_order and (eec_period is None or eec_affine_first_order):
        msg = 'nonlinear EEC needs an active interval and its distinct nonlinear operator assertion'
        raise ValueError(msg)
    if (
        isinstance(eec_sign, bool)
        or not isinstance(eec_sign, Integral)
        or eec_sign not in (-1, 1)
        or (eec_sign == -1 and eec_period is None)
    ):
        msg = 'eec_sign must be explicit +1 or -1 with an active correction interval'
        raise ValueError(msg)
    if steady_period is not None and (not np.isfinite(steady_period) or steady_period <= 0):
        msg = 'steady_period must be finite and positive'
        raise ValueError(msg)
    if steady_atol < 0 or steady_rtol < 0 or not np.isfinite(steady_atol + steady_rtol):
        msg = 'steady tolerances must be finite and nonnegative'
        raise ValueError(msg)
    if step_origin is not None and (
        step_times is None
        or isinstance(step_origin, bool)
        or not isinstance(step_origin, Real)
        or not np.isfinite(step_origin)
        or step_origin > study.t0
        or eec_period is not None
    ):
        msg = 'step_origin requires an explicit finite fixed schedule without EEC'
        raise ValueError(msg)
    if restart is not None and restart.step_origin != step_origin:
        msg = 'checkpoint step_origin does not match the explicit schedule'
        raise ValueError(msg)
    if step_times is not None:
        grid = np.asarray(step_times, dtype=float)
        if (
            adaptive
            or grid.ndim != 1
            or len(grid) < 2
            or not np.all(np.isfinite(grid))
            or not np.all(np.diff(grid) > 0)
            or grid[0] != study.t0
            or grid[-1] != study.t1
        ):
            msg = 'step_times must increase from t0 to t1 and cannot combine with adaptive'
            raise ValueError(msg)
    if eec_period is not None:
        if eec_nonlinear_first_order and (
            eec_sign != 1 or integrator.name != 'bdf1' or adaptive or step_times is not None
        ):
            msg = 'nonlinear EEC requires uniform full-period BDF1 without step tables or adaptive controls'
            raise ValueError(msg)
        if (
            isinstance(eec_period, bool)
            or not isinstance(eec_period, Real)
            or not np.isfinite(eec_period)
            or eec_period <= 0
        ):
            msg = 'eec_period requires a positive finite nonboolean interval'
            raise ValueError(msg)
        if (
            not np.isfinite(study.dt)
            or study.dt <= 0
            or not study.t0 + study.dt > study.t0
            or not study.t1 - study.dt < study.t1
            or not study.t0 + eec_period > study.t0
        ):
            msg = 'EEC requires representable advancing time steps and correction intervals'
            raise ValueError(msg)
        if eec_sign == -1 and integrator.name == 'bdf1' and step_times is not None:
            msg = 'anti-periodic EEC requires fixed uniform BDF1 without step tables'
            raise ValueError(msg)
        if integrator.name == 'bdf2' or eec_nonlinear_first_order:
            intervals = (study.t1 - study.t0) / study.dt
            ticks = eec_period / study.dt
            if (
                not np.isfinite(intervals + ticks)
                or ticks < 1
                or not np.isclose(intervals, round(intervals), rtol=0.0, atol=1e-9)
                or not np.isclose(ticks, round(ticks), rtol=0.0, atol=1e-9)
                or (
                    step_times is not None
                    and not np.allclose(np.diff(step_times), study.dt, rtol=0.0, atol=study.dt * 1e-12)
                )
            ):
                msg = (
                    'nonlinear EEC requires a uniform BDF1 grid with positive integer ticks and aligned endpoints'
                    if eec_nonlinear_first_order
                    else 'EEC requires a uniform BDF2 grid with positive integer ticks and aligned final endpoint/correction period'
                )
                raise ValueError(msg)
        if not (eec_affine_first_order or eec_nonlinear_first_order):
            msg = 'TP-EEC requires an explicit assertion of a time-invariant affine first-order operator'
            raise ValueError(msg)
        if adaptive or integrator.name not in ('bdf1', 'bdf2') or not np.isfinite(eec_period) or eec_period <= 0:
            msg = 'eec_period requires a positive period and fixed/table first-order BDF march'
            raise ValueError(msg)
        if owns_history(problem):
            msg = 'TP-EEC cannot correct material history'
            raise ValueError(msg)
    if adaptive and integrator.name != 'bdf2':
        msg = f'adaptive stepping is implemented for BDF2 only, got integrator {integrator.name}'
        raise ValueError(msg)
    u0 = np.asarray(u0, dtype=float)
    if on_checkpoint is not None and checkpoint_key is None:
        msg = 'on_checkpoint requires checkpoint_key'
        raise ValueError(msg)
    if checkpoint_key is not None:
        validate_history_owner(problem)
    state = integrator.start(u0, study.t0, v0, a0)
    if initial_history is not None:
        state = seeded_bdf2_state(state, initial_history, integrator, restart, study.dt)
    if restart is not None:
        restart.validate_restart(checkpoint_key, integrator.name, study.t0, u0.size)
        validate_eec_restart(restart, eec_period, eec_sign, study.dt, eec_nonlinear_first_order)
        restore_history_checkpoint(problem, restart)
        state = IntegratorState(
            us=[np.asarray(u) for u in restart.states],
            t=restart.time,
            v=None if restart.velocity is None else np.asarray(restart.velocity),
            a=None if restart.acceleration is None else np.asarray(restart.acceleration),
        )
        u0 = state.last_u
    dt = study.dt
    lattice = adaptive_lattice(
        study, adaptive, timestep_quantum, timestep_origin, dt if restart is None else restart.previous_dt
    )
    if adaptive:
        # Compare the BDF2 step with the linear predictor from accepted history.
        # For a smooth exact trajectory their difference is
        # 0.5*u''*h*(h+h_prev) + higher orders: O(h²) at comparable spacings,
        # not the BDF2 local truncation error. The square-root controller sizes this
        # curvature indicator; atol is not a certified trajectory-error bound.
        # A rejected step halves and retries; accepted steps retain the existing gain. The
        # event locator's bracket-and-bisect machinery is step-agnostic, so the adaptive
        # steps and the eventful march compose -- the events' own adaptive wiring is a
        # follow-up (the events.py cells are fixed-grid by construction).
        h = dt
        h_prev = dt if restart is None else restart.previous_dt
        h_lo = dt / 1000.0 if h_min is None else h_min
        h_hi = dt * 1000.0 if h_max is None else h_max
        if not np.isfinite(atol) or atol <= 0 or not np.isfinite(h_lo + h_hi) or not 0 < h_lo <= h_hi:
            msg = 'adaptive stepping requires positive finite atol and 0 < h_min <= h_max'
            raise ValueError(msg)
        if lattice is not None:
            if h_hi < lattice.quantum:
                msg = 'adaptive h_max must permit at least one time lattice tick'
                raise ValueError(msg)
            h_lo = max(lattice.quantum, np.ceil(h_lo / lattice.quantum) * lattice.quantum)
            if h_lo > h_hi:
                msg = 'adaptive bounds must contain at least one time lattice step'
                raise ValueError(msg)
        t_end = study.t1
        solution = TransientSolution(times=[study.t0], states=[u0], state_rates=[], rate_state_overrides={})
        while state.t < t_end - 1e-14:
            h = min(h, t_end - state.t)
            t_new = state.t + h
            if lattice is not None:
                current_tick = lattice.tick(state.t)
                count = min(lattice.steps(h), lattice.tick(t_end) - current_tick)
                h = count * lattice.quantum
                t_new = lattice.time(current_tick + count)
                if not t_new > state.t:
                    msg = 'adaptive time lattice cannot make representable clock progress'
                    raise ValueError(msg)
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
                line_search=line_search,
                gmin_stages=gmin_stages,
                held=held,
            )
            # The curvature indicator: |u_new - u_extrap|, with linear continuation from the
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
                if err > atol:
                    solution.indicator_unmet_times.append(t_new)
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
                assert solution.state_rates is not None
                solution.state_rates.append(form.u_dot(u_new))
                if isinstance(problem, HistoryProblem):
                    problem.commit_history(u_new, t_new)
                h_prev = h
                if on_checkpoint is not None:
                    on_checkpoint(_checkpoint_of(problem, state, h_prev, checkpoint_key, integrator.name))
                if (
                    steady_period is not None
                    and not solution.nonconverged_times
                    and periodic_converged(solution.times, solution.states, steady_period, steady_atol, steady_rtol)
                ):
                    break
                if err > 0.0:
                    h = min(h_hi, max(h_lo, h * min(2.0, 1.2 * (atol / err) ** 0.5)))
            else:
                # reject: halve and retry (bounded by h_lo -- at the floor the step is taken
                # with the contract's label rather than spun forever)
                h = max(h_lo, h / 2.0)
        if checkpoint_key is not None:
            solution.checkpoint = _checkpoint_of(problem, state, h_prev, checkpoint_key, integrator.name)
        return solution
    n_steps = _full_fixed_steps(study)
    # A SPAN THAT IS NOT A WHOLE NUMBER OF STEPS ends on one shorter step landing exactly on t1.
    has_remainder = step_times is None and study.t1 - (study.t0 + n_steps * dt) > 1e-12 * max(
        dt, abs(study.t1 - study.t0)
    )
    solution = TransientSolution(times=[study.t0], states=[u0], state_rates=[], rate_state_overrides={})
    if eec_period is not None:
        solution.correction_model = (
            'cycle_integrated_nonlinear_tangent' if eec_nonlinear_first_order else 'affine_cycle_mean'
        )
    saved_eec = None if restart is None else restart.eec_state
    eec_origin = study.t0 if saved_eec is None else saved_eec.origin_time
    target_origin = eec_origin if eec_period is not None else study.t0
    first_tick = round((study.t0 - target_origin) / dt)
    targets = (
        step_times[1:]
        if step_times is not None
        else [target_origin + (first_tick + k + 1) * dt for k in range(n_steps)] + ([study.t1] if has_remainder else [])
    )
    next_correction = (
        (study.t0 + eec_period if saved_eec is None else saved_eec.next_correction_time)
        if eec_period is not None
        else None
    )
    if eec_period is not None:
        count = max(0.0, (study.t1 - next_correction) / eec_period + 1.0)
        if not np.isfinite(count) or count > len(targets) + 1e-9:
            msg = 'TP-EEC correction period requires positive integer steps between boundaries'
            raise ValueError(msg)
        for index in range(int(np.floor(count + 1e-12))):
            boundary = next_correction + index * eec_period
            if not any(abs(t - boundary) <= eec_period * 1e-12 for t in targets):
                msg = 'TP-EEC requires step times at every correction period'
                raise ValueError(msg)
    cycle_start = bdf2_boundary_moment(state.us) if integrator.name == 'bdf2' and len(state.us) >= 2 else u0.copy()
    if saved_eec is not None:
        cycle_start = np.asarray(saved_eec.cycle_start_moment)
    previous_dt = dt if restart is None else restart.previous_dt
    integrated_tangent = None
    cycle_converged = True
    if saved_eec is not None and saved_eec.cycle_tangent is not None:
        integrated_tangent = saved_eec.cycle_tangent.matrix()
        cycle_converged = saved_eec.cycle_converged

    def checkpoint():
        tangent = None
        if eec_nonlinear_first_order:
            tangent = (
                CycleTangentCSR.of(integrated_tangent)
                if integrated_tangent is not None
                else CycleTangentCSR((u0.size, u0.size), (0,) * (u0.size + 1), (), ())
            )
        eec_state = (
            None
            if eec_period is None
            else EECCheckpointState(
                eec_origin,
                next_correction,
                tuple(float(value) for value in cycle_start),
                float(eec_period),
                int(eec_sign),
                bootstrap_pending=integrator.name == 'bdf2' and len(state.us) == 1,
                cycle_tangent=tangent,
                cycle_converged=cycle_converged if eec_nonlinear_first_order else True,
            )
        )
        return _checkpoint_of(
            problem, state, previous_dt, checkpoint_key, integrator.name, eec_state=eec_state, step_origin=step_origin
        )

    for t_new in targets:
        dt = study.dt if step_times is None and not (has_remainder and t_new == targets[-1]) else t_new - state.t
        if integrator.name == 'bdf2' and len(state.us) >= 2 and dt != previous_dt:
            form = _bdf2_variable_form(state, dt, previous_dt, t_new)
        else:
            form = integrator.prepare(state, dt, t_new)
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
            line_search=line_search,
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
        if eec_period is not None and integrator.name == 'bdf2' and len(state.us) == 1:
            cycle_start = 0.5 * (state.last_u + u_new)
        state = integrator.advance(state, u_new, dt, t_new)
        solution.times.append(t_new)
        solution.states.append(u_new)
        assert solution.state_rates is not None
        solution.state_rates.append(form.u_dot(u_new))
        # THE STEP JOINED THE TRAJECTORY, so its material history is the one the next step starts
        # from -- including a step labelled non-converged: the march continues from `u_new`, and a
        # history left at the previous step would be a trajectory the material never followed.
        if isinstance(problem, HistoryProblem):
            problem.commit_history(u_new, t_new)
        previous_dt = dt
        if eec_nonlinear_first_order:
            cycle_converged = cycle_converged and converged
            zero = np.zeros_like(u_new)
            tangent = problem.jacobian(u_new, zero, zero, t_new, DerivativeWeights(alpha0=1.0)) * dt
            integrated_tangent = tangent if integrated_tangent is None else integrated_tangent + tangent
        steady = (
            steady_period is not None
            and not solution.nonconverged_times
            and periodic_converged(solution.times, solution.states, steady_period, steady_atol, steady_rtol)
        )
        if next_correction is not None and abs(t_new - next_correction) <= eec_period * 1e-12:
            boundary = bdf2_boundary_moment(state.us) if integrator.name == 'bdf2' else u_new
            if converged and (not eec_nonlinear_first_order or cycle_converged):
                corrected = (
                    nonlinear_error_correction(problem, boundary, cycle_start, t_new, integrated_tangent, solve=solve)
                    if eec_nonlinear_first_order
                    else explicit_error_correction(
                        problem, boundary, cycle_start, eec_period, t_new, solve=solve, sign=eec_sign
                    )
                )
                # The correction is an algorithmic state jump, not a physical time step.
                # Shift every stored history state equally: preserve BDF2, never bootstrap BE.
                shift = corrected - boundary
                state = IntegratorState(us=[value + shift for value in state.us], t=t_new)
                solution.corrected_times.append(t_new)
                cycle_start = corrected.copy()
            else:
                cycle_start = boundary.copy()
            next_correction = eec_origin + (round((next_correction - eec_origin) / eec_period) + 1) * eec_period
            integrated_tangent, cycle_converged = None, True
        if on_checkpoint is not None:
            on_checkpoint(checkpoint())
        if steady:
            break
    if held is not None:
        logger.info('held-factor march: factorizations=%d solves=%d', held.factorizations, held.solves)
    if checkpoint_key is not None:
        solution.checkpoint = checkpoint()
    return solution
