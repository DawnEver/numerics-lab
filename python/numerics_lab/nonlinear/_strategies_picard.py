"""Picard-family iteration strategies (damped / adaptive / nu-relaxation / plain), split from
`nonlinear/__init__.py` because that module crossed the 800-line band. See its docstring for the
NonlinearSystem/IterationStrategy protocol these implement.
"""

from __future__ import annotations

import numpy as np

from motronics.gauss.nonlinear import (
    _ALPHA_CALM_STEPS,
    _ALPHA_GROW,
    _ALPHA_MIN,
    _RTOL_ENERGY_PCT,
    AssembledState,
    IterationResult,
    NonlinearSystem,
    _energy_error_pct,
    _energy_flux_convergence,
    _has_converged,
    increment_ratio,
)

__all__ = [
    'AdaptiveDampedPicardStrategy',
    'DampedPicardStrategy',
    'NuRelaxationDampedPicardStrategy',
    'PicardStrategy',
]


class DampedPicardStrategy:
    """Successive substitution with adaptive under-relaxation -- the current MEC/FEA production loop.

    Extracted VERBATIM from ``mec/_sparse.py``'s inline loop (bit-for-bit): assemble at the top,
    break on a one-step-delayed convergence flag (so the returned state is the converged iterate's),
    halve ``alpha`` (floored at :data:`_ALPHA_MIN`) whenever the energy error grows, converge when the
    energy change drops below :data:`_RTOL_ENERGY_PCT`, and damp the update ``x += alpha*(solve-x)``.
    A linear system converges after one solve.

    Also ratchets ``alpha`` down (x0.6, floored at :data:`_ALPHA_MIN`) on an
    energy-delta SIGN FLIP -- the same class of damping the FEA frozen-damped loop applies
    (``discretizations/fem/_newton.py``). The ``err > prev_err`` rule above only fires on a
    MONOTONICALLY growing relative error; a period-2 limit cycle (energy alternating between two
    values every step, e.g. a sharp BH knee overshooting each iteration) has a roughly CONSTANT
    relative error, so that rule never trips and alpha stays at 1.0 forever (MANUAL-20260721-414
    follow-up: MEC's Picard loop never converged on the FEMM-matching PCHIP BH spline). The sign-flip
    trigger catches exactly that oscillation.

    Convergence is :func:`_has_converged` -- energy AND iterate, UNCONDITIONALLY. That test used to
    live here, behind ``if ratcheted``, on the reasoning that only a damped iterate can sit on a
    flat-but-wrong energy plateau. Both halves of that were wrong: the plateau is a property of the
    ENERGY CRITERION (so the other eleven strategies needed it too), and gating it on an earlier,
    unrelated damping event meant the strategy that OWNED the protection did not apply it on an
    undamped run -- which its own text conceded was "today's common case". It is now one shared
    function; see its docstring for the measurement.

    ALPHA RECOVERS. The ratchet was one-way, so a single early overshoot damped the entire smooth
    tail: measured on ipm_uv_cummins mec:quad, alpha sat at ``1.0 x 0.5 x 0.6**3 = 0.108`` while the
    energy error read 0.00%, and each rotor position spent 44-61 iterations at ~3 s crawling to the
    increment bar. After :data:`_ALPHA_CALM_STEPS` undisturbed steps alpha grows back by
    :data:`_ALPHA_GROW`, capped at ``alpha_cap`` -- the step size that last misbehaved, never
    retried. This changes the RATE, not the criterion: the fixed point and the test are unchanged.
    """

    name = 'picard-damped'

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        alpha, prev_err = 1.0, np.inf
        prev_delta_sign = 0
        alpha_cap = 1.0  # never recover back to a step size that already misbehaved
        calm = 0  # consecutive steps with neither an error growth nor a sign flip
        ran = 0
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            # BOTH ratchets need the oscillation to be BIGGER THAN THE CONVERGENCE TOLERANCE. Below
            # it the energy is at its own floating-point noise floor, where `delta`'s sign flips at
            # random -- and a sign flip is what the second ratchet reads as a limit cycle. Measured
            # on ipm_uv_cummins mec:quad: the energy error reached 0.00% early, the sign then
            # alternated on noise for the rest of the solve, alpha was driven to 0.108 and pinned
            # there, and each rotor position spent 44-61 iterations crawling to the increment bar at
            # a tenth of the step size. A limit cycle is BY DEFINITION an oscillation that does not
            # decay; one already inside the tolerance is not one.
            meaningful = err >= rtol_energy_pct
            alpha_before = alpha
            disturbed = False
            if meaningful and np.isfinite(prev_err) and err > prev_err:
                alpha = max(_ALPHA_MIN, alpha * 0.5)
                disturbed = True
            if last_energy is not None:
                delta = state.total_energy - last_energy
                sign = (delta > 0) - (delta < 0)
                if meaningful and prev_delta_sign != 0 and sign not in (0, prev_delta_sign):
                    alpha = max(_ALPHA_MIN, alpha * 0.6)
                    disturbed = True
                prev_delta_sign = sign
            if disturbed:
                # ONLY when the shrink actually MOVED alpha. At the floor `max(_ALPHA_MIN, alpha*k)`
                # is a no-op, and recording the cap anyway sets `alpha_cap = _ALPHA_MIN` -- after
                # which the recovery gate `alpha < alpha_cap` reads `0.1 < 0.1` and is false for the
                # rest of the solve. That is the one-way ratchet this recovery leg was ADDED to
                # remove, surviving one level down: measured on dcm_fdm_rect, 557 of 605 iterations
                # pinned at alpha 0.100 with the energy error already at 0.00%, every solve running
                # to its 100-iteration cap because a tenth-sized step cannot reach the increment bar.
                # A step size that never actually applied did not misbehave, so it forbids nothing.
                if alpha < alpha_before:
                    alpha_cap = alpha_before  # the step size that MISBEHAVED, never retried
                calm = 0
            else:
                calm += 1
                if calm >= _ALPHA_CALM_STEPS:
                    if alpha < alpha_cap:
                        alpha = min(alpha_cap, alpha * _ALPHA_GROW)
                    else:
                        # THE CAP ITSELF RELEASES, once alpha has climbed back to it AND the solve
                        # has stayed calm for another full window. Without this the cap is a second
                        # one-way ratchet behind the first: a step size that misbehaved FAR FROM THE
                        # SOLUTION stays forbidden for the whole solve, even after the iterate has
                        # settled and the energy sits at its noise floor. The physics near
                        # convergence is not the physics that misbehaved at iteration 3.
                        #
                        # MEASURED (dcm_fdm_rect) after releasing only the alpha FLOOR: 557
                        # iterations pinned at 0.100 became 447 at 0.108 -- recovery fired and the
                        # sweep still took 658 iterations, because 0.108 was now the ceiling.
                        # `mec-picard-runs-to-its-cap-while-already-converged` (2026-08-26)
                        # predicted it: "the `meaningful` guard prevents NEW reductions; it does not
                        # release an old cap."
                        #
                        # SAFE BECAUSE THE GLOBALISER IS UNCHANGED: any real energy growth or sign
                        # flip above the convergence tolerance re-arms both ratchets on the very
                        # next step, dropping alpha AND resetting the cap to whatever misbehaved.
                        # A limit cycle is still damped; what is no longer damped is the smooth
                        # tail after one early transient.
                        alpha_cap = min(1.0, alpha_cap * _ALPHA_GROW)
                        alpha = min(alpha_cap, alpha * _ALPHA_GROW)
                    calm = 0
            prev_err = err
            x_solve = system.solve_linear(state)
            increment = increment_ratio(x, x_solve)
            converged = _has_converged(err, x, x_solve, rtol_energy_pct)
            last_energy = state.total_energy
            x = x + alpha * (x_solve - x)
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, alpha, increment)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class AdaptiveDampedPicardStrategy:
    """Self-tuning damped Picard -- alpha both shrinks on overshoot AND recovers on progress (sec 1).

    The default :class:`DampedPicardStrategy` only ever RATCHETS alpha DOWN (halve on energy-error
    growth, floored, never raised): safe, but once a hard early step forces alpha small it stays small,
    over-damping the easy tail. This variant self-tunes in BOTH directions -- it still halves alpha when
    the energy error grows (the robust globaliser behaviour), but it also GROWS alpha back toward 1
    (by :data:`_ALPHA_GROW`, capped at 1) on a step that reduced the error, so a case that starts stiff
    and relaxes recovers full Picard steps and finishes in fewer iterations. It converges to the SAME
    fixed point damped-Picard reaches (same map, only the relaxation schedule differs); on a smoothly
    contractive network the two are step-for-step identical (alpha never has to drop, so it never grows).

    Uses ONLY ``assemble`` + ``solve_linear`` (no tangent), so it drops onto BOTH MEC and FEA. This is
    the "expose the FEA inline energy-overshoot alpha-halving as a registry strategy for MEC+FEA"
    deliverable, generalised with the recovery leg. Same delayed-break energy bookkeeping.
    """

    name = 'picard-adaptive'

    def __init__(self, grow: float = _ALPHA_GROW, shrink: float = 0.5) -> None:
        self.grow = grow  # multiplicative alpha recovery on a step that reduced the error
        self.shrink = shrink  # multiplicative alpha cut on a step that grew the error

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        alpha, prev_err = 1.0, np.inf
        ran = 0
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            if np.isfinite(prev_err):
                if err > prev_err:
                    alpha = max(_ALPHA_MIN, alpha * self.shrink)  # overshoot -> damp harder
                elif err < prev_err:
                    alpha = min(1.0, alpha * self.grow)  # progress -> recover toward the full step
            prev_err = err
            x_solve = system.solve_linear(state)
            increment = increment_ratio(x, x_solve)
            converged = _has_converged(err, x, x_solve, rtol_energy_pct)
            last_energy = state.total_energy
            x = x + alpha * (x_solve - x)
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, alpha, increment)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class NuRelaxationDampedPicardStrategy:
    """The FEA reluctivity-under-relaxed Picard loop, MOVED here from the deleted inline assembly loop.

    plan-multi-layer-decoupling P5: the FEA frozen loop (former ``GalerkinSystem.frozen_damped_solve``)
    damped the RELUCTIVITY, not the solution vector -- under-relaxing ``nu`` inside the assembly kernel,
    per element, before the stiffness/source were formed (``nu_used = (1-alpha)*nu_prev +
    alpha*nu(B)``). :class:`DampedPicardStrategy` damps the SOLUTION instead and cannot reproduce that
    iteration map (measured ~2.6e-5 relative on the field, ~5.5e-5 on energy -- close, but a DIFFERENT
    fixed-point trajectory), which is why the frozen loop stayed a separate, inline, kernel-coupled
    code path for as long as it did.

    **The move this class makes possible:** assembly (:meth:`raw_state`) and the reluctivity's per-
    element under-relaxation turn out to be exactly the SAME linear operation reordered. Every raw
    stiffness entry and every PM-remanence source entry in the T3/Q4/VEM kernels is
    ``reluctivity_e * <fixed geometric factor>`` -- i.e. LINEAR in the per-element reluctivity scalar,
    and assembly (summing element contributions into one COO/RHS) is a LINEAR map. So mixing the
    per-element reluctivity BEFORE assembly (what the deleted kernel did) and mixing the assembled
    RAW matrix/RHS AFTER assembly (what this strategy does) are the same operator:

        reluctivity_k = (1-alpha)*reluctivity_(k-1) + alpha*nu_target_k
        ==(linear assembly)==>
        S_k = (1-alpha)*S_(k-1) + alpha*S(nu_target_k),   T_k = (1-alpha)*T_(k-1) + alpha*T(nu_target_k)

    where ``S_(k-1)``/``T_(k-1)`` are literally last iteration's already-mixed raw arrays (no per-
    element state needs to cross the kernel boundary -- only the previous raw ``(s_val, T_mat)`` pair).
    A conductor-current source term (nu-independent, identical every iteration) mixes back to itself
    trivially since the mixing weights ``(1-alpha, alpha)`` always sum to 1.

    **This identity REQUIRES alpha to be a single GLOBAL scalar, uniform across every element**, exactly
    as the deleted loop always used it: ``(1-alpha)*S(w) != S((1-alpha)*w)`` in general, but IS an
    identity when the SAME scalar ``1-alpha`` multiplies the whole raw array (which is what a per-
    element-uniform alpha guarantees by construction). A future optimisation that lets alpha vary PER
    ELEMENT would silently invalidate this equivalence with nothing else going red -- there is
    deliberately no runtime assertion here (alpha is a private ``float`` local, never a per-element
    array, so nothing today CAN pass a non-uniform alpha through this seam); this paragraph is the
    guard until a per-element-alpha strategy is actually proposed.

    Not bit-for-bit with the deleted loop (floating-point reassociation: ``(1-a)*x+a*y`` computed
    before vs after multiplying by the geometric factor rounds differently), but the SAME operator in
    exact arithmetic -- goldens are verified within their existing numerical tolerance, not by hash.

    Duck-types on ``raw_state``/``fold_raw`` (present only on ``GalerkinSystem``) rather than importing
    the FEA module, keeping this module solver-agnostic -- the same seam style
    :class:`DampedPicardStrategy` uses via the generic ``assemble``/``solve_linear``, one level lower
    (raw pre-BC-fold arrays instead of a folded :class:`AssembledState`) because the mixing has to
    happen BEFORE the boundary fold to match the deleted loop's order of operations exactly.
    """

    name = 'nu-damped'

    #: WHAT THIS STRATEGY NEEDS OF A SYSTEM, as data. The requirement was enforced in `solve` and
    #: stated in three docstrings, and nothing could ASK it -- so `test_iteration_method_matrix`
    #: generated a `nu-damped` cell for the MEC `_ReluctanceSystem`, which cannot supply either
    #: attribute, and read the designed refusal as a red (heavy shard 1, 2026-08-29). A boundary
    #: only the failure announces is a boundary no caller can plan around.
    requires = ('raw_state', 'fold_raw')

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        raw_state = getattr(system, 'raw_state', None)
        fold_raw = getattr(system, 'fold_raw', None)
        if raw_state is None or fold_raw is None:
            # THE WRAPPER IS TRANSPARENT, so naming it is naming the wrong object. A budgeted
            # system delegates every attribute it does not define, including a MISSING one -- so
            # `_TimeBudgetedSystem` appearing here says only that a budget was set, while the
            # system that actually lacks the two attributes is inside it. Measured 2026-08-29:
            # the message read "got _TimeBudgetedSystem" and sent the reader to the budget wrapper
            # introduced in 55e8fd37c, which is not the reason this combination cannot run.
            inner = getattr(system, 'unwrapped', system)
            named = type(inner).__name__
            if inner is not system:
                named = f'{named} (inside {type(system).__name__})'
            msg = f"'nu-damped' requires a system exposing raw_state/fold_raw; got {named}"
            raise TypeError(msg)
        x = system.initial_guess()
        alpha = 1.0
        alpha_min = 0.05  # frozen floor for the reluctivity under-relaxation factor
        prev_delta_sign = 0
        last_energy: float | None = None
        last_flux: float | None = None
        prev_s_val = None  # last iteration's MIXED raw stiffness values (None = no prior iterate yet)
        prev_t_mat = None  # last iteration's MIXED raw source vector
        state: AssembledState | None = None
        is_convergence = False  # one-step-delayed break: the pass AFTER convergence only refreshes state
        reached_tol = False
        ran = 0
        for i in range(max_iter):
            s_row, s_col, s_val, t_mat, total_energy = raw_state(x)
            if prev_s_val is None:
                mixed_s_val, mixed_t_mat = s_val, t_mat  # first pass: nu_prev<0 -> reluctivity=nu_target
            else:
                mixed_s_val = (1.0 - alpha) * prev_s_val + alpha * s_val
                mixed_t_mat = (1.0 - alpha) * prev_t_mat + alpha * t_mat
            state = fold_raw(s_row, s_col, mixed_s_val, mixed_t_mat, total_energy)
            if is_convergence:
                break  # refresh-only pass: state now reflects the converged iterate; stop without solving
            total_flux = float(np.max(np.abs(x)))
            err, _flux_err, is_converged = _energy_flux_convergence(
                total_energy, total_flux, last_energy, last_flux, rtol_energy_pct, rtol_energy_pct
            )
            if is_converged:
                is_convergence = True
                reached_tol = True
            # Frozen alpha adaptation: halve (x0.6, floored) on an energy-delta SIGN FLIP -- verbatim
            # from the deleted loop, never raised back up.
            if last_energy is not None:
                delta = total_energy - last_energy
                sign = (delta > 0) - (delta < 0)
                if prev_delta_sign != 0 and sign not in (0, prev_delta_sign):
                    alpha = max(alpha * 0.6, alpha_min)
                prev_delta_sign = sign
            last_energy, last_flux = total_energy, total_flux
            prev_s_val, prev_t_mat = mixed_s_val, mixed_t_mat
            x_prev = x
            x = system.solve_linear(state)
            # This strategy judges on the energy/flux PAIR, not on `_has_converged`, so it has no
            # increment of its own; it is measured here purely so the observer sees the same two
            # numbers everywhere and a capped run is readable without knowing which strategy ran.
            increment = increment_ratio(x_prev, x)
            ran = i + 1
            if system.is_linear:
                is_convergence = True
            system.on_iteration(i, state, err, alpha, increment)
        converged = bool(reached_tol or system.is_linear)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class PicardStrategy:
    """Plain successive substitution -- NO damping (``x_{k+1} = solve``).

    The naive baseline that historically diverged/oscillated on strongly-saturating B-H (which is
    exactly WHY the damped variant exists). First principle: KEEP it, so its failure mode is a
    characterized, labelled datum (a convergence-matrix ``xfail``), not a silent exclusion. Same
    assemble-at-top / delayed-break convergence bookkeeping as the damped variant, minus the alpha.
    """

    name = 'picard'

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        ran = 0
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            x_solve = system.solve_linear(state)
            increment = increment_ratio(x, x_solve)
            converged = _has_converged(err, x, x_solve, rtol_energy_pct)
            last_energy = state.total_energy
            x = x_solve  # undamped: take the full substitution
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0, increment)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)
