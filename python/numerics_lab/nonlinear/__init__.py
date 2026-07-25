"""Pluggable nonlinear-iteration strategies, decoupled from the governing equation.

Design ``.claude/memory/2026/07/09/design-solver-iteration-decoupling.md`` (a179884): the
governing equation knows how to ASSEMBLE (given the current iterate, produce the system matrix,
the energy, and whatever payload the caller returns); the STRATEGY knows how to ITERATE (Picard,
damped-Picard, Newton). They meet at the small :class:`NonlinearSystem` protocol, so a solver
picks its nonlinear method without the equation code changing.

First principle (``AGENTS.md``): IMPLEMENT every method and characterize it -- never exclude one
because an experiment showed it inferior. Plain Picard's historical divergence on saturating B-H
is a labelled datum, not a hidden exclusion. The default is the damped-Picard extracted VERBATIM
from the MEC/FEA inline loops, so wiring this seam leaves every golden bit-for-bit.

This module drives the iterate only; it never assembles physics or touches a mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = [
    'ITERATION_REGISTRY',
    'AdaptiveDampedPicardStrategy',
    'AndersonStrategy',
    'AssembledState',
    'BFGSStrategy',
    'BroydenStrategy',
    'ContinuationStrategy',
    'DampedPicardStrategy',
    'FrozenDampedStrategy',
    'IterationResult',
    'IterationStrategy',
    'LBFGSStrategy',
    'ModifiedNewtonStrategy',
    'NewtonKrylovStrategy',
    'NewtonLineSearchStrategy',
    'NewtonStrategy',
    'NonlinearSystem',
    'PicardStrategy',
    'get_strategy',
]

# Damped-Picard parameters (moved verbatim from mec/_sparse.py so the extracted loop is identical).
_ALPHA_MIN = 0.1  # floor for the adaptive under-relaxation factor
_ALPHA_GROW = 1.5  # adaptive-Picard recovery factor: grow alpha back toward 1 on a progressing step
_RTOL_ENERGY_PCT = 0.01  # convergence: total-energy change below this fraction of a percent


@dataclass
class AssembledState:
    """One assembly of the governing equation at the current iterate.

    Attributes:
        system_matrix: the (reduced) linear system a strategy solves for the next iterate.
        total_energy: the scalar the convergence test tracks.
        payload: equation-specific extras the CALLER reads back off the final state (e.g. the
            per-branch energy/flux/B for MEC); the strategy never inspects it.

    """

    system_matrix: object
    total_energy: float
    payload: object = None


@dataclass
class IterationResult:
    """What a strategy returns: the converged iterate, the final assembly, and run metadata."""

    x: np.ndarray
    state: AssembledState | None
    iterations: int
    converged: bool


class NonlinearSystem(Protocol):
    """The seam an equation implements so a strategy can drive it (see module docstring).

    ``assemble`` / ``solve_linear`` are the Picard interface (fixed-point substitution
    ``x <- solve_linear(assemble(x))``). ``residual`` / ``tangent`` / ``solve_system`` are the
    Newton interface (``tangent @ dx = -residual``); ``tangent`` may return ``None`` for a combo
    that cannot supply a Jacobian, and NewtonStrategy falls back to a Picard step.
    """

    is_linear: bool

    # Every parameter below is positional-only (``/``): every real caller (the strategies in this
    # module) invokes them positionally, and the two implementers name their iterate differently
    # (MEC's ``phi_loop``, FEA's ``A_mat``) because it carries different physics -- a loop flux vs.
    # a nodal vector potential. Requiring a shared PARAMETER NAME here would be a "declaration that
    # lies" for whichever implementer's name is more honest; positional-only makes the Protocol
    # match how it is actually consumed instead (found via pyright while composing the P3 seam --
    # `gauss/evaluation.py` -- surfaced this pre-existing mismatch on both `_ReluctanceSystem` and
    # `GalerkinSystem`).
    def initial_guess(self) -> np.ndarray: ...
    def assemble(self, x: np.ndarray, /) -> AssembledState: ...
    def solve_linear(self, state: AssembledState, /) -> np.ndarray: ...

    def residual(self, x: np.ndarray, state: AssembledState, /) -> np.ndarray:
        """Residual r(x) whose root is the solution (Newton drives ``tangent @ dx = -r``)."""

    def residual_norm(self, x: np.ndarray, state: AssembledState, /) -> float:
        """Scalar convergence measure a line search minimises -- the norm over the FREE DOFs only.

        Defaults to ``||residual||`` (correct for MEC, whose reduced loop residual has no pinned rows).
        FEA overrides it to EXCLUDE the Dirichlet rows: the full-space FEA residual ``S A - T`` carries a
        large, constant residual on the (un-pinned in the raw operator) Dirichlet nodes that the
        increment (``dA = 0`` there) can never reduce, so an unfiltered ``||r||`` would make a
        backtracking line search reject every step. Only the free-DOF residual actually drops to zero.
        """
        return float(np.linalg.norm(self.residual(x, state)))

    def tangent(self, x: np.ndarray, state: AssembledState, /) -> object | None:
        """Optional Newton tangent operator; return ``None`` to fall back to a Picard step.

        The return type is deliberately opaque (``object``, not e.g. a sparse matrix): MEC returns
        a raw ``scipy.sparse`` matrix, FEA a reduced-tangent bundle (``dict``) that only its own
        ``solve_system`` understands -- the caller (a strategy) passes it back verbatim, never
        inspects its shape. An unannotated stub here previously let pyright infer ``-> None`` from
        the docstring-only body, which is exactly the "declaration that lies" AGENTS.md names: it
        made every REAL implementation (returning a matrix/dict) structurally incompatible with the
        Protocol it exists to satisfy.
        """

    def solve_system(self, matrix, rhs: np.ndarray, /) -> np.ndarray:
        """Solve ``matrix @ y = rhs`` with the equation's own singular-system guard.

        ``matrix`` is whatever :meth:`tangent` returned for this equation (a raw sparse matrix for
        MEC, a reduced-tangent bundle for FEA); the strategy passes it back verbatim, so only Newton-
        family strategies that obtained it from :meth:`tangent` call this. Quasi-Newton strategies that
        need a SECANT solve of an arbitrary rhs use :meth:`solve_secant` instead (which every equation
        implements uniformly), because the secant ``system_matrix`` is not always in ``solve_system``'s
        accepted form (FEA's ``solve_system`` takes the reduced-tangent bundle, not a raw matrix).
        """

    def solve_secant(self, state: AssembledState, rhs: np.ndarray, /) -> np.ndarray:
        """Solve the SECANT system ``S @ y = rhs`` for an arbitrary rhs, with BC reduction folded in.

        The uniform ``H0`` action the quasi-Newton family (Broyden / BFGS / L-BFGS) needs: ``S`` is the
        assembled secant operator at the current iterate, reduced through the equation's own Dirichlet /
        periodic / hanging machinery so ``y`` lives in the full DOF space. ``solve_linear`` is the
        special case ``rhs = source``. Default: ``solve_system(state.system_matrix, rhs)`` (correct for
        MEC, whose ``solve_system`` accepts a raw matrix); FEA overrides it to reduce ``rhs`` like a
        source and solve the reduced secant.
        """
        return self.solve_system(state.system_matrix, rhs)

    def on_iteration(self, i: int, state: AssembledState, err: float, alpha: float, /) -> None:
        """Optional per-iteration hook (logging); default no-op."""

    def scale_source(self, lam: float, /) -> NonlinearSystem:
        """Return a copy of this system with its SOURCE scaled by ``lam`` (continuation stepping).

        Optional -- only the ``continuation`` strategy calls it. An equation that cannot cheaply
        rescale its source (conductor current + magnet Br) raises :class:`NotImplementedError`, and
        the strategy degrades to solving the full-source problem directly. ``lam=1`` returns the
        original (full-source) system.
        """


class IterationStrategy(Protocol):
    name: str

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult: ...


def _energy_error_pct(total_energy: float, last_energy: float | None) -> float:
    if last_energy is None:
        return np.inf
    denom = max(abs(total_energy), abs(last_energy), 1e-12)
    return abs(total_energy - last_energy) / denom * 100.0


class DampedPicardStrategy:
    """Successive substitution with adaptive under-relaxation -- the current MEC/FEA production loop.

    Extracted VERBATIM from ``mec/_sparse.py``'s inline loop (bit-for-bit): assemble at the top,
    break on a one-step-delayed convergence flag (so the returned state is the converged iterate's),
    halve ``alpha`` (floored at :data:`_ALPHA_MIN`) whenever the energy error grows, converge when the
    energy change drops below :data:`_RTOL_ENERGY_PCT`, and damp the update ``x += alpha*(solve-x)``.
    A linear system converges after one solve.

    Also ratchets ``alpha`` down (x0.6, floored at :data:`_ALPHA_MIN`, never raised again) on an
    energy-delta SIGN FLIP -- the same class of damping the FEA frozen-damped loop applies
    (``discretizations/fem/_newton.py``). The ``err > prev_err`` rule above only fires on a
    MONOTONICALLY growing relative error; a period-2 limit cycle (energy alternating between two
    values every step, e.g. a sharp BH knee overshooting each iteration) has a roughly CONSTANT
    relative error, so that rule never trips and alpha stays at 1.0 forever (MANUAL-20260721-414
    follow-up: MEC's Picard loop never converged on the FEMM-matching PCHIP BH spline). The sign-flip
    trigger catches exactly that oscillation.

    Once alpha has EVER been ratcheted below 1.0, the energy criterion alone is no longer trusted:
    a heavily-damped iterate can sit on a near-flat plateau of ``total_energy`` that is NOT the true
    Picard fixed point (measured: a saturating-BH case read 0.00% energy change for 20+ iterations
    at a spurious energy level ~14x the physical one, only escaping and reaching the genuine
    fixed point ~15 iterations later when left to run). So once damped, convergence ALSO requires
    the RAW (undamped) Picard increment ``x_solve - x`` to be small -- the direct fixed-point test
    ``x_solve == x``, immune to a flat-but-wrong energy plateau. A run that never ratchets (today's
    common case) never evaluates this extra test: bit-for-bit with the pre-ratchet loop.
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
        ratcheted = False
        ran = 0
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            if np.isfinite(prev_err) and err > prev_err:
                alpha = max(_ALPHA_MIN, alpha * 0.5)
                ratcheted = True
            if last_energy is not None:
                delta = state.total_energy - last_energy
                sign = (delta > 0) - (delta < 0)
                if prev_delta_sign != 0 and sign not in (0, prev_delta_sign):
                    alpha = max(_ALPHA_MIN, alpha * 0.6)
                    ratcheted = True
                prev_delta_sign = sign
            prev_err = err
            x_solve = system.solve_linear(state)
            if err < rtol_energy_pct:
                if not ratcheted:
                    converged = True
                else:
                    x_scale = max(float(np.linalg.norm(x)), 1e-12)
                    increment_rel = float(np.linalg.norm(x_solve - x)) / x_scale
                    if increment_rel < rtol_energy_pct / 100.0:
                        converged = True
            last_energy = state.total_energy
            x = x + alpha * (x_solve - x)
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, alpha)
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
            if err < rtol_energy_pct:
                converged = True
            last_energy = state.total_energy
            x_solve = system.solve_linear(state)
            x = x + alpha * (x_solve - x)
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, alpha)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class FrozenDampedStrategy:
    """Registry seam for the frozen FEA ``GalerkinSystem.frozen_damped_solve`` loop (WP1 contract 343).

    ``fea_newton_solve`` (``maxwell/fea/_newton.py``) still fast-paths ``None``/``'picard-damped'``
    straight onto ``system.frozen_damped_solve()`` -- that inline reluctivity-under-relaxed loop is
    frozen and differs from :class:`DampedPicardStrategy` in exactly the ways documented on
    ``frozen_damped_solve`` (damping lives in the assembly kernel's ``nu_relax``, not the solution
    vector). This strategy widens the SEAM so that fast path is also reachable BY NAME through
    :data:`ITERATION_REGISTRY`/:func:`get_strategy`, for callers that want to dispatch generically
    rather than special-case the FEA default. It adapts the frozen solve's raw
    ``(S_mat, A_mat, T_mat, B_mat, B_norm_mat, Energy_mat)`` tuple to the uniform
    :class:`IterationResult`, duck-typing on ``system.frozen_damped_solve`` (present only on
    ``GalerkinSystem``) rather than importing the FEA module (keeps this module solver-agnostic).
    """

    name = 'frozen-damped'

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        frozen_solve = getattr(system, 'frozen_damped_solve', None)
        if frozen_solve is None:
            msg = f"'frozen-damped' requires a system exposing frozen_damped_solve; got {type(system).__name__}"
            raise TypeError(msg)
        _s_mat, a_mat, t_mat, _b_mat, _b_norm_mat, _energy_mat = frozen_solve()
        state = AssembledState(system_matrix=_s_mat, total_energy=float(_energy_mat.sum()), payload={'T_mat': t_mat})
        # The frozen loop returns fields only, so it PUBLISHES its own convergence bookkeeping on the
        # system (``last_iterations`` / ``last_converged``). This used to report
        # ``iterations=max_iter, converged=True`` unconditionally -- two fabricated numbers, which is
        # why accuracy._ITERATION_BANDS['frozen-damped'] (level='validated', measured=3 iters) could
        # not be gated by anything: the count it claimed was never the count that ran, and a
        # non-converged solve still reported success (R1.4, 2026-07-25).
        iterations = getattr(system, 'last_iterations', None)
        converged = getattr(system, 'last_converged', None)
        if iterations is None or converged is None:
            msg = (
                f"'frozen-damped' requires the system to publish last_iterations/last_converged after "
                f'frozen_damped_solve; {type(system).__name__} did not'
            )
            raise TypeError(msg)
        return IterationResult(x=a_mat, state=state, iterations=int(iterations), converged=bool(converged))


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
            if err < rtol_energy_pct:
                converged = True
            last_energy = state.total_energy
            x = system.solve_linear(state)  # undamped: take the full substitution
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class NewtonStrategy:
    """Newton-Raphson: ``J(x_k) dx = -r(x_k); x_{k+1} = x_k + dx``.

    Uses the equation's ``tangent`` (secant + differential ``dnu`` term); at ``x=0`` the field is
    zero so ``dnu=0`` and the first Newton step equals the first Picard iterate, then the true
    Jacobian gives quadratic convergence near the solution (Picard is linear and needs damping on
    saturating B-H). Falls back to a Picard step whenever ``tangent`` returns ``None`` (a combo with
    no Jacobian), so wiring Newton never breaks a solver that cannot supply one. Same assemble-at-top
    / one-step-delayed-break / energy-change convergence bookkeeping as the Picard strategies.
    """

    name = 'newton'

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
            if err < rtol_energy_pct:
                converged = True
            last_energy = state.total_energy
            jac = system.tangent(x, state)
            if jac is None:
                x = system.solve_linear(state)  # no Jacobian available -> Picard substitution
            else:
                dx = system.solve_system(jac, -system.residual(x, state))
                x = x + dx
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class AndersonStrategy:
    """Anderson acceleration of the Picard fixed-point map (roadmap sec 1).

    The Picard map is ``g(x) = solve_linear(assemble(x))`` and its fixed-point residual is
    ``f(x) = g(x) - x``. Plain/damped Picard mixes only the LAST iterate; Anderson mixes the last
    ``m`` iterates by the least-squares combination of their residuals that minimises the mixed
    residual, giving multi-secant (quasi-Newton-like) acceleration while using ONLY ``assemble`` +
    ``solve_linear`` -- so it drops onto BOTH MEC and FEA with no tangent. Converges to the SAME
    fixed point damped-Picard reaches, in fewer iterations on a contractive map.

    Bookkeeping matches the Picard strategies (assemble-at-top, one-step-delayed break on the
    energy-change test) so the returned state is the converged iterate's and the reference-energy
    gate holds. ``beta`` is the mixing (relaxation) factor; ``m`` the history depth.
    """

    name = 'anderson'

    def __init__(self, m: int = 5, beta: float = 1.0) -> None:
        self.m = m
        self.beta = beta

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        ran = 0
        x_hist: list[np.ndarray] = []  # past g(x_k) images
        f_hist: list[np.ndarray] = []  # past residuals f_k = g(x_k) - x_k
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            if err < rtol_energy_pct:
                converged = True
            last_energy = state.total_energy
            gx = system.solve_linear(state)  # one Picard image g(x_k)
            f = gx - x  # fixed-point residual
            x_hist.append(gx)
            f_hist.append(f)
            if len(f_hist) > self.m + 1:
                x_hist.pop(0)
                f_hist.pop(0)
            if system.is_linear or len(f_hist) == 1:
                # First step (no history to mix) or a linear system (one solve is exact): plain
                # substitution, identical first iterate to damped-Picard at alpha=1.
                x = gx
            else:
                # Least-squares Anderson mix: minimise || f_k - dF @ gamma || over the m residual
                # differences, then x <- (X_k @ w) with the relaxed combination of the g-images.
                dF = np.column_stack([f_hist[j + 1] - f_hist[j] for j in range(len(f_hist) - 1)])
                dX = np.column_stack([x_hist[j + 1] - x_hist[j] for j in range(len(x_hist) - 1)])
                gamma, *_ = np.linalg.lstsq(dF, f, rcond=None)
                x = gx - dX @ gamma
                if self.beta != 1.0:
                    x = x + (self.beta - 1.0) * (f - dF @ gamma)
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, self.beta)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


def _lu_factorize(matrix):
    """LU-factorize a scipy-sparse ``matrix`` for repeated solves, or ``None`` if it is not one.

    Modified-Newton (and a modified-Picard) hold the SAME system matrix across several inner solves;
    a single ``splu`` factor reused for each rhs gives ``x`` BIT-FOR-BIT identical to re-running
    ``spsolve`` per iteration (scipy's ``spsolve`` for a 1-D rhs IS ``splu(A).solve(b)`` with the same
    default column permutation), while paying the O(nnz^1.x) factorization ONCE instead of every step.

    Returns ``None`` for a non-scipy tangent (e.g. FEA's opaque reduced-increment bundle), so those
    equations transparently fall back to the equation's own ``solve_system``.
    """
    # Local import (matches NewtonKrylovStrategy): keep scipy off this module's hot default (Picard)
    # import path; only the factorization-reusing strategies pull it in.
    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.sparse.linalg import splu  # noqa: PLC0415

    if not sp.issparse(matrix):
        return None
    return splu(sp.csc_matrix(matrix))


class ModifiedNewtonStrategy:
    """Newton reusing the cached tangent -- AND its factorization -- for ``k`` steps (roadmap sec 1).

    A full Newton re-forms and re-factorizes ``tangent`` every step; the modified Newton holds the
    Jacobian fixed for ``k`` residual/solve steps between refreshes, trading quadratic convergence for
    a much cheaper step (one factorization amortized over ``k`` solves). Because the held matrix is
    IDENTICAL across those steps, its LU factor is computed ONCE at each refresh and reused
    (:func:`_lu_factorize`) -- the actual cost saving, and bit-for-bit the same iterate the per-step
    ``spsolve`` produced. Uses only ``tangent`` / ``residual`` / ``solve_system`` (the Newton
    interface); falls back to a Picard step when ``tangent`` returns ``None`` and to the equation's own
    ``solve_system`` for a non-scipy tangent (FEA's opaque bundle). Converges to the same root as
    Newton; same delayed-break energy bookkeeping.

    MANUAL-20260721-419: a HELD tangent step is only amortized cheaply while it is still a good local
    model. On the sharper-curvature FEMM-matching BH spline the tangent goes stale mid-hold and the
    step over-shoots into a spurious root -- a near-zero field on the sdm map, an outright energy
    blow-up on MEC/FEA (``newton`` / ``newton-linesearch``, which refresh every step, land the correct
    root on the same maps). So every HELD (non-refresh) step is checked against the RAW (unfiltered)
    ``residual`` norm before/after: if the step would GROW ``||residual||``, the held tangent is stale
    for the current curvature -- refresh it right there and retake the step with the fresh one,
    instead of marching toward the spurious root.

    Deliberately the raw residual, not :meth:`NonlinearSystem.residual_norm`: that hook is
    scale-RELATIVE (``||g(x)-x|| / ||x||``), which is exactly blind to this failure mode -- once a
    stale step has already pushed ``x`` far outside the physical range, ``g(x)`` saturates to a value
    that no longer scales with ``x``, so the RATIO plateaus near a fixed asymptote even as the
    ABSOLUTE residual (and ``x`` itself) keeps growing without bound (measured: a stale sdm step took
    ``||x||`` 1.6e3 -> 1.4e8 while the relative residual read a near-constant ~1.0 throughout). The raw
    norm has no such blind spot: it grows in lockstep with the divergence. A held step that genuinely
    shrinks the residual (every low-curvature case, and the common case generally) never trips this
    guard, so the amortization -- and its bit-for-bit iterate -- is unchanged.
    """

    name = 'modified-newton'

    def __init__(self, refresh_every: int = 3) -> None:
        self.refresh_every = max(1, refresh_every)

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        ran = 0
        cached_jac = None
        cached_lu = None  # splu factor of cached_jac, reused across the held (non-refresh) steps
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            if err < rtol_energy_pct:
                converged = True
            last_energy = state.total_energy
            refresh = i % self.refresh_every == 0 or cached_jac is None
            if refresh:
                cached_jac = system.tangent(x, state)  # refresh (re-assemble)
                cached_lu = _lu_factorize(cached_jac)  # re-factorize ONCE; reused below until next refresh
            if cached_jac is None:
                x = system.solve_linear(state)  # no Jacobian -> Picard substitution
            else:
                r = np.asarray(system.residual(x, state), dtype=float)
                # Reuse the cached factorization: bit-for-bit the spsolve step, without re-factorizing.
                dx = cached_lu.solve(-r) if cached_lu is not None else system.solve_system(cached_jac, -r)
                if refresh:
                    x = x + dx  # freshly-refreshed tangent: trusted, no staleness check needed
                else:
                    # Staleness guard (MANUAL-20260721-419): a HELD step must not GROW the RAW
                    # (unfiltered) residual norm -- see the class docstring for why the scale-relative
                    # residual_norm hook is blind to this. It never grows on a low-curvature hold, so
                    # this branch is a no-op there and the iterate stays bit-for-bit the pre-guard step.
                    r0 = float(np.linalg.norm(r))
                    trial = x + dx
                    r1 = float(np.linalg.norm(np.asarray(system.residual(trial, system.assemble(trial)), dtype=float)))
                    if np.isfinite(r1) and r1 <= r0:
                        x = trial
                    else:
                        # Stale for the current curvature: refresh right here and retake the step with
                        # the fresh tangent, instead of marching toward the spurious root.
                        cached_jac = system.tangent(x, state)
                        cached_lu = _lu_factorize(cached_jac)
                        if cached_jac is None:
                            x = system.solve_linear(state)
                        else:
                            dx = cached_lu.solve(-r) if cached_lu is not None else system.solve_system(cached_jac, -r)
                            x = x + dx
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class BroydenStrategy:
    """Quasi-Newton (Broyden good) -- an approximate inverse Jacobian from residual history (sec 1).

    Builds the Newton direction from a low-rank secant approximation of the inverse Jacobian, updated
    each step from the ``(dx, df)`` pair (``df = r(x_{k+1}) - r(x_k)``) -- so it needs only
    ``residual`` and an initial linear apply, positioning between Picard and Newton where the exact
    ``tangent`` is unavailable (returns ``None``). The initial inverse-Jacobian action is one solve of
    the secant ``system_matrix`` (available on every equation); subsequent steps apply the accumulated
    rank-1 Sherman-Morrison updates with no further factorization. Converges to the same root; same
    delayed-break energy bookkeeping.
    """

    name = 'quasi-newton'

    def __init__(self, m: int = 10) -> None:
        self.m = m  # cap the stored rank-1 update pairs

    def _apply_inverse(self, system, state, r, updates):
        # H0^{-1} r via one secant solve, then the stored rank-1 Broyden updates.
        v = system.solve_secant(state, r)
        for s_vec, denom, u in updates:
            v = v + (s_vec @ v) / denom * u
        return v

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        ran = 0
        updates: list[tuple] = []  # (s_row_vector, denom, u_column) rank-1 inverse-Jacobian updates
        prev_r = None
        prev_dx = None
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            if err < rtol_energy_pct:
                converged = True
            last_energy = state.total_energy
            r = system.residual(x, state)
            if prev_r is not None:
                # Broyden "good" inverse update: H <- H + ((dx - H df) / (dx.H df)) dx.H, applied as a
                # stored rank-1 pair so future applies stay matrix-free.
                df = r - prev_r
                h_df = self._apply_inverse(system, state, df, updates)
                denom = float(prev_dx @ h_df)
                if abs(denom) > 1e-30:
                    u = (prev_dx - h_df) / denom
                    updates.append((prev_dx.copy(), 1.0, u))
                    if len(updates) > self.m:
                        updates.pop(0)
            dx = -self._apply_inverse(system, state, r, updates)
            x = x + dx
            prev_r, prev_dx = r, dx
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class BFGSStrategy:
    """Quasi-Newton (full BFGS) -- a SYMMETRIC low-rank inverse-Jacobian from secant pairs (sec 1).

    Broyden's update is the general (unsymmetric) secant; BFGS is its symmetric-positive-definite
    sibling, the right choice for the SPD-ish magnetostatic root ``r(x) = S(x) x - b`` whose true
    tangent is symmetric. It maintains ``H ~ J^{-1}`` by the classic rank-2 BFGS inverse update from
    each ``(s = dx, y = r_{k+1} - r_k)`` pair, applied matrix-free through the stored history on top of
    an initial ``H0`` action = one secant solve of ``system_matrix`` (the same seed Broyden uses). Needs
    only ``residual`` + that one linear apply -- no exact ``tangent``. Converges to the same root; same
    delayed-break energy bookkeeping. ``m`` caps the stored pairs (full BFGS = the whole history).

    The BFGS inverse-Hessian action is computed by the L-BFGS two-loop recursion over the retained
    pairs (mathematically identical to the dense ``H <- (I - rho s y^T) H (I - rho y s^T) + rho s s^T``
    update, applied matrix-free so no dense ``H`` is ever formed); the limited-memory sibling
    :class:`LBFGSStrategy` uses the SAME recursion but caps the history much shorter.
    """

    name = 'bfgs'

    def __init__(self, m: int = 20) -> None:
        self.m = m  # cap the stored (s, y, rho) secant pairs (full BFGS keeps the whole history)

    def _apply_inverse(self, system, state, r, pairs):
        # H r via the L-BFGS two-loop recursion over the stored pairs, H0 r = one secant solve.
        q = r.copy()
        alphas = []
        for s_vec, y_vec, rho in reversed(pairs):
            a = rho * float(s_vec @ q)
            q = q - a * y_vec
            alphas.append(a)
        z = system.solve_secant(state, q)
        for (s_vec, y_vec, rho), a in zip(pairs, reversed(alphas), strict=True):
            b = rho * float(y_vec @ z)
            z = z + (a - b) * s_vec
        return z

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        ran = 0
        pairs: list[tuple] = []  # (s = dx, y = dr, rho = 1/(y.s)) BFGS secant pairs
        prev_r = None
        prev_dx = None
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            if err < rtol_energy_pct:
                converged = True
            last_energy = state.total_energy
            r = system.residual(x, state)
            if prev_r is not None:
                y = r - prev_r
                ys = float(y @ prev_dx)
                if abs(ys) > 1e-30:  # curvature condition; skip a non-informative (near-zero) pair
                    pairs.append((prev_dx.copy(), y, 1.0 / ys))
                    if len(pairs) > self.m:
                        pairs.pop(0)
            dx = -self._apply_inverse(system, state, r, pairs)
            x = x + dx
            prev_r, prev_dx = r, dx
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class LBFGSStrategy:
    """Quasi-Newton (limited-memory BFGS) -- the two-loop recursion on a bounded secant history (sec 1).

    Identical inverse-Hessian model to :class:`BFGSStrategy` but capping the stored history to the last
    ``m`` ``(s, y)`` pairs, so the per-step cost and memory stay O(m*n) regardless of iteration count --
    the form used for large systems where keeping the full history is prohibitive. The initial ``H0``
    action is one secant ``system_matrix`` solve (the same seed the other quasi-Newtons use), so it
    needs only ``residual`` + that one linear apply and no exact ``tangent``. Converges to the same root
    as full BFGS / Newton on the SPD-ish magnetostatic problem; same delayed-break energy bookkeeping.
    """

    name = 'l-bfgs'

    def __init__(self, m: int = 8) -> None:
        self.m = m  # limited-memory depth (the last m secant pairs)

    def _two_loop(self, system, state, r, pairs):
        # Standard L-BFGS two-loop recursion: backward pass accumulates alphas, H0 apply, forward pass.
        q = r.copy()
        alphas = []
        for s_vec, y_vec, rho in reversed(pairs):
            a = rho * float(s_vec @ q)
            q = q - a * y_vec
            alphas.append(a)
        z = system.solve_secant(state, q)  # H0 z = q (one secant solve)
        for (s_vec, y_vec, rho), a in zip(pairs, reversed(alphas), strict=True):
            b = rho * float(y_vec @ z)
            z = z + (a - b) * s_vec
        return z

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        ran = 0
        pairs: list[tuple] = []  # (s = dx, y = dr, rho = 1/(y.s)) limited-memory secant pairs
        prev_r = None
        prev_dx = None
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            if err < rtol_energy_pct:
                converged = True
            last_energy = state.total_energy
            r = system.residual(x, state)
            if prev_r is not None:
                y = r - prev_r
                ys = float(y @ prev_dx)
                if abs(ys) > 1e-30:  # curvature condition; skip a non-informative pair
                    pairs.append((prev_dx.copy(), y, 1.0 / ys))
                    if len(pairs) > self.m:
                        pairs.pop(0)
            dx = -self._two_loop(system, state, r, pairs)
            x = x + dx
            prev_r, prev_dx = r, dx
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class NewtonLineSearchStrategy:
    """Damped Newton with a backtracking line search on ``||residual||`` (roadmap sec 1).

    Today's ``newton`` takes the full step ``x += dx``; deep saturation can make that overshoot and
    diverge. This scales the Newton step by ``lam`` from an Armijo backtracking line search that
    shrinks ``lam`` until ``||r(x + lam dx)||`` decreases -- the robust globaliser. Needs ``residual``
    (have it) plus ``tangent`` (falls back to a damped Picard step when ``tangent`` returns ``None``).
    Converges to the same root as Newton (at ``lam=1`` near the solution it IS Newton); same
    delayed-break energy bookkeeping.
    """

    name = 'newton-linesearch'

    def __init__(self, max_backtracks: int = 12, shrink: float = 0.5) -> None:
        self.max_backtracks = max_backtracks
        self.shrink = shrink

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
            if err < rtol_energy_pct:
                converged = True
            last_energy = state.total_energy
            jac = system.tangent(x, state)
            if jac is None:
                x = system.solve_linear(state)  # no Jacobian -> Picard substitution
            else:
                r = system.residual(x, state)
                dx = system.solve_system(jac, -r)
                # Minimise the FREE-DOF residual norm (residual_norm), NOT the raw ||r||: FEA's full
                # residual carries a large constant Dirichlet-row term the increment can't reduce, which
                # would make every backtracking step look like an increase and stall the line search.
                r0 = system.residual_norm(x, state)
                lam = 1.0
                for _bt in range(self.max_backtracks):
                    # r(x+lam dx) is assembled implicitly: re-assemble at the trial point and read
                    # its residual (the equation owns assembly). Accept the first lam that reduces it.
                    trial = x + lam * dx
                    if system.residual_norm(trial, system.assemble(trial)) < r0:
                        break
                    lam *= self.shrink
                x = x + lam * dx
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class NewtonKrylovStrategy:
    """Newton whose LINEAR system is solved by a Krylov solver -- optionally Jacobian-free (sec 1).

    When ``tangent`` is available the Newton system ``J dx = -r`` is solved by GMRES through the
    linear-solver seam (roadmap sec 2) instead of a direct factorization -- the path to large 3D /
    eddy-current systems. When ``tangent`` returns ``None`` it goes Jacobian-FREE (JFNK): the
    directional derivative is approximated ``J v ~ (r(x + eps v) - r(x)) / eps`` (needs only
    ``residual``), wrapped as a matrix-free ``LinearOperator`` GMRES applies. Converges to the same
    root as Newton; same delayed-break energy bookkeeping.
    """

    name = 'newton-krylov'

    def __init__(self, eps: float = 1e-7) -> None:
        self.eps = eps

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        # Local imports: the Krylov seam + LinearOperator are only needed by this one strategy, and
        # keeping them here avoids a scipy import on the module's hot default (Picard) path.
        import scipy.sparse as _sp  # noqa: PLC0415
        from scipy.sparse.linalg import LinearOperator, gmres  # noqa: PLC0415

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
            if err < rtol_energy_pct:
                converged = True
            last_energy = state.total_energy
            r = system.residual(x, state)
            jac = system.tangent(x, state)
            n = x.shape[0]
            if jac is not None and not (_sp.issparse(jac) or isinstance(jac, LinearOperator)):
                # The equation returned an OPAQUE tangent bundle (FEA's reduced-increment dict), not a
                # scipy operator GMRES can apply. Its own solve_system does the correct reduced Newton
                # increment solve (direct), so route through it: still an exact Newton step converging to
                # the same root -- a true reduced matrix-free FEA Krylov operator is a deferred follow-on.
                dx = system.solve_system(jac, -r)
            else:
                if jac is not None:
                    op = jac  # scipy sparse tangent (MEC) -> GMRES on the exact Jacobian
                else:
                    # Jacobian-free directional derivative around the current iterate (frozen state).
                    r_norm = float(np.linalg.norm(r)) or 1.0

                    def matvec(v, _r=r, _x=x, _rn=r_norm):
                        eps = self.eps * (1.0 + float(np.linalg.norm(_x))) / (float(np.linalg.norm(v)) or 1.0)
                        r_pert = system.residual(_x + eps * v, system.assemble(_x + eps * v))
                        return (r_pert - _r) / eps

                    op = LinearOperator((n, n), matvec=matvec)
                dx, info = gmres(op, -r, rtol=1e-8, atol=0.0, maxiter=4 * max(n, 1))
                if info != 0 or not np.all(np.isfinite(dx)):
                    dx = system.solve_linear(state) - x  # Krylov breakdown -> Picard substitution
            x = x + dx
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class ContinuationStrategy:
    """Load / source stepping: solve a ramp ``0 < lam_1 < ... < lam_n = 1`` of the source (sec 1).

    An OUTER driver for strong-saturation / large-current operating points where a single solve of the
    full source diverges. It scales the source (conductor current + magnet Br) by ``lam`` via the
    equation's ``scale_source`` hook, solving each sub-problem with an INNER strategy seeded by the
    previous solution (natural continuation). The last sub-solve (``lam=1``) is the full problem, so
    the converged field is the true solution. Degrades to a single full-source inner solve when the
    equation cannot rescale its source (``scale_source`` raises ``NotImplementedError``).
    """

    name = 'continuation'

    def __init__(self, steps: int = 4, inner: IterationStrategy | None = None) -> None:
        self.steps = max(1, steps)
        self.inner = inner or DampedPicardStrategy()

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        lambdas = [(k + 1) / self.steps for k in range(self.steps)]
        result: IterationResult | None = None
        seed: np.ndarray | None = None
        for lam in lambdas:
            try:
                sub = system if lam == 1.0 and self.steps == 1 else system.scale_source(lam)
            except NotImplementedError:
                # The equation cannot rescale its source: solve the full problem directly (no ramp).
                return self.inner.solve(system, max_iter=max_iter, rtol_energy_pct=rtol_energy_pct)
            if seed is not None:
                sub = _SeededSystem(sub, seed)
            result = self.inner.solve(sub, max_iter=max_iter, rtol_energy_pct=rtol_energy_pct)
            seed = result.x
        # ``result.state`` came from the scaled sub-system; re-assemble on the FULL system at the
        # converged iterate so the returned payload/energy are the true (lam=1) fields.
        final_state = system.assemble(result.x)
        return IterationResult(x=result.x, state=final_state, iterations=result.iterations, converged=result.converged)


class _SeededSystem:
    """Wrap a :class:`NonlinearSystem` to start from a supplied iterate (continuation warm start).

    Delegates every method to the wrapped system except :meth:`initial_guess`, which returns the seed
    (the previous continuation step's converged field) so each sub-solve starts near its solution.
    """

    def __init__(self, base: NonlinearSystem, seed: np.ndarray) -> None:
        self._base = base
        self._seed = seed

    def initial_guess(self) -> np.ndarray:
        return self._seed

    def __getattr__(self, name: str) -> object:
        return getattr(self._base, name)


ITERATION_REGISTRY: dict[str, IterationStrategy] = {
    'picard-damped': DampedPicardStrategy(),
    'frozen-damped': FrozenDampedStrategy(),
    'picard-adaptive': AdaptiveDampedPicardStrategy(),
    'picard': PicardStrategy(),
    'newton': NewtonStrategy(),
    'anderson': AndersonStrategy(),
    'modified-newton': ModifiedNewtonStrategy(),
    'quasi-newton': BroydenStrategy(),
    'bfgs': BFGSStrategy(),
    'l-bfgs': LBFGSStrategy(),
    'newton-linesearch': NewtonLineSearchStrategy(),
    'newton-krylov': NewtonKrylovStrategy(),
    'continuation': ContinuationStrategy(),
}


def get_strategy(name: str | None) -> IterationStrategy:
    """Resolve an iteration-strategy name; ``None`` -> the default damped-Picard (today's behaviour)."""
    if name is None:
        return ITERATION_REGISTRY['picard-damped']
    try:
        return ITERATION_REGISTRY[name]
    except KeyError as e:
        msg = f'unknown iteration strategy {name!r}; known: {sorted(ITERATION_REGISTRY)}'
        raise KeyError(msg) from e
