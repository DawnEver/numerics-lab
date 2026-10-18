"""Multi-secant and quasi-Newton strategies: Anderson acceleration and the Broyden / BFGS / L-BFGS
family -- none of which factorises the tangent.

Split out of `_strategies_newton.py` on 2026-09-29, which keeps the factorising Newton family and
continuation.
"""

from __future__ import annotations

import numpy as np

from numerics_lab.nonlinear._seam import (
    _RTOL_ENERGY_PCT,
    AssembledState,
    IterationResult,
    NonlinearSystem,
    _energy_error_pct,
    _has_converged,
    _StagnationWatch,
    increment_ratio,
    require_secant_system,
)
from numerics_lab.quasi_newton import BroydenUpdate, SecantPair, broyden_apply, broyden_update, secant_pair, two_loop

__all__ = [
    'AndersonStrategy',
    'BFGSStrategy',
    'BroydenStrategy',
    'LBFGSStrategy',
]


def _two_loop(system, state, r, pairs):
    """H r over the stored secant pairs; H0 r = one secant solve of the system."""
    return two_loop(r, pairs, lambda q: require_secant_system(system).solve_secant(state, q))


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
        watch = _StagnationWatch(rtol_energy_pct)
        x_hist: list[np.ndarray] = []  # past g(x_k) images
        f_hist: list[np.ndarray] = []  # past residuals f_k = g(x_k) - x_k
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            last_energy = state.total_energy
            x_prev = x
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
            increment = increment_ratio(x_prev, x)
            converged = _has_converged(
                err, x_prev, x, rtol_energy_pct, material_change=state.material_change
            ) or watch.observe(err, increment, state.material_change)
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, self.beta, increment)
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
        return broyden_apply(require_secant_system(system).solve_secant(state, r), updates)

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        ran = 0
        watch = _StagnationWatch(rtol_energy_pct)
        require_secant_system(system)
        updates: list[BroydenUpdate] = []  # rank-1 inverse-Jacobian updates, oldest first
        prev_r = None
        prev_dx = None
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            last_energy = state.total_energy
            x_prev = x
            r = system.residual(x, state)
            if prev_r is not None:
                # Broyden "good" inverse update: H <- H + ((dx - H df) / (dx.H df)) dx.H, applied as a
                # stored rank-1 pair so future applies stay matrix-free.
                assert prev_dx is not None
                df = r - prev_r
                h_df = self._apply_inverse(system, state, df, updates)
                update = broyden_update(prev_dx, h_df)
                if update is not None:
                    updates.append(update)
                    if len(updates) > self.m:
                        updates.pop(0)
            dx = -self._apply_inverse(system, state, r, updates)
            x = x + dx
            prev_r, prev_dx = r, dx
            increment = increment_ratio(x_prev, x)
            converged = _has_converged(
                err, x_prev, x, rtol_energy_pct, material_change=state.material_change
            ) or watch.observe(err, increment, state.material_change)
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0, increment)
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

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        ran = 0
        watch = _StagnationWatch(rtol_energy_pct)
        require_secant_system(system)
        pairs: list[SecantPair] = []  # (s = dx, y = dr, rho = 1/(y.s)), oldest first
        prev_r = None
        prev_dx = None
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            last_energy = state.total_energy
            x_prev = x
            r = system.residual(x, state)
            if prev_r is not None:
                assert prev_dx is not None
                y = r - prev_r
                pair = secant_pair(prev_dx, y)  # None: a non-informative (near-zero curvature) pair
                if pair is not None:
                    pairs.append(pair)
                    if len(pairs) > self.m:
                        pairs.pop(0)
            dx = -_two_loop(system, state, r, pairs)
            x = x + dx
            prev_r, prev_dx = r, dx
            increment = increment_ratio(x_prev, x)
            converged = _has_converged(
                err, x_prev, x, rtol_energy_pct, material_change=state.material_change
            ) or watch.observe(err, increment, state.material_change)
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0, increment)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class LBFGSStrategy(BFGSStrategy):
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
        super().__init__(m)  # limited-memory depth (the last m secant pairs)

    # `solve` is BFGS's own: the two differ only in `m`, the depth of the history the shared
    # `_two_loop` recursion is applied over.
