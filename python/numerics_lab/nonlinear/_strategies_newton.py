"""Newton-family and continuation iteration strategies, split from `nonlinear/__init__.py` because
that module crossed the 800-line band. See its docstring for the NonlinearSystem/IterationStrategy
protocol these implement.
"""

from __future__ import annotations

import numpy as np

from motronics.gauss.nonlinear import (
    _RTOL_ENERGY_PCT,
    AssembledState,
    IterationResult,
    IterationStrategy,
    NonlinearSystem,
    _energy_error_pct,
    _has_converged,
)
from motronics.gauss.nonlinear._strategies_picard import DampedPicardStrategy

__all__ = [
    'AndersonStrategy',
    'BFGSStrategy',
    'BroydenStrategy',
    'ContinuationStrategy',
    'LBFGSStrategy',
    'ModifiedNewtonStrategy',
    'NewtonKrylovStrategy',
    'NewtonLineSearchStrategy',
    'NewtonStrategy',
]


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
            last_energy = state.total_energy
            x_prev = x
            jac = system.tangent(x, state)
            if jac is None:
                x = system.solve_linear(state)  # no Jacobian available -> Picard substitution
            else:
                dx = system.solve_system(jac, -system.residual(x, state))
                x = x + dx
            converged = _has_converged(err, x_prev, x, rtol_energy_pct)
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
            converged = _has_converged(err, x_prev, x, rtol_energy_pct)
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
            last_energy = state.total_energy
            x_prev = x
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
            converged = _has_converged(err, x_prev, x, rtol_energy_pct)
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
            last_energy = state.total_energy
            x_prev = x
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
            converged = _has_converged(err, x_prev, x, rtol_energy_pct)
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
            last_energy = state.total_energy
            x_prev = x
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
            converged = _has_converged(err, x_prev, x, rtol_energy_pct)
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
            last_energy = state.total_energy
            x_prev = x
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
            converged = _has_converged(err, x_prev, x, rtol_energy_pct)
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
            last_energy = state.total_energy
            x_prev = x
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
            converged = _has_converged(err, x_prev, x, rtol_energy_pct)
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
            last_energy = state.total_energy
            x_prev = x
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
            converged = _has_converged(err, x_prev, x, rtol_energy_pct)
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
