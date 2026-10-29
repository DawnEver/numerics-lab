"""Newton-family and continuation iteration strategies, split from `nonlinear/__init__.py` because
that module crossed the 800-line band. See its docstring for the NonlinearSystem/IterationStrategy
protocol these implement.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator, gmres, splu

from numerics_lab.line_search import backtracking, sufficient_decrease
from numerics_lab.linear import reject_nonfinite_solution
from numerics_lab.linear.native_direct import native_factor
from numerics_lab.nonlinear._seam import (
    _RTOL_ENERGY_PCT,
    AssembledState,
    FactorizableSystem,
    IterationResult,
    IterationStrategy,
    NewtonTangentUnavailable,
    NonlinearSystem,
    ResidualReferencedSystem,
    SourceScalableSystem,
    _energy_error_pct,
    _has_converged,
    _StagnationWatch,
    increment_ratio,
    require_secant_system,
    unwrap_system,
)
from numerics_lab.nonlinear._strategies_picard import DampedPicardStrategy
from numerics_lab.nonlinear.local import (
    local_change,
    newton_linear_tolerance,
)

logger = logging.getLogger(__name__)

__all__ = [
    'HELD_FACTOR_REFACTOR_EVERY',
    'ContinuationStrategy',
    'HeldFactorNewtonStrategy',
    'ModifiedNewtonStrategy',
    'NewtonKrylovStrategy',
    'NewtonLineSearchStrategy',
    'NewtonStrategy',
]


def _free_residual_norm(system, x, state) -> float:
    """``system.residual_norm`` when the system defines it, else the Protocol's default ``||r||``.

    A duck-typed system inherits nothing from a ``Protocol``, so the default body has to be
    spelled where it is consumed.
    """
    try:
        norm_of = system.residual_norm
    except AttributeError:
        return float(np.linalg.norm(system.residual(x, state)))
    return float(norm_of(x, state))


def _residual_reference(system, state, seed_norm: float) -> float:
    """The scale the relative residual is taken against: ``max(||r(x_0)||, ||rhs||)``.

    The seed residual alone is the right-hand side only for a ZERO seed. A warm start near the
    root (a sweep's previous position) has a small seed residual, and a bar relative to it would
    demand a tighter answer the better the guess -- so a system that knows its free-DOF source norm
    exposes it as ``rhs_norm(state)`` and the larger of the two is used, which is the rule the 3-D
    H(curl) march already applies (``ref = max(||f_f||, ||R_f^0||)``). A system without the hook
    is judged against its seed residual, exact for the zero seed every such system starts from.
    """
    inner = unwrap_system(system)
    if not isinstance(inner, ResidualReferencedSystem):
        return seed_norm
    return max(seed_norm, float(inner.rhs_norm(state)))


class NewtonStrategy:
    """Newton-Raphson with a backtracking line search -- THE Newton every family runs.

    ``J(x_k) dx = -r(x_k)``, then ``x_{k+1} = x_k + lam dx`` with the first ``lam`` in
    ``1, shrink, shrink**2, ...`` whose trial satisfies the Armijo decrease
    ``||r(x + lam dx)|| <= (1 - c lam) ||r(x)||`` on the FREE-DOF residual norm
    (:meth:`NonlinearSystem.residual_norm`). Near the root ``lam = 1`` is always accepted, so the
    iteration IS Newton there and converges quadratically; far from it the search keeps a deep-
    saturation overshoot from running away.

    ONE IMPLEMENTATION, 2026-09-25. There used to be two -- ``newton`` (full step, energy-change
    stop) and ``newton-linesearch`` (this search, the same energy stop) -- plus a hand-rolled march
    per family. The equation owns the tangent (FEA and MEC assemble the exact
    ``nu + 2 B^2 dnu/dB^2`` Jacobian from the declared B-H law, vacuum tail included; SDM applies a
    Jacobian-free directional derivative of its own pipeline); this class owns everything else.

    CONVERGENCE IS JUDGED ON THE EQUATION, not on a proxy for it: the relative residual
    ``||r(x_k)|| / max(||r(x_0)||, ||rhs||)`` (:func:`_residual_reference`) AND the increment, both below ``rtol = rtol_energy_pct / 100``, through the one rule
    :func:`_has_converged`. The energy change is still computed, for the log and for the
    stagnation break that stops a solve parked on its arithmetic floor.

    AN UNACCEPTED SEARCH TAKES THE FULL STEP. When no ``lam`` decreases the residual the direction
    is either a genuine Newton step out of a basin the norm cannot see (the first step from a zero
    field into saturation is the measured case) or a step at the roundoff floor, where the residual
    no longer decreases because it cannot. Plain Newton takes the full step in both, and doing the
    same here means the search can only ever help. The trial assembled for the accepted step is the
    state of the next iterate, so an accepted ``lam = 1`` costs one assembly, one tangent and one
    linear solve -- the same as the full-step Newton it replaces.

    A system with no tangent RAISES :class:`NewtonTangentUnavailable` (``UNSUPPORTED-RAISES``); the
    Picard step this used to substitute silently ran a different algorithm under Newton's name.
    """

    name = 'newton'
    #: How this strategy decides it has converged: on the equation's RESIDUAL (every other
    #: strategy stops on the energy change). The caller that picks the tolerance reads this, so the
    #: residual bar and the fixed-point bar cannot be confused (``core.newton_rtol`` against
    #: ``core.fixed_point_rtol``, user ruling 2026-09-26).
    stop_rule = 'residual'

    def __init__(
        self,
        max_backtracks: int = 12,
        shrink: float = 0.5,
        armijo: float = 1e-4,
        *,
        refactor_every: int | None = None,
        stall_ratio: float = 0.5,
    ) -> None:
        self.max_backtracks = int(max_backtracks)
        if self.max_backtracks < 0:
            msg = f'max_backtracks must be nonnegative, got {max_backtracks!r}'
            raise ValueError(msg)
        self.shrink = float(shrink)
        self.armijo = float(armijo)
        #: ``None`` is full Newton: a fresh tangent and factor every iteration. An integer ``k`` is
        #: the MODIFIED Newton (user ruling 2026-09-26, 3A): the tangent is formed and factored at
        #: most every ``k`` iterations and its factor back-substituted in between, refreshed early
        #: when the held step stalls -- see :meth:`_held_solver` and :meth:`solve`.
        if refactor_every is not None and int(refactor_every) < 1:
            msg = f'refactor_every must be None (full Newton) or >= 1, got {refactor_every!r}'
            raise ValueError(msg)
        self.refactor_every = None if refactor_every is None else int(refactor_every)
        self.stall_ratio = float(stall_ratio)
        #: Tangent factorizations the LAST ``solve`` paid for -- the machine-independent record a
        #: test pins and a benchmark reports (a wall-clock is a fact about a moment).
        self.factorizations = 0

    def _held_solver(self, system: NonlinearSystem, x: np.ndarray, state: AssembledState):
        """Form the tangent at ``x`` and return ``rhs -> J^{-1} rhs`` for it.

        Full Newton hands the tangent straight back to ``system.solve_system`` -- the path every
        family ran before the modified variant existed, bit for bit. The modified Newton FACTORS it
        once so the held steps only back-substitute: through the system's own ``factorize(jac)``
        when it has one (FEA's tangent is a reduced bundle only it can fold and recover), else
        through ``splu`` on a scipy-sparse tangent. A tangent that is neither (the SDM's
        Jacobian-free operator) cannot be factored, so it is re-solved each time: still correct,
        only the tangent formation is saved.
        """
        jac = system.tangent(x, state)
        if jac is None:
            raise NewtonTangentUnavailable(system, self.name)
        self.factorizations += 1
        if self.refactor_every is None:
            return lambda rhs: system.solve_system(jac, rhs)
        if isinstance(system, FactorizableSystem):
            return system.factorize(jac)
        lu = _lu_factorize(jac)
        if lu is None:
            return lambda rhs: system.solve_system(jac, rhs)
        return lambda rhs: reject_nonfinite_solution(lu.solve(np.asarray(rhs, dtype=float)), self.name)

    def _search(self, system: NonlinearSystem, x: np.ndarray, dx: np.ndarray, r_norm: float):
        """The Armijo backtracking along ``dx`` on the residual norm: ``(lam, accepted, trial, trial_state, trial_norm, full_step)``.

        The merit is ``||r(x + lam dx)||``, whose slope along an exact Newton step is ``-||r(x)||``.
        """

        def trial_at(lam: float) -> tuple[np.ndarray, AssembledState, float]:
            trial = x + lam * dx
            trial_state = system.assemble(trial)
            return trial, trial_state, _free_residual_norm(system, trial, trial_state)

        decreases = sufficient_decrease(r_norm, -r_norm, c1=self.armijo)
        search = backtracking(
            trial_at, lambda lam, t: decreases(lam, t[2]), tries=self.max_backtracks + 1, shrink=self.shrink
        )
        assert search.trial is not None and search.first is not None  # every Newton trial evaluates
        trial, trial_state, trial_norm = search.trial
        return search.step, search.accepted, trial, trial_state, trial_norm, search.first

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        self.factorizations = 0
        x = system.initial_guess()
        if max_iter <= 0:
            # NO ITERATION RUNS NO ASSEMBLY, the contract every strategy keeps: the caller gets
            # back its pre-initialised fields, never an assembly nobody asked for.
            return IterationResult(x=x, state=None, iterations=0, converged=False)
        state = system.assemble(x)
        r_norm = _free_residual_norm(system, x, state)
        reference = _residual_reference(system, state, r_norm)
        if r_norm == 0.0 or not np.isfinite(reference):
            # A seed that already satisfies the equation (an unexcited model, a warm start AT the
            # root) is the solution; a non-finite seed residual is refused by the caller's guards.
            return IterationResult(x=x, state=state, iterations=1, converged=r_norm == 0.0)
        last_energy: float | None = state.total_energy
        # THE TRACE STARTS AT THE SEED (user ruling 2026-10-04): `on_iteration` reports the state an
        # iteration STARTS from, beside the step taken out of it -- the Picard-family convention --
        # so the first hook call carries the seed (rel = 1 from A = 0), and the closing state, which
        # no hook call sees, is the caller's to record from the result. `start_err` is that state's
        # own energy change against its predecessor, `inf` for the seed.
        start_state, start_err = state, np.inf
        converged = False
        ran = 0
        watch = _StagnationWatch(rtol_energy_pct)
        held = None
        age = 0
        stale = False
        for i in range(max_iter):
            ran = i + 1
            previous_local = state.local_values
            try:
                set_tolerance = system.set_linear_tolerance
            except AttributeError:
                set_tolerance = None
            if set_tolerance is not None:
                set_tolerance(newton_linear_tolerance(r_norm / reference) * reference)
            # FULL NEWTON REFRESHES EVERY ITERATION; the modified one when its factor is ``k`` old,
            # or the last held step contracted the residual by less than ``stall_ratio``.
            fresh = held is None or self.refactor_every is None or age >= self.refactor_every or stale
            if fresh:
                held, age, stale = self._held_solver(system, x, state), 0, False
            rhs = -system.residual(x, state)
            assert held is not None  # the first iteration and every stale factor refresh above
            dx = held(rhs)
            x_prev = x
            lam, accepted, trial, trial_state, trial_norm, full_step = self._search(system, x, dx, r_norm)
            if not fresh and not (accepted and lam == 1.0):
                # A HELD FACTOR THAT NEEDS A LINE SEARCH IS NO LONGER A NEWTON MODEL of this iterate:
                # refresh here and retake the step, rather than march a stale direction (the failure
                # the legacy energy-stopped modified variant never guarded against).
                held, age = self._held_solver(system, x, state), 0
                dx = held(rhs)
                lam, accepted, trial, trial_state, trial_norm, full_step = self._search(system, x, dx, r_norm)
            age += 1
            if not accepted:
                trial, trial_state, trial_norm = full_step
                lam = 1.0
                if self.max_backtracks > 0:
                    # RE-ASSEMBLED, not reused: the last assembly was at the SMALLEST trial, and an
                    # equation may keep per-element state from its latest assembly (FEA's blocks do),
                    # so the returned state must be the one assembled at the iterate returned.
                    trial_state = system.assemble(trial)
            stale = self.refactor_every is not None and not trial_norm <= self.stall_ratio * r_norm
            x, state, r_norm = trial, trial_state, trial_norm
            err = _energy_error_pct(state.total_energy, last_energy)
            last_energy = state.total_energy
            # THE RULE IS ASKED ABOUT THE FULL CORRECTION `dx`, not the damped `lam dx`: a line
            # search that exhausted its backtracks takes a shrink**k-scaled step, and read as an
            # iterate that stopped moving that is a `converged=True` far from the root -- measured
            # 2026-09-27 on case_s fea,tri (two identical 1.45e-5 increments, then "converged").
            x_full = x_prev + dx
            increment = increment_ratio(x_prev, x_full)
            converged = _has_converged(
                err,
                x_prev,
                x_full,
                rtol_energy_pct,
                material_change=state.material_change,
                residual_ratio=r_norm / reference,
            ) or watch.observe(err, increment, state.material_change)
            if system.is_linear:
                converged = True
            if not system.is_linear and (previous_local or state.local_values):
                converged = converged and local_change(previous_local, state.local_values) < rtol_energy_pct / 100.0
            system.on_iteration(i, start_state, start_err, lam, increment)
            start_state, start_err = state, err
            if converged:
                break
        # ONE line per solve, the numbers a tolerance or strategy ruling is judged by: the relative
        # residual it stopped at and the factorizations it paid (full Newton: one per iteration).
        logger.info(
            '%s stopped: iterations=%d converged=%s relres=%.3e factorizations=%d energy=%.12e',
            self.name,
            ran,
            converged,
            r_norm / reference,
            self.factorizations,
            state.total_energy,
        )
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
    if not sp.issparse(matrix):
        return None
    native = native_factor(matrix)
    return native if native is not None else splu(sp.csc_matrix(matrix))


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
        watch = _StagnationWatch(rtol_energy_pct)
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
            increment = increment_ratio(x_prev, x)
            converged = _has_converged(
                err, x_prev, x, rtol_energy_pct, material_change=state.material_change
            ) or watch.observe(err, increment, state.material_change)
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0, increment)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class NewtonLineSearchStrategy(NewtonStrategy):
    """The registry name ``newton-linesearch``, kept for callers that spell it -- and it IS ``newton``.

    It was a second Newton loop until 2026-09-25: the same backtracking search, bolted onto the
    energy-change stop, beside a ``newton`` that took the full step. The search is now part of
    :class:`NewtonStrategy`, so this subclass changes nothing but the name it reports; two
    registry entries naming one algorithm is said here rather than discovered.
    """

    name = 'newton-linesearch'


#: How many iterations one held tangent factor serves before it is refreshed regardless. Chosen
#: against the measured Newton counts it has to beat (10-26 stationary, ~12 per transient step,
#: 2026-09-25): at 4 a Newton that needed 12 pays at most 3-4 factorizations plus the stall refreshes.
HELD_FACTOR_REFACTOR_EVERY = 4


class HeldFactorNewtonStrategy(NewtonStrategy):
    """The MODIFIED Newton, ``newton-held-factor`` -- the same class, holding its factor (ruling 3A).

    DECLARED, SELECTABLE, NOT THE DEFAULT (user ruling 2026-09-26). The tangent is formed and
    factored at most every :data:`HELD_FACTOR_REFACTOR_EVERY` iterations and back-substituted in
    between; it is refreshed EARLY when a held step needs a line search (retaken at once with the
    fresh factor) or contracts the residual by less than ``stall_ratio`` (refreshed for the next).
    Convergence is judged by the SAME residual rule on the SAME ``core.newton_rtol`` as full Newton,
    so it can only stop where full Newton would have been allowed to: what changes is the path and
    the price per iteration (a back-substitution against a factorization), never the bar.

    It is a NEW name rather than the older ``modified-newton``, deliberately: that one is a separate
    energy-stopped loop mirrored by the Rust port's golden, falls back to a Picard substitution when
    no tangent exists, and was measured diverging on case_d fea (2026-09-25). Reusing its name
    for a different algorithm would be one name for two algorithms -- the declaration-that-lies shape.
    """

    name = 'newton-held-factor'

    def __init__(self, refactor_every: int = HELD_FACTOR_REFACTOR_EVERY, stall_ratio: float = 0.5) -> None:
        super().__init__(refactor_every=refactor_every, stall_ratio=stall_ratio)


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
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        ran = 0
        watch = _StagnationWatch(rtol_energy_pct)
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
            if jac is not None and not (sp.issparse(jac) or isinstance(jac, LinearOperator)):
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

                    op = aslinearoperator(SimpleNamespace(shape=(n, n), matvec=matvec))
                dx, info = gmres(op, -r, rtol=1e-8, atol=0.0, maxiter=4 * max(n, 1))
                if info != 0 or not np.all(np.isfinite(dx)):
                    dx = system.solve_linear(state) - x  # Krylov breakdown -> Picard substitution
            x = x + dx
            increment = increment_ratio(x_prev, x)
            converged = _has_converged(
                err, x_prev, x, rtol_energy_pct, material_change=state.material_change
            ) or watch.observe(err, increment, state.material_change)
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, 1.0, increment)
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
        if not isinstance(unwrap_system(system), SourceScalableSystem):
            return self.inner.solve(system, max_iter=max_iter, rtol_energy_pct=rtol_energy_pct)
        if not isinstance(system, SourceScalableSystem):
            msg = 'the nonlinear wrapper does not delegate scale_source'
            raise TypeError(msg)
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
        assert result is not None  # steps is at least one
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

    @property
    def unwrapped(self) -> NonlinearSystem:
        return self._base

    @property
    def is_linear(self) -> bool:
        return bool(self._base.is_linear)

    def assemble(self, x, /):
        return self._base.assemble(x)

    def solve_linear(self, state, /):
        return self._base.solve_linear(state)

    def residual(self, x, state, /):
        return self._base.residual(x, state)

    def residual_norm(self, x, state, /):
        return self._base.residual_norm(x, state)

    def tangent(self, x, state, /):
        return self._base.tangent(x, state)

    def solve_system(self, matrix, rhs, /):
        return self._base.solve_system(matrix, rhs)

    def solve_secant(self, state, rhs, /):
        return require_secant_system(self._base).solve_secant(state, rhs)

    def on_iteration(self, i, state, err, alpha, increment, /):
        return self._base.on_iteration(i, state, err, alpha, increment)

    def scale_source(self, lam, /):
        if isinstance(self._base, SourceScalableSystem):
            return self._base.scale_source(lam)
        msg = 'the wrapped equation cannot scale_source'
        raise NotImplementedError(msg)
