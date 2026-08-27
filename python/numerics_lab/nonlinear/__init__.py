"""Pluggable nonlinear-iteration strategies, decoupled from the governing equation.

Design ``src/motronics/gauss/.claude/memory/2026/07/09/design-solver-iteration-decoupling.md`` (a179884): the
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

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = [
    'DEFAULT_SOLVE_BUDGET_S',
    'ITERATION_REGISTRY',
    'AdaptiveDampedPicardStrategy',
    'AndersonStrategy',
    'AssembledState',
    'BFGSStrategy',
    'BroydenStrategy',
    'ContinuationStrategy',
    'DampedPicardStrategy',
    'IterationBudgetExceeded',
    'IterationResult',
    'IterationStrategy',
    'LBFGSStrategy',
    'ModifiedNewtonStrategy',
    'NewtonKrylovStrategy',
    'NewtonLineSearchStrategy',
    'NewtonStrategy',
    'NonlinearSystem',
    'NuRelaxationDampedPicardStrategy',
    'PicardStrategy',
    'get_strategy',
    'with_time_budget',
]

# Damped-Picard parameters (moved verbatim from mec/_sparse.py so the extracted loop is identical).
_ALPHA_MIN = 0.1  # floor for the adaptive under-relaxation factor
_ALPHA_GROW = 1.5  # adaptive-Picard recovery factor: grow alpha back toward 1 on a progressing step
_RTOL_ENERGY_PCT = 0.01  # convergence: total-energy change below this fraction of a percent
#: Consecutive UNDISTURBED steps (no energy-error growth, no energy-delta sign flip) before the
#: damped Picard loop grows ``alpha`` back. The ratchet used to be one-way, so a single early
#: overshoot damped the whole smooth tail: measured on ipm_uv_cummins mec:quad, alpha stuck at
#: ``1.0 x 0.5 x 0.6**3 = 0.108`` and every rotor position took 44-61 iterations at ~3 s each.
#: Recovery is capped at ``alpha_cap`` -- the step size that last misbehaved is never retried --
#: so a genuine limit cycle still settles instead of oscillating between the two regimes.
_ALPHA_CALM_STEPS = 3


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

    def on_iteration(self, i: int, state: AssembledState, err: float, alpha: float, increment: float, /) -> None:
        """Optional per-iteration hook (logging); default no-op.

        ``increment`` is ``||x_next - x|| / ||x||``, the half of the convergence rule the energy
        error cannot express. Both halves are passed because a hook that saw only ``err`` reported
        a capped solve and a converged one identically.
        """

    def scale_source(self, lam: float, /) -> NonlinearSystem:
        """Return a copy of this system with its SOURCE scaled by ``lam`` (continuation stepping).

        Optional -- only the ``continuation`` strategy calls it. An equation that cannot cheaply
        rescale its source (conductor current + magnet Br) raises :class:`NotImplementedError`, and
        the strategy degrades to solving the full-source problem directly. ``lam=1`` returns the
        original (full-source) system.
        """


class IterationBudgetExceeded(RuntimeError):
    """A nonlinear solve ran past the wall-clock it was given, and stopped ITSELF.

    RAISED RATHER THAN RETURNED, and the deciding fact is not a preference: `IterationResult`
    already carries `converged`, and NOTHING in `hamilton/` reads it -- every occurrence of the word
    in that package is prose. A solve that returned `converged=False` here would be "a warning on an
    unchanged success return", and the test of that shape is whether the CALLER can tell. It cannot.
    """

    def __init__(self, *, budget_s: float, elapsed_s: float, where: str) -> None:
        self.budget_s = float(budget_s)
        self.elapsed_s = float(elapsed_s)
        self.where = where
        super().__init__(
            f'nonlinear solve exceeded its {budget_s:g} s budget before {where} '
            f'({elapsed_s:g} s elapsed). The budget stops the solve at the next expensive call '
            f'rather than letting the tier be killed around it -- an unbounded solve inside a '
            f'bounded tier costs every other result in the run its verdict, which is how a heavy '
            f'run returns INCONCLUSIVE instead of one red test.'
        )


class _TimeBudgetedSystem:
    """A `NonlinearSystem` that refuses to START another expensive call once its time is spent.

    WRAPPING THE SYSTEM, NOT THE STRATEGY, is the whole design. The cost lives in `solve_linear`
    (for sdm, a dense truncated-SVD least squares) and `assemble`; putting the check there is
    putting it where the money is spent, and it leaves all four `IterationStrategy` implementations
    and their Protocol untouched. A budget threaded through four loops would be four places to
    forget it.

    OVERSHOOT IS BOUNDED BY ONE CALL, said plainly because the alternative is a promise the code
    cannot keep: no Python-level check can unwind a call already inside LAPACK. That is the same
    limitation `pytest.mark.timeout`'s thread method hits on Windows -- it dumps stacks and takes
    the process -- and the difference here is that the refusal arrives as a typed exception the
    caller can attribute, instead of as a missing summary nobody can attribute to anything.

    THE BUDGETED SET IS EVERY EXPENSIVE ENTRY POINT, not just the two the Picard default happens to
    use: `assemble`, `solve_linear`, `solve_system`, `solve_secant` and `tangent`. Leaving the
    Newton-family calls unbudgeted would put a hole in the guard exactly where a caller who switched
    strategy would not think to look, and a scope claim with an unbudgeted path in it is the
    "declaration that lies" shape. `BUDGETED_CALLS` names the set so a test can hold this docstring
    to it rather than trusting it.

    Everything else delegates, so a budgeted system is the same system: `is_linear`, `initial_guess`,
    `residual`, `on_iteration` and any implementation-private attribute a strategy reads all pass
    straight through -- none of them solves anything.
    """

    #: The calls this wrapper refuses to START once the budget is gone. Declared as data so the
    #: scope claim above is checkable instead of merely asserted.
    BUDGETED_CALLS = ('assemble', 'solve_linear', 'solve_system', 'solve_secant', 'tangent')

    def __init__(self, inner, budget_s: float, clock) -> None:
        self._inner = inner
        self._budget_s = float(budget_s)
        self._clock = clock
        self._started = clock()

    def __getattr__(self, name: str) -> object:
        # Reached only for names this wrapper does not define, which is every part of the
        # protocol that costs nothing -- `initial_guess`, `residual`, `on_iteration`, `is_linear`.
        return getattr(self._inner, name)

    def _check(self, where: str) -> None:
        elapsed = self._clock() - self._started
        if elapsed > self._budget_s:
            raise IterationBudgetExceeded(budget_s=self._budget_s, elapsed_s=elapsed, where=where)

    def assemble(self, x, /):
        self._check('assemble')
        return self._inner.assemble(x)

    def solve_linear(self, state, /):
        self._check('solve_linear')
        return self._inner.solve_linear(state)

    def solve_system(self, matrix, rhs, /):
        self._check('solve_system')
        return self._inner.solve_system(matrix, rhs)

    def solve_secant(self, state, rhs, /):
        self._check('solve_secant')
        return self._inner.solve_secant(state, rhs)

    def tangent(self, x, state, /):
        self._check('tangent')
        return self._inner.tangent(x, state)


#: Default wall-clock ceiling for ONE nonlinear solve, in seconds. Consumed by the sdm, fea and mec
#: drivers alike -- the mechanism is physics-free, and so is the reason a solve must be able to stop.
#:
#: THE ARITHMETIC THAT PICKS IT is the TIER's, not the solve's. The budget is per solve and a
#: rotation sweep runs at least one per rotor position, so what has to hold is
#: ``n_positions * budget < the test's own timeout``. `ipm_uv` declares ``n_positions = 7`` and
#: `test_torque_ripple_sweep_discovers_and_runs` carries ``@pytest.mark.timeout(5400)``:
#: 7 x 500 = 3500 s fits. At 900 s it is 6300 s -- over the timeout, so on Windows the firing
#: `pytest-timeout` ``os._exit``s the WORKER and every result queued behind it is lost. A budget too
#: GENEROUS is therefore not the safe direction; it is no budget at all.
#:
#: THE MEASUREMENTS IT SITS BETWEEN: ~200 s per solve for the cases that complete, and 15-17 min for
#: the sdm case that hangs. MEASURED 2026-08-27, the reason this is no longer sdm-only: probing the
#: five `dcm_*` ripple cases whose results were MISSING from a truncated heavy run, `dcm_fdm_pear`
#: (FEA-Galerkin) and `dcm_fdm_rect` (MEC Picard) both ran past 900 s standalone. Neither is an sdm
#: solve, so the sdm-only budget could not have reached them.
#:
#: WHAT THIS DELIBERATELY COSTS: a solve that legitimately needs longer now RAISES. That trade is
#: taken knowingly -- a tier that returns INCONCLUSIVE loses EVERY result, so one named red leg is
#: strictly more information than none. A caller that knows its solve is genuinely long passes its
#: own ``seconds``.
DEFAULT_SOLVE_BUDGET_S = 500.0


def with_time_budget(system, *, seconds: float | None, clock=None):
    """`system`, refusing to start another expensive call after `seconds`. `None` returns it unchanged.

    Returning the SAME OBJECT for `None` is deliberate: opting out must be indistinguishable from
    never having opted in, so a caller that passes no budget cannot be running through a wrapper
    that might behave differently in some path nobody exercised.
    """
    if seconds is None:
        return system
    return _TimeBudgetedSystem(system, seconds, clock or time.monotonic)


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


def _has_converged(
    err: float,
    x: np.ndarray,
    x_next: np.ndarray,
    rtol_energy_pct: float,
) -> bool:
    """The ONE convergence rule: the energy has settled AND the iterate has stopped moving.

    A FLAT ENERGY IS NOT A FIXED POINT. `total_energy` can sit on a near-flat plateau that is not
    the solution -- measured on a saturating-BH case, which read 0.00% energy change for 20+
    iterations at an energy level ~14x the physical one before escaping and reaching the genuine
    fixed point ~15 iterations later. On an energy-only test that is a `converged=True` return at a
    wrong answer, and the caller cannot tell.

    That failure mode is a property of the ENERGY CRITERION, not of any one strategy. It was
    previously guarded inside :class:`DampedPicardStrategy` alone, and even there only behind
    ``if ratcheted`` -- so it did not fire on an undamped run of the very strategy that owned it,
    and the other eleven strategies never had it. Hence one function, consulted by all of them.

    The second half is the direct fixed-point test ``x_next == x``, immune to a flat-but-wrong
    energy plateau, on the step the strategy is ABOUT to take. It costs nothing: every strategy has
    already computed ``x_next`` by the time it asks. ``rtol_energy_pct`` is a PERCENT and the
    increment is a FRACTION, so the ``/ 100`` is the unit conversion between them, not a second,
    tighter tolerance.
    """
    return err < rtol_energy_pct and increment_ratio(x, x_next) < rtol_energy_pct / 100.0


def increment_ratio(x: np.ndarray, x_next: np.ndarray, /) -> float:
    """``||x_next - x|| / ||x||`` -- the SECOND half of :func:`_has_converged`, on its own.

    Split out so the number can be REPORTED, not only tested. A run that stops at ``max_iter``
    used to log the energy error alone, and the two states that produces are opposite: an energy
    plateau at a genuine fixed point (the criterion is too strict) and an energy plateau the
    iterate is still wandering across (the criterion is right and the solve has not converged).
    Measured 2026-08-27, one heavy shard capped 104 solves at 100/100 while logging 0.00% energy
    error, and the log could not distinguish those two -- which is why this is observable now.
    """
    x_scale = max(float(np.linalg.norm(x)), 1e-12)
    return float(np.linalg.norm(np.asarray(x_next) - np.asarray(x))) / x_scale


def _energy_flux_convergence(
    total_energy: float,
    total_flux: float,
    last_energy: float | None,
    last_flux: float | None,
    rtol_energy_pct: float,
    rtol_flux_pct: float,
) -> tuple[float, float, bool]:
    """Energy-AND-flux convergence test (fraction of a percent each), both below tolerance to pass.

    A private copy of ``hamilton.discretizations.fem._post._check_convergence``'s formula -- NOT an
    import of it: ``gauss`` sits BELOW ``hamilton`` in the dependency arrow (``hamilton -> ... ->
    gauss -> core -> contracts``), so this module cannot import from ``hamilton`` without inverting
    it. :class:`NuRelaxationDampedPicardStrategy` needs this two-metric check (not the single-metric
    :func:`_energy_error_pct` the other strategies use) to reproduce the removed FEA frozen loop's
    exact convergence test verbatim.
    """
    if last_energy is None or last_flux is None:
        return np.inf, np.inf, False
    energy_denom = max(abs(total_energy), abs(last_energy), 1e-12)
    flux_denom = max(abs(total_flux), abs(last_flux), 1e-12)
    energy_error_pct = abs(total_energy - last_energy) / energy_denom * 100.0
    flux_error_pct = abs(total_flux - last_flux) / flux_denom * 100.0
    return energy_error_pct, flux_error_pct, energy_error_pct < rtol_energy_pct and flux_error_pct < rtol_flux_pct


#: THE STRATEGY CLASSES LIVE IN TWO SIBLING MODULES, split 2026-08-22 because this file crossed
#: the 800-line band: `_strategies_picard.py` (Picard family) and `_strategies_newton.py`
#: (Newton family + continuation). Each imports the protocols and helpers above FROM THIS MODULE
#: (a circular import that works because they are defined before this import runs); the public
#: API here is unchanged -- every name in ``__all__`` still resolves off this module.
from motronics.gauss.nonlinear._strategies_newton import (  # noqa: E402
    AndersonStrategy,
    BFGSStrategy,
    BroydenStrategy,
    ContinuationStrategy,
    LBFGSStrategy,
    ModifiedNewtonStrategy,
    NewtonKrylovStrategy,
    NewtonLineSearchStrategy,
    NewtonStrategy,
)
from motronics.gauss.nonlinear._strategies_picard import (  # noqa: E402
    AdaptiveDampedPicardStrategy,
    DampedPicardStrategy,
    NuRelaxationDampedPicardStrategy,
    PicardStrategy,
)

ITERATION_REGISTRY: dict[str, IterationStrategy] = {
    'picard-damped': DampedPicardStrategy(),
    'nu-damped': NuRelaxationDampedPicardStrategy(),
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
