"""The SEAM every nonlinear strategy iterates across: the system protocol, the assembled state and
result, the time budget, and the convergence rule (energy, residual, stagnation, rising increment).

Split out of the package `__init__` on 2026-09-29 as the leaf the strategy modules import, so the
package no longer needs a circular import after its own definitions to reach them.
"""

from __future__ import annotations

__all__ = [
    'AssembledState',
    'IterationBudgetExceeded',
    'IterationResult',
    'IterationStrategy',
    'NewtonTangentUnavailable',
    'NonlinearSystem',
    'RawStateSystem',
    'WrappedNonlinearSystem',
    'increment_ratio',
    'run_solve_budget_s',
    'set_run_solve_budget_s',
    'unwrap_system',
    'with_time_budget',
]

import itertools
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, overload, runtime_checkable

import numpy as np

# Damped-Picard parameters (moved verbatim from mec/_sparse.py so the extracted loop is identical).
_ALPHA_MIN = 0.1  # floor for the damped under-relaxation factor
_ALPHA_GROW = 1.5  # damped-Picard recovery factor after a calm window
_RTOL_ENERGY_PCT = 0.01  # convergence: total-energy change below this fraction of a percent
#: Consecutive UNDISTURBED steps (no energy-error growth, no energy-delta sign flip) before the
#: damped Picard loop grows ``alpha`` back. The ratchet used to be one-way, so a single early
#: overshoot damped the whole smooth tail: measured on case_a mec:quad, alpha stuck at
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
        material_change: relative change of the IRREVERSIBLE material state this assembly's trial
            implies (the caller's material loop); ``0`` for a system whose materials are all
            reversible. The material term of :func:`_has_converged`.
        local_values: snapshots of per-element fields, one array per independently
            scaled quantity the caller chooses; Newton checks the worst element.

    """

    system_matrix: object
    total_energy: float
    payload: object = None
    material_change: float = 0.0
    #: Per-element independently dimensioned fields checked by the local Newton criterion.
    local_values: tuple[np.ndarray, ...] = ()


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
    that cannot supply a Jacobian, and NewtonStrategy then RAISES :class:`NewtonTangentUnavailable`.
    """

    @property
    def is_linear(self) -> bool: ...

    # Every parameter below is positional-only (``/``): every real caller (the strategies in this
    # module) invokes them positionally, and the two implementers name their iterate differently
    # (MEC's ``phi_loop``, FEA's ``A_mat``) because it carries different physics -- a loop flux vs.
    # a nodal vector potential. Requiring a shared PARAMETER NAME here would be a "declaration that
    # lies" for whichever implementer's name is more honest; positional-only makes the Protocol
    # match how it is actually consumed instead (found via pyright while composing the P3 seam --
    # `hamilton/discretizations/evaluation.py` -- surfaced this pre-existing mismatch on both `_ReluctanceSystem` and
    # `GalerkinSystem`).
    def initial_guess(self) -> np.ndarray: ...
    def assemble(self, x: np.ndarray, /) -> AssembledState: ...
    def solve_linear(self, state: AssembledState, /) -> np.ndarray: ...

    def residual(self, x: np.ndarray, state: AssembledState, /) -> np.ndarray:
        """Residual r(x) whose root is the solution (Newton drives ``tangent @ dx = -r``)."""
        ...

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
        """Optional Newton tangent operator; ``None`` makes a Newton strategy raise, never step Picard.

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
        need a SECANT solve of an arbitrary rhs require :class:`SecantSystem`, because the
        secant ``system_matrix`` is not always in ``solve_system``'s
        accepted form (FEA's ``solve_system`` takes the reduced-tangent bundle, not a raw matrix).
        """
        ...

    def on_iteration(self, i: int, state: AssembledState, err: float, alpha: float, increment: float, /) -> None:
        """Optional per-iteration hook (logging); default no-op.

        ``increment`` is ``||x_next - x|| / ||x||``, the half of the convergence rule the energy
        error cannot express. Both halves are passed because a hook that saw only ``err`` reported
        a capped solve and a converged one identically.
        """


@runtime_checkable
class SecantSystem(Protocol):
    """Optional arbitrary-rhs secant action, including the equation's physical constraints."""

    def solve_secant(self, state: AssembledState, rhs: np.ndarray, /) -> np.ndarray: ...


@runtime_checkable
class SourceScalableSystem(Protocol):
    """Optional source rescaling, not a requirement on Picard/Newton equations."""

    def scale_source(self, lam: float, /) -> NonlinearSystem:
        """Return a copy of this system with its SOURCE scaled by ``lam`` (continuation stepping).

        Optional -- only the ``continuation`` strategy calls it. An equation that cannot cheaply
        rescale its source (conductor current + magnet Br) raises :class:`NotImplementedError`, and
        the strategy degrades to solving the full-source problem directly. ``lam=1`` returns the
        original (full-source) system.
        """
        ...


@runtime_checkable
class ResidualReferencedSystem(Protocol):
    """Optional source norm used to normalize the actual residual."""

    def rhs_norm(self, state: AssembledState, /) -> float: ...


@runtime_checkable
class FactorizableSystem(Protocol):
    """Optional equation-owned factorization of its opaque tangent bundle."""

    def factorize(self, matrix, /) -> Callable[[np.ndarray], np.ndarray]: ...


@runtime_checkable
class ResidualStoppingStrategy(Protocol):
    """Optional declared stopping rule; absence means the energy rule."""

    @property
    def stop_rule(self) -> str: ...


@runtime_checkable
class RawStateSystem(Protocol):
    """Optional pre-boundary-fold assembly capability required by ``coefficient-damped``."""

    def raw_state(self, x: np.ndarray, /) -> tuple: ...
    def fold_raw(self, s_row, s_col, s_val, t_mat, total_energy, /) -> AssembledState: ...


@runtime_checkable
class WrappedNonlinearSystem(Protocol):
    """A transparent nonlinear wrapper with an explicitly reachable wrapped system."""

    @property
    def unwrapped(self) -> NonlinearSystem: ...


@overload
def unwrap_system(system: NonlinearSystem) -> NonlinearSystem: ...


@overload
def unwrap_system(system: object) -> object: ...


def unwrap_system(system: object) -> object:
    """Return the concrete system beneath any explicit transparent wrappers."""
    while isinstance(system, WrappedNonlinearSystem):
        system = system.unwrapped
    return system


def has_secant_system(system: object) -> bool:
    """Whether the concrete equation and its transparent carrier expose secant solves."""
    return isinstance(unwrap_system(system), SecantSystem) and isinstance(system, SecantSystem)


def require_secant_system(system: NonlinearSystem) -> SecantSystem:
    """Check the actual capability, then delegate through wrappers (including their budgets)."""
    if not has_secant_system(system):
        msg = f'{type(unwrap_system(system)).__name__} cannot run quasi-Newton without solve_secant'
        raise TypeError(msg)
    assert isinstance(system, SecantSystem)
    return system


class NewtonTangentUnavailable(TypeError):
    """A Newton-family strategy was asked to drive a system that supplies no tangent.

    RAISED, NEVER A SILENT PICARD STEP (``UNSUPPORTED-RAISES``). The Newton strategy used to fall
    back to ``solve_linear`` whenever ``tangent`` returned ``None``, so a run that asked for Newton
    on a tangent-less system got a Picard iteration under Newton's name -- iteration counts, rates
    and a strategy label all describing an algorithm that never ran. Every production system now
    supplies a tangent (FEA and MEC exact, SDM a Jacobian-free directional derivative, the spline
    space exact); a system that cannot is asked to run a fixed-point strategy by name instead.
    """

    def __init__(self, system: object, strategy: str) -> None:
        inner = unwrap_system(system)
        super().__init__(
            f'{type(inner).__name__} returned no tangent, so {strategy!r} cannot run on it. Newton needs '
            f'the Jacobian of the residual; supply `tangent` on the system, or select a fixed-point '
            f"strategy by name (e.g. 'picard-damped', 'anderson')."
        )


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

    @property
    def unwrapped(self):
        """The system this one wraps -- what a diagnostic should NAME.

        Defined here rather than left to `_inner` because a caller reaching through a private
        attribute is a caller that breaks when the private moves, and every diagnostic that wants
        to say WHICH system lacks a capability needs this. The wrapper delegates missing attributes
        too, so its own type name never answers that question.
        """
        return self._inner

    @property
    def is_linear(self) -> bool:
        return bool(self._inner.is_linear)

    def initial_guess(self):
        return self._inner.initial_guess()

    def residual(self, x, state, /):
        return self._inner.residual(x, state)

    def residual_norm(self, x, state, /):
        return self._inner.residual_norm(x, state)

    def on_iteration(self, i, state, err, alpha, increment, /):
        return self._inner.on_iteration(i, state, err, alpha, increment)

    def scale_source(self, lam, /):
        return self._inner.scale_source(lam)

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
#: ``n_positions * budget < the test's own timeout``. `case_a` declares ``n_positions = 7`` and
#: `test_torque_ripple_sweep_discovers_and_runs` carries ``@pytest.mark.timeout(5400)``:
#: 7 x 500 = 3500 s fits. At 900 s it is 6300 s -- over the timeout, so on Windows the firing
#: `pytest-timeout` ``os._exit``s the WORKER and every result queued behind it is lost. A budget too
#: GENEROUS is therefore not the safe direction; it is no budget at all.
#:
#: THE MEASUREMENTS IT SITS BETWEEN: ~200 s per solve for the cases that complete, and 15-17 min for
#: the sdm case that hangs. MEASURED 2026-08-27, the reason this is no longer sdm-only: probing the
#: five ripple cases whose results were MISSING from a truncated heavy run, `case_r`
#: (FEA-Galerkin) and `case_h` (MEC Picard) both ran past 900 s standalone. Neither is an sdm
#: solve, so the sdm-only budget could not have reached them.
#:
#: WHAT THIS DELIBERATELY COSTS: a solve that legitimately needs longer now RAISES. That trade is
#: taken knowingly -- a tier that returns INCONCLUSIVE loses EVERY result, so one named red leg is
#: strictly more information than none. A caller that knows its solve is genuinely long passes its
#: own ``seconds``.
DEFAULT_SOLVE_BUDGET_S = 500.0

#: The run-scoped override of :data:`DEFAULT_SOLVE_BUDGET_S`, or ``None`` for the default. A
#: PROCESS property, like the FEMM call ceiling: the CLI boundary applies ``[solve] budget`` (``--set`` over the
#: case's table) once per run (by the downstream CLI), and every
#: driver not handed its own ``seconds`` reads :func:`run_solve_budget_s`.
#:
#: DECLARED, NOT DERIVED FROM MESH SIZE. The default above is priced by the TIER's arithmetic, not
#: by a solve's cost; a budget that grew with the cell count would grow exactly where that bound is
#: tightest, and its per-cell rate would be a property of whichever machine measured it.
_RUN_SOLVE_BUDGET_S: dict[str, float | None] = {'seconds': None}


def set_run_solve_budget_s(seconds: float | None) -> None:
    """Set the run's solve budget in seconds (> 0); ``None`` restores :data:`DEFAULT_SOLVE_BUDGET_S`."""
    if seconds is not None and not float(seconds) > 0.0:
        msg = f'a solve budget must be > 0 s, got {seconds!r} (a zero budget refuses the first call)'
        raise ValueError(msg)
    _RUN_SOLVE_BUDGET_S['seconds'] = None if seconds is None else float(seconds)


def run_solve_budget_s() -> float:
    """The budget a driver applies when its caller passes none: the run's override, else the default."""
    held = _RUN_SOLVE_BUDGET_S['seconds']
    return DEFAULT_SOLVE_BUDGET_S if held is None else held


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
    *,
    material_change: float,
    residual_ratio: float | None = None,
) -> bool:
    """The ONE convergence rule: the equation is satisfied (or its energy settled), the iterate has
    stopped moving, AND the irreversible material state has stopped changing.

    TWO FIRST HALVES, ONE SHARED REST (2026-09-25, the Newton unification). A strategy that holds
    the RESIDUAL -- the Newton family, which forms it every step anyway -- passes
    ``residual_ratio = ||r(x_next)|| / max(||r(x_0)||, ||rhs||)`` and is judged on it: the root of
    the equation is the thing being sought, so the residual is the direct measure and the energy
    change is only a proxy for it. The denominator is the right-hand side the solve has to cancel
    (the seed residual, or the source norm when a warm start makes that smaller), so the ratio is
    scale-free in every unit and every family. A fixed-point
    strategy that never forms a residual passes ``None`` and keeps the energy half described below.
    The increment and material halves are shared, so neither can call a wandering iterate converged.

    THE MATERIAL TERM (D4, 2026-09-25). A reversible law (steel B-H) is a function of the iterate,
    so the iterate test already sees it settle and its change is ``0``. A one-way remanence clamp
    is NOT: its state is history, and an iterate can sit still for a step while the clamp is still
    walking down the recoil line. ``material_change`` is that state's relative change on this
    step, held to the same bar as the increment. It is REQUIRED (keyword, no default) so a strategy
    cannot consult the rule without deciding what it passes.

    A FLAT ENERGY IS NOT A FIXED POINT. `total_energy` can sit on a near-flat plateau that is not
    the solution -- measured on a saturating-BH case, which read 0.00% energy change for 20+
    iterations at an energy level ~14x the physical one before escaping and reaching the genuine
    fixed point ~15 iterations later. On an energy-only test that is a `converged=True` return at a
    wrong answer, and the caller cannot tell.

    That failure mode is a property of the ENERGY CRITERION, not of any one strategy. It was
    previously guarded inside :class:`DampedPicardStrategy` alone, and even there only behind
    ``if ratcheted`` -- so it did not fire on an undamped run of the very strategy that owned it,
    and the other eleven strategies never had it. Hence one function.

    "CONSULTED BY ALL OF THEM" IS A CLAIM, AND IT WAS FALSE FOR TWO YEARS OF READERS UNTIL
    2026-09-06. `coefficient-damped` -- the FEA DEFAULT -- judged on its own energy/flux pair and never
    called this, and `continuation` is an outer driver that delegates. The first is now fixed and
    the second is exempt by kind; both are DECLARED in
    `tests/architecture/domain/test_every_iteration_strategy_consults_the_one_rule.py`, which reads
    the registry rather than a list, so the sentence above is a fact a test enforces and not prose.

    The second half is the direct fixed-point test ``x_next == x``, immune to a flat-but-wrong
    energy plateau, on the step the strategy is ABOUT to take. It costs nothing: every strategy has
    already computed ``x_next`` by the time it asks. ``rtol_energy_pct`` is a PERCENT and the
    increment is a FRACTION, so the ``/ 100`` is the unit conversion between them, not a second,
    tighter tolerance.
    """
    rtol = rtol_energy_pct / 100.0
    settled = residual_ratio < rtol if residual_ratio is not None else err < rtol_energy_pct
    return settled and increment_ratio(x, x_next) < rtol and float(material_change) < rtol


#: Consecutive steps of the stagnation window. MEASURED (2026-09-01): on
#: `case_b` newton's energy error passes rtol by iteration ~12 while the increment
#: pins at ~3.57e-4 -- the stationary limit of the hanging-fold update -- so `_has_converged`'s
#: increment floor (rtol/100 = 1e-4, 2-4.5x tighter than the stationary increment on every measured
#: case) is never met and the solve burns the whole 100-step cap: ~85 wasted iterations, ~123 s.
#: K must OUTLAST the documented false-plateau hazard -- the saturating-BH plateau that read 0.00%
#: energy change for 20+ iterations (`_has_converged`'s docstring) -- so K = 25; it fires on
#: case_b newton at ~iteration 39.
_STAGNATION_WINDOW = 25

#: How flat the increment must be across the window to count as STATIONARY (max/min). The false
#: plateau's increment WANDERS, so this band -- not the energy half -- is what refuses it.
_STAGNATION_MAX_MIN_RATIO = 2.0


class _StagnationWatch:
    """The stagnation break, consulted NEXT TO `_has_converged` -- never instead of it.

    `_has_converged` is the FAST break: energy settled AND the increment below rtol/100, checked
    on the step about to be taken. It is unreachable when the increment has a stationary floor
    ABOVE rtol/100 that is not the iterate wandering -- measured above, the hanging-fold update
    pins it at ~3.5e-4. This watch is the STAGNATION break for exactly that shape: the energy
    error has been under rtol for `_STAGNATION_WINDOW` CONSECUTIVE steps AND the increment is
    stationary over the same window (max/min <= `_STAGNATION_MAX_MIN_RATIO`).

    The energy half is NOT weakened -- every window step must independently satisfy `err < rtol`,
    the same bar the fast break sets. What replaces the increment FLOOR is the increment
    STATIONARITY test, and the wandering-increment false plateau fails it by construction: a
    20-step 0.00% energy plateau whose increment moves by more than the 2x band never fires this
    break, which is the hazard `_has_converged`'s docstring measured and the reason the window is
    25, not 5.

    A sliding window, so "K consecutive" costs no explicit reset: one step over rtol (or one
    wandering increment) sits INSIDE the window and blocks the break until it has aged out.

    STATIONARITY IS A BAND *AND* A TREND, and it used to be only the band -- see `observe` for the
    measured creep that slipped through: an increment rising 0.7% a step stays inside a 2x band for
    a hundred iterations while diverging, and the break called it converged.
    """

    def __init__(self, rtol_energy_pct: float, window: int = _STAGNATION_WINDOW) -> None:
        self._rtol = float(rtol_energy_pct)
        self._window = int(window)
        self._errs: deque[float] = deque(maxlen=self._window)
        self._increments: deque[float] = deque(maxlen=self._window)

    def observe(self, err: float, increment: float, material_change: float, /) -> bool:
        """Record one iteration's pair; True iff the full window is settled AND stationary.

        A stationary INCREMENT is no escape while the irreversible material state is still moving:
        ``material_change`` must be under the same bar :func:`_has_converged` holds it to.
        """
        if float(material_change) >= self._rtol / 100.0:
            self._errs.clear()
            self._increments.clear()
            return False
        self._errs.append(float(err))
        self._increments.append(float(increment))
        if len(self._errs) < self._window:
            return False
        if not all(e < self._rtol for e in self._errs):
            return False
        lo = min(self._increments)
        hi = max(self._increments)
        if hi > _STAGNATION_MAX_MIN_RATIO * max(lo, 1e-300):
            return False
        # STATIONARY IS NOT THE SAME AS INSIDE A BAND, and the max/min test alone cannot tell them
        # apart: an increment CREEPING UP by 0.7% a step stays inside a 2x band for 100 iterations
        # while diverging the whole time. MEASURED 2026-09-06 on `case_a mec:quad` --
        # 1.91e-3 -> 2.05e-3 across ten steps, band ratio 1.07, and the break fired and returned
        # `converged = True` at 20x the increment bar. This is the "declaration that lies" shape in
        # the docstring above: it says STATIONARY and tested only a band.
        half = len(self._increments) // 2
        early = sum(itertools.islice(self._increments, half)) / half
        late = sum(itertools.islice(self._increments, len(self._increments) - half, None)) / half
        return late <= early


#: Consecutive steps of a RISING increment before the iterate is damped on the increment alone.
#: Symmetric with :data:`_ALPHA_CALM_STEPS` on purpose: the same amount of evidence that licenses a
#: recovery licenses a damping, so neither direction is cheaper to trigger than the other. One step
#: is not enough -- an increment jitters -- and the measured blow-out rose monotonically for ten.
_INCREMENT_RISE_STEPS = 3


class _IncrementTrend:
    """What the ITERATE says about the step size -- the channel the ENERGY goes deaf on.

    THE ONE REGISTRY for "what should alpha do?" when the energy has nothing to say, consulted by
    every damped Picard leg instead of each spelling its own half of it.

    Both alpha SHRINK ratchets are gated on ``meaningful`` (``err >= rtol_energy_pct``) and that is
    right and measured: under the tolerance the energy delta's SIGN is floating-point noise, and a
    noise-driven shrink is what pinned alpha at 0.108 on case_a. But the recovery leg then
    read the SAME deaf channel and drew the opposite conclusion from the same silence, so on an
    energy plateau alpha grew every ``_ALPHA_CALM_STEPS`` with nothing able to object.

    MEASURED 2026-09-06 through ``downstream run cases/case_a --select
    mec,quad --max-num-iter 600``, the `[sparse]` leg on `picard-damped`. At iteration 119 the
    increment was falling steadily toward the 1e-4 bar (2.03e-4 -> 1.95e-4 -> 1.89e-4) at alpha
    0.137. Recovery raised alpha to 0.205 then 0.308 and the increment blew out ~20x (3.70e-4 ->
    7.59e-4 -> 1.40e-3 -> 3.91e-3) until the damage finally reached the ENERGY and re-armed a
    shrink. At the production cap of 100 the solve stops inside that cycle and reports
    ``converged = False`` on an iterate that was, twenty iterations earlier, nearly converged.

    BOTH DIRECTIONS, and blocking the growth alone was measured to be half a fix: with only
    :meth:`permits_growth` the same run parked at alpha 0.150 with the increment creeping UP
    (1.91e-3 -> 2.05e-3 over ten steps) -- no longer cycling, still not contracting, and nothing
    left that could move alpha, because the energy is deaf and the recovery gate only ever refuses.
    So a SUSTAINED rise damps: :meth:`claim_damping` is the increment's own overshoot signal, the
    exact counterpart of ``err > prev_err`` for a plateau where ``err`` cannot express it.

    ASYMMETRIC EVIDENCE ON PURPOSE. One non-contracting step is enough to REFUSE a growth (refusing
    costs a solve nothing but iterations at the current step size); ``_INCREMENT_RISE_STEPS``
    consecutive rises are required to force a shrink (shrinking is what over-damping is made of).
    """

    def __init__(self) -> None:
        self._last: float | None = None
        self._rising = 0

    def observe(self, increment: float, /) -> None:
        """Record this step's increment. Called once per iteration, after it is computed."""
        increment = float(increment)
        self._rising = self._rising + 1 if self._last is not None and increment > self._last else 0
        self._last = increment

    def permits_growth(self) -> bool:
        """May alpha grow? Only while the last step did not move the iterate MORE than the one before.

        No history reads as permission: refusing a recovery for want of evidence would throttle
        every solve's first steps, which is the crawl the recovery leg exists to remove.
        """
        return self._rising == 0

    def claim_damping(self) -> bool:
        """Has the increment risen for ``_INCREMENT_RISE_STEPS`` running? CONSUMES the demand.

        Consuming is what makes one sustained rise cost ONE shrink rather than one per step for as
        long as the rise lasts -- an unconsumed demand would drive alpha to its floor in three more
        iterations, which is precisely the over-damping the `meaningful` guard was added to stop.
        """
        if self._rising < _INCREMENT_RISE_STEPS:
            return False
        self._rising = 0
        return True


#: How small the SHAPE (mean-removed) part of an increment must be, RELATIVE TO THE INCREMENT, for
#: :func:`increment_ratio` to read the increment as pure gauge drift and report the shape instead.
#:
#: DERIVED FROM BOTH MEASURED POPULATIONS, not fitted to either. The gauge-drift defect measured
#: mean-removed 1.0e-06 against a raw 3.5e-02 -- a shape fraction of ~3e-05, which is 33x below
#: this line. A genuine 1 % shape change riding on a large constant offset measures 2.1e-02, which
#: is 21x above it. Nothing observed lands between, and the two margins are what make the number a
#: separator rather than a tuned constant.
#:
#: STATED AS A FRACTION OF THE INCREMENT rather than as a "constant fraction near 1", because the
#: two differ exactly where it matters: a constant fraction of 0.9999 and one of 0.99978 are 1.7e-04
#: apart, so a threshold between them would sit inside the noise of both measurements.
_GAUGE_ONLY_SHAPE_FRACTION = 1e-3


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
    dx = np.asarray(x_next) - np.asarray(x)
    dx_norm = float(np.linalg.norm(dx))
    if dx_norm == 0.0:
        return 0.0  # the degenerate edge, where the constant fraction is undefined
    if dx.size < 2:
        # A ONE-COMPONENT ITERATE HAS NO SHAPE TO SEPARATE FROM ITS GAUGE. The relaxation below
        # splits `dx` into a constant mode and a mean-removed shape; for n = 1 the shape subspace
        # is {0}, so `dx - dx.mean()` is EXACTLY zero for every possible increment and the test
        # `shape_norm / dx_norm < _GAUGE_ONLY_SHAPE_FRACTION` passes unconditionally. The function
        # then returned 0.0 -- "the iterate has stopped moving" -- on a step of any size.
        # MEASURED on the plateau fixture: `increment_ratio([5.0], [7.5])` returned 0.0, so
        # `_has_converged` fired at x = 7.5 against a fixed point of 10.0 and six strategies
        # reported `converged=True` three iterations into a solve that was still crossing the
        # plateau their own tests were written to refuse.
        # This is not a tolerance: the decomposition carries no information at n = 1, so the raw
        # ratio is the only thing there is to report.
        return dx_norm / x_scale
    # A UNIFORM SHIFT IN A GAUGE-FREE POTENTIAL IS NOT MOVEMENT. In 2-D magnetostatics the unknown
    # is the vector potential `A`, defined only up to an additive constant unless a Dirichlet row
    # pins it; that constant carries no energy and no `B`, so an iterate drifting along it has
    # reached the fixed point of every physical quantity while `||dx|| / ||x||` goes on reporting a
    # solution that will not settle. MEASURED 2026-09-06 on `case_c`, 400
    # consecutive increments instrumented inside the newton and picard strategies: raw increment
    # 3.5e-02..2.7e-01 against a criterion needing < 1e-4, mean-removed 1.0e-06..1.8e-06, constant
    # fraction 1.0000 to four decimals on all 400, energy flat at 185.69 to six digits. All four
    # one-hot solves ran to the 100-iteration cap and the matrix they produced was refused by
    # `require_physical_inductance` (eigenvalue -5.76e-05 H at 364.177 A; 19 % reciprocity residual
    # at 36.4 A). The solve was physically converged; the CRITERION was measuring a gauge mode.
    #
    # ITS TWIN IS `gauss.linear.constrained.null_space_pinning.gauge_pin_constant_mode`, and the two are halves of one fix:
    # that one stops the LINEAR solve from having no reference at all, this one stops the NONLINEAR
    # criterion from counting the resulting drift as movement.
    #
    # THE RELAXATION IS NARROW, and that is the floor on it. Subtracting the mean unconditionally
    # would be the wrong shape -- a real change can carry a nonzero mean (an anchored field growing
    # everywhere) and removing it would under-report genuine movement. So it fires only when the
    # SHAPE part of the increment is negligible against the increment itself, and the moment a
    # shape change enters, the raw ratio is what answers.
    #
    # AND IT NEEDS A PROPER SUBSPACE TO SUBTRACT, which is the guard below (2026-09-22). Removing
    # the mean removes ONE dimension of `dx`. At `dx.size == 1` that is every dimension: the
    # constant mode IS the whole space, `dx - dx.mean()` is identically the zero vector for any
    # increment whatever, and this function returned 0.0 for a 1-DOF solve that had just moved from
    # 0 to 20. `_has_converged` ANDs on that number, so EVERY single-unknown nonlinear solve
    # reported converged on its first iteration -- "the iterate has stopped moving" derived from a
    # measurement that cannot express movement.
    #
    # STRUCTURALLY ZERO IS NOT NEGLIGIBLY SMALL, and the test above cannot tell them apart: the
    # relaxation's own warrant is that it "fires only when the SHAPE part of the increment is
    # negligible AGAINST THE INCREMENT ITSELF", a statement about a ratio of two real quantities.
    # When the shape part does not exist, there is no ratio to be small.
    if dx.size > 1:
        shape_norm = float(np.linalg.norm(dx - dx.mean()))
        if shape_norm / dx_norm < _GAUGE_ONLY_SHAPE_FRACTION:
            return shape_norm / x_scale
    return dx_norm / x_scale
