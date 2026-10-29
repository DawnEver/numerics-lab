"""Pluggable nonlinear-iteration strategies, decoupled from the governing equation.

Design ``.claude/memory/2026/07/09/design-solver-iteration-decoupling.md`` (a179884): the
governing equation knows how to ASSEMBLE (given the current iterate, produce the system matrix,
the energy, and whatever payload the caller returns); the STRATEGY knows how to ITERATE (Picard,
damped-Picard, Newton). They meet at the small :class:`NonlinearSystem` protocol, so a solver
picks its nonlinear method without the equation code changing.

First principle (``AGENTS.md``): IMPLEMENT every method and characterize it -- never exclude one
because an experiment showed it inferior. Plain Picard's historical divergence on saturating B-H
is a labelled datum, not a hidden exclusion. The default is Newton with a backtracking line search
(user ruling 2026-09-25, every family); the damped-Picard extracted verbatim from the old MEC/FEA
inline loops stays selectable by name, as do the others.

This module drives the iterate only; it never assembles physics or touches a mesh.
"""

from __future__ import annotations

from numerics_lab.nonlinear._seam import (
    DEFAULT_SOLVE_BUDGET_S,
    AssembledState,
    IterationBudgetExceeded,
    IterationResult,
    IterationStrategy,
    NewtonTangentUnavailable,
    NonlinearSystem,
    RawStateSystem,
    ResidualStoppingStrategy,
    has_secant_system,
    increment_ratio,
    run_solve_budget_s,
    set_run_solve_budget_s,
    unwrap_system,
    with_time_budget,
)
from numerics_lab.nonlinear._strategies_newton import (
    ContinuationStrategy,
    HeldFactorNewtonStrategy,
    ModifiedNewtonStrategy,
    NewtonKrylovStrategy,
    NewtonLineSearchStrategy,
    NewtonStrategy,
)
from numerics_lab.nonlinear._strategies_picard import (
    CoefficientDampedPicardStrategy,
    DampedPicardStrategy,
    PicardStrategy,
)
from numerics_lab.nonlinear._strategies_quasi import AndersonStrategy, BFGSStrategy, BroydenStrategy, LBFGSStrategy

__all__ = [
    'DEFAULT_ITERATION_STRATEGY',
    'DEFAULT_SOLVE_BUDGET_S',
    'ITERATION_REGISTRY',
    'AndersonStrategy',
    'AssembledState',
    'BFGSStrategy',
    'BroydenStrategy',
    'CoefficientDampedPicardStrategy',
    'ContinuationStrategy',
    'DampedPicardStrategy',
    'HeldFactorNewtonStrategy',
    'IterationBudgetExceeded',
    'IterationResult',
    'IterationStrategy',
    'LBFGSStrategy',
    'ModifiedNewtonStrategy',
    'NewtonKrylovStrategy',
    'NewtonLineSearchStrategy',
    'NewtonStrategy',
    'NewtonTangentUnavailable',
    'NonlinearSystem',
    'PicardStrategy',
    'get_strategy',
    'increment_ratio',
    'run_solve_budget_s',
    'set_run_solve_budget_s',
    'strategy_requires',
    'strategy_runs_on',
    'strategy_stops_on_residual',
    'with_time_budget',
]


ITERATION_REGISTRY: dict[str, IterationStrategy] = {
    'picard-damped': DampedPicardStrategy(),
    'coefficient-damped': CoefficientDampedPicardStrategy(),
    'picard': PicardStrategy(),
    'newton': NewtonStrategy(),
    'anderson': AndersonStrategy(),
    'modified-newton': ModifiedNewtonStrategy(),
    'quasi-newton': BroydenStrategy(),
    'bfgs': BFGSStrategy(),
    'l-bfgs': LBFGSStrategy(),
    'newton-linesearch': NewtonLineSearchStrategy(),
    'newton-held-factor': HeldFactorNewtonStrategy(),
    'newton-krylov': NewtonKrylovStrategy(),
    'continuation': ContinuationStrategy(),
}

_STRATEGY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    'coefficient-damped': ('raw_state', 'fold_raw'),
    'quasi-newton': ('solve_secant',),
    'bfgs': ('solve_secant',),
    'l-bfgs': ('solve_secant',),
}


def strategy_requires(name: str) -> tuple[str, ...]:
    """The system attributes strategy ``name`` needs, or ``()`` when it needs nothing special.

    Quasi-Newton needs arbitrary-rhs secant solves; `coefficient-damped` needs pre-fold assembly.
    Continuation can run without source scaling by solving the full source directly.
    These are capability boundaries, not convergence properties.
    """
    ITERATION_REGISTRY[name]
    return _STRATEGY_REQUIREMENTS.get(name, ())


def strategy_stops_on_residual(name: str) -> bool:
    """Whether strategy ``name`` judges convergence on the equation's residual (the Newton family).

    Declared on the strategy (``stop_rule = 'residual'``), not listed here, so a Newton variant
    registered later answers by construction. Every other strategy stops on the energy change, and
    the two bars are different quantities with different defaults. An unknown name raises.
    """
    strategy = get_strategy(name)
    return isinstance(strategy, ResidualStoppingStrategy) and strategy.stop_rule == 'residual'


def strategy_runs_on(name: str, system: object) -> bool:
    """Whether ``system`` supplies everything strategy ``name`` declares it needs.

    The question `test_iteration_method_matrix` has to ask before it generates a cell. It is asked
    of the SYSTEM rather than of the discretization's name because a budget wrapper, a decorator or
    a future system type all answer correctly by construction, while a list of solver names would
    have to be maintained alongside the registry it describes.
    """
    requirements = strategy_requires(name)
    if not requirements:
        return True
    if requirements == ('solve_secant',):
        return has_secant_system(system)
    return requirements == ('raw_state', 'fold_raw') and isinstance(unwrap_system(system), RawStateSystem)


#: The strategy ``get_strategy(None)`` resolves to: Newton, in every family (user ruling 2026-09-25,
#: extended from FEA to MEC and SDM the same day). A FAMILY's default is a config field read in
#: ``hamilton`` (``core.<family>_iteration_strategy``) -- ``gauss`` reads no configuration -- and
#: every one of those fields defaults to this name; this constant is what a caller that names no
#: family (a unit fixture, a study with no field of its own) gets.
DEFAULT_ITERATION_STRATEGY = 'newton'


def get_strategy(name: str | None) -> IterationStrategy:
    """Resolve an iteration-strategy name; ``None`` -> :data:`DEFAULT_ITERATION_STRATEGY` (Newton)."""
    if name is None:
        return ITERATION_REGISTRY[DEFAULT_ITERATION_STRATEGY]
    try:
        return ITERATION_REGISTRY[name]
    except KeyError as e:
        msg = f'unknown iteration strategy {name!r}; known: {sorted(ITERATION_REGISTRY)}'
        raise KeyError(msg) from e
