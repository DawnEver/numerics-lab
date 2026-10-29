"""A nonlinear solve that cannot finish must RAISE, not run until something else kills it.

MEASURED 2026-08-25, from `output/heavy.log`: `heavy` returned INCONCLUSIVE -- *"no arithmetic to
check: pytest reported no collection or no summary"* -- because
`test_torque_ripple_sweep_discovers_and_runs[case_a]` (the 10th of 16 ripple cases; the log's
progress line reads `xFxxxxxxF` then `Timeout`) hung inside `_umath_linalg.lstsq`, reached via the
sdm rotation sweep. `pytest.mark.timeout(5400)` could not interrupt it: on Windows the thread method
cannot unwind a C-level call, so it dumped stacks and took the PROCESS down. One unbounded solve
therefore cost the whole tier its verdict -- not one red test, but every result in the run.

WHY A RETURN FLAG WOULD NOT DO, and this is the deciding fact rather than a preference:
`IterationResult.converged` already exists, and NOTHING in `hamilton/` reads it -- every occurrence
of the word in that package is prose. A solve that quietly returned `converged=False` would be
"a warning on an unchanged success return", the shape the taste rules name as forbidden, and the
test of it is whether the CALLER can tell. It cannot. So the budget raises.

WHY THE SYSTEM AND NOT THE STRATEGY. The expensive call is `system.solve_linear` (the dense
truncated-SVD least squares, measured at exponent 2.68 on this box). Wrapping the SYSTEM puts the
check exactly where the cost is incurred and leaves all four `IterationStrategy` implementations
and their Protocol untouched -- a budget threaded through four loops would be four places to forget
it. Overshoot is bounded by one call, which is the honest guarantee: the deadline cannot interrupt
a call already inside LAPACK, and pretending otherwise would be the same lie in a new place.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.bound_names import bound

from numerics_lab.nonlinear import IterationBudgetExceeded, with_time_budget
from numerics_lab.nonlinear._seam import _TimeBudgetedSystem

pytestmark = pytest.mark.unit


class _Counting:
    """A minimal `NonlinearSystem` that records how often each expensive entry point ran."""

    is_linear = False

    def __init__(self) -> None:
        self.assembles = 0
        self.solves = 0

    def initial_guess(self) -> np.ndarray:
        return np.zeros(3)

    def assemble(self, x, /):
        self.assembles += 1
        return f'state-for-{np.asarray(x).sum()}'

    def solve_linear(self, state, /) -> np.ndarray:
        self.solves += 1
        return np.ones(3)

    def on_iteration(self, *args) -> None:
        return None


def _elapsed(values):
    """A monotonic clock the test drives, so no arm depends on real time passing."""
    seq = iter(values)
    return lambda: next(seq)


def test_within_budget_nothing_changes() -> None:
    """THE OTHER SIDE OF THE RATCHET: an affordable solve must be untouched, delegation included."""
    inner = _Counting()
    budgeted = with_time_budget(inner, seconds=100.0, clock=_elapsed([0.0, 1.0, 2.0, 3.0, 4.0]))
    state = budgeted.assemble(budgeted.initial_guess())
    out = budgeted.solve_linear(state)
    assert inner.assembles == 1
    assert inner.solves == 1
    assert np.array_equal(out, np.ones(3))
    # Attributes the strategies read must pass straight through, or the wrapper changes the solve
    # it is only supposed to time.
    assert budgeted.is_linear is False


def test_an_expired_budget_raises_on_the_next_expensive_call() -> None:
    """The defect this exists for: the solve stops itself instead of waiting to be killed."""
    inner = _Counting()
    # start 0, first assemble at 1 (cheap, allowed), then the clock is past the deadline.
    budgeted = with_time_budget(inner, seconds=10.0, clock=_elapsed([0.0, 1.0, 50.0]))
    budgeted.assemble(np.zeros(3))
    with pytest.raises(IterationBudgetExceeded) as excinfo:
        budgeted.solve_linear('state')
    assert inner.solves == 0, 'the expensive call must not be entered once the budget is gone'
    assert '10' in str(excinfo.value), f'the refusal must name the budget: {excinfo.value}'
    assert '50' in str(excinfo.value) or '49' in str(excinfo.value), (
        f'the refusal must name the MEASURED elapsed time, not just the ceiling: {excinfo.value}'
    )


def test_the_budget_cannot_interrupt_a_call_already_running() -> None:
    """The guarantee stated honestly: overshoot is bounded by ONE call, never zero.

    A test asserting instant cancellation would be asserting something the mechanism cannot do --
    LAPACK holds the GIL-free thread and no Python-level check reaches inside it. Pinning the
    ACTUAL bound is what keeps the docstring from promising more than the code delivers.
    """
    inner = _Counting()
    budgeted = with_time_budget(inner, seconds=10.0, clock=_elapsed([0.0, 1.0, 2.0, 99.0]))
    budgeted.assemble(np.zeros(3))
    budgeted.solve_linear('state')  # entered while still inside budget -- runs to completion
    assert inner.solves == 1
    with pytest.raises(IterationBudgetExceeded):
        budgeted.assemble(np.zeros(3))


def test_no_budget_is_a_pass_through() -> None:
    """`None` must be today's behaviour EXACTLY, so opting in is the only thing that changes a solve."""
    inner = _Counting()
    assert with_time_budget(inner, seconds=None) is inner


def test_every_declared_expensive_call_is_actually_budgeted() -> None:
    """The scope claim, held to its own declaration instead of trusted.

    The first version budgeted only `assemble` and `solve_linear` -- the two the Picard default
    happens to use -- which left the whole Newton family unbudgeted: a hole exactly where a caller
    who switched strategy would not think to look. `BUDGETED_CALLS` is the declaration; this plants
    an expired budget and calls each name through the REAL wrapper, so a method added to the tuple
    without a check (or a check dropped from a method still in it) reddens here.
    """
    args_for = {
        'assemble': (np.zeros(3),),
        'solve_linear': ('state',),
        'solve_system': ('matrix', np.zeros(3)),
        'solve_secant': ('state', np.zeros(3)),
        'tangent': (np.zeros(3), 'state'),
    }
    assert set(args_for) == set(_TimeBudgetedSystem.BUDGETED_CALLS), (
        'this test and the declaration disagree about which calls exist'
    )
    for name, args in args_for.items():
        budgeted = with_time_budget(_Counting(), seconds=1.0, clock=_elapsed([0.0, 99.0]))
        with pytest.raises(IterationBudgetExceeded, match=name):
            bound(_TimeBudgetedSystem.__module__, f'_TimeBudgetedSystem.{name}')(budgeted, *args)


def test_the_error_carries_the_numbers_a_reader_has_to_act_on() -> None:
    """A refusal naming no measurement sends its reader to re-derive what the guard already knew."""
    inner = _Counting()
    budgeted = with_time_budget(inner, seconds=2.5, clock=_elapsed([0.0, 7.5]))
    with pytest.raises(IterationBudgetExceeded) as excinfo:
        budgeted.assemble(np.zeros(3))
    err = excinfo.value
    assert err.budget_s == pytest.approx(2.5)
    assert err.elapsed_s == pytest.approx(7.5)
