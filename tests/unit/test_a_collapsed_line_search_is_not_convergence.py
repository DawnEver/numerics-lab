"""A line search that shrank its step to nothing has not found the root, and must not say it has.

MEASURED 2026-09-27 on `case_s` fea,tri under `newton-linesearch`: iteration 3 -> 4 moved the
energy 7.59 -> 7.32 J, then iterations 4 and 5 each reported the SAME increment, 1.45e-5, and the
solve returned `converged=True`. That increment is `0.5**12 * |dx|` -- all twelve backtracks failed
to lower the residual, and the strategy took the last, 2.4e-4-scaled step anyway. The convergence
rule was then handed `x_prev -> x_prev + lam dx`, so a COLLAPSED STEP read as a stopped iterate.

The fixed-point question is "is the Newton CORRECTION small", i.e. is the root near; `lam dx` answers
"how far did I dare to move", which is small both at the root and wherever the direction is bad.
So the rule is consulted on the full correction `dx`, which equals `lam dx` whenever `lam = 1`.
"""

from __future__ import annotations

import numpy as np

from numerics_lab.nonlinear import AssembledState, get_strategy

_X_STAR = 10.0


class _BadDirectionSystem:
    """``r(x) = x - x*`` with a tangent of the WRONG SIGN, so every Newton step points away.

    No backtrack can lower ``|r|`` along such a direction, so the line search exhausts itself and
    the accepted step is ``0.5**12`` of a correction that is still ``|x - x*|`` long. The energy is
    held almost flat so the energy half of the rule passes and only the increment half decides.
    """

    is_linear = False

    def initial_guess(self) -> np.ndarray:
        return np.array([9.9])  # 1 % from the root: a 0.5**12 step is 2.5e-6 relative, under the 1e-4 bar

    def assemble(self, x, /) -> AssembledState:
        return AssembledState(system_matrix=None, total_energy=100.0 + 1e-9 * float(np.asarray(x)[0]))

    def residual(self, x, state, /) -> np.ndarray:
        return np.asarray(x, dtype=float) - _X_STAR

    def residual_norm(self, x, state, /) -> float:
        return float(abs(float(np.asarray(x)[0]) - _X_STAR))

    def tangent(self, x, state, /):
        return -1.0  # the true derivative is +1

    def solve_system(self, matrix, rhs, /):
        return np.asarray(rhs, dtype=float) / matrix

    def solve_linear(self, state, /):
        raise NotImplementedError  # a tangent is offered, so the Picard fallback is never taken

    def on_iteration(self, i, state, err, alpha, increment, /) -> None:
        return None


class _GoodDirectionSystem(_BadDirectionSystem):
    """The PLANTED CONTROL: the honest tangent, so the full step lands on the root at ``lam = 1``."""

    def tangent(self, x, state, /):
        return 1.0


def test_a_collapsed_line_search_is_not_reported_as_convergence():
    result = get_strategy('newton-linesearch').solve(_BadDirectionSystem(), max_iter=6)
    x = float(result.x[0])
    assert not result.converged, (
        f'newton-linesearch reported converged=True at x={x:.6f}, {abs(x - _X_STAR):.4f} from the root: '
        'every backtrack failed and the 0.5**12-scaled step was read as a stopped iterate'
    )


def test_an_honest_newton_step_still_converges():
    result = get_strategy('newton-linesearch').solve(_GoodDirectionSystem(), max_iter=6)
    assert result.converged
    assert abs(float(result.x[0]) - _X_STAR) < 1e-9
