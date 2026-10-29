"""A line-search trial the problem cannot evaluate is a REJECTED trial, not a failed march.

The condensed field-circuit scheme evaluates its outer residual through an inner field root; a
backtracking trial far from the root can leave that inner Newton without a root, measured on the
case_a six-terminal default run (inner residual 160.95 at an outer trial current). The problem says
so by raising :class:`TrialNotEvaluable`; the line search halves the step as it does for a trial whose
residual is too large. The planted problem: ``r = u - 1`` with a tangent five times too small, so the
first Newton step from 0 lands at 5, where the residual refuses to evaluate (``u > 2``).
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import TrialNotEvaluable, integrate_transient


class _GuardedRoot:
    def __init__(self) -> None:
        self.refused = 0

    def residual(self, u, u_dot, u_ddot, t):
        if u[0] > 2.0:
            self.refused += 1
            msg = f'inner solve has no root at u={u[0]}'
            raise TrialNotEvaluable(msg)
        return u - 1.0 + 0.0 * u_dot

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.atleast_2d(0.2 * weights.alpha0.real)


_STUDY = TimeWindow(t0=0.0, t1=0.3, dt=0.1, integrator=IntegratorKind.BDF2)


def test_an_unevaluable_trial_is_halved_and_the_march_reaches_the_root() -> None:
    problem = _GuardedRoot()
    sol = integrate_transient(problem, _STUDY, np.zeros(1), max_newton=50)
    assert sol.converged
    assert float(sol.states[-1][0]) == pytest.approx(1.0, abs=1.0e-9)
    # PLANTED CONTROL: the guard really fired -- the halving, not a lucky step, reached the root.
    assert problem.refused > 0


def test_trial_not_evaluable_is_a_runtime_error_so_existing_callers_keep_their_contract() -> None:
    assert issubclass(TrialNotEvaluable, RuntimeError)
