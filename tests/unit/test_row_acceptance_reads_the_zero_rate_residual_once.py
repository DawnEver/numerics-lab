"""One ``RowAcceptance`` verdict reads the zero-rate residual ``F(u, 0)`` ONCE.

The scale reads ``|F(u, 0)|`` and the homogeneity screen reads the signed ``F(u, 0)`` at the SAME state and
time; evaluating it twice was a whole extra residual per Newton iterate (measured downstream 2026-11-12 on
an 800k-step LLC march: 32619 duplicate residual evaluations per 16000 steps).
"""

from types import SimpleNamespace

import numpy as np

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._step import RowAcceptance


class _Counting:
    def __init__(self) -> None:
        self.residuals = 0

    def residual(self, state, _rate, _acceleration, _time):
        self.residuals += 1
        return np.array([state[0] - 1.0, state[1] + 2.0 * state[0]])

    def jacobian(self, _state, _rate, _acceleration, _time, weights):
        return weights.alpha0 * np.array([[1.0, 0.0], [2.0, 1.0]])


def test_one_verdict_evaluates_the_zero_rate_residual_once():
    problem = _Counting()
    form = SimpleNamespace(
        t_eval=0.0,
        weights=DerivativeWeights(1.0, 0.0, 0.0),
        u_arg=lambda state: state,
        u_dot=np.zeros_like,
        u_ddot=np.zeros_like,
    )
    u = np.array([1.0, -2.0])
    assert RowAcceptance(problem, form).accepts(u, problem.residual(u, 0, 0, 0.0))
    assert problem.residuals == 2  # the caller's own residual, plus ONE read of F(u, 0)
