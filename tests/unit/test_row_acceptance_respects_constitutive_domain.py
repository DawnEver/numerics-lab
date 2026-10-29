"""A numerical row scale cannot query an undeclared constitutive origin."""

from types import SimpleNamespace

import numpy as np
import pytest

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._step import _solve_step


@pytest.mark.parametrize('correct_small_row', [False, True])
def test_row_acceptance_never_invents_a_zero_state_outside_the_actual_law_domain(correct_small_row):
    target = 1e-30
    observed = []

    class Problem:
        def residual(self, state, _rate, _acceleration, _time):
            observed.append(state.copy())
            if state[0] <= 0:
                msg = 'the logarithmic constitutive state must be positive'
                raise ValueError(msg)
            return np.array([np.log(state[0]) - np.log(2.0), state[1] - target])

        def jacobian(self, state, _rate, _acceleration, _time, weights):
            return weights.alpha0 * np.diag([1 / state[0], 1.0])

    form = SimpleNamespace(
        t_eval=0.0,
        weights=DerivativeWeights(1.0, 0.0, 0.0),
        u_arg=lambda state: state,
        u_dot=np.zeros_like,
        u_ddot=np.zeros_like,
    )
    root, converged, _iterations, _residual, _on_floor = _solve_step(
        Problem(),
        form,
        np.array([2.0, 0.0]),
        solve=lambda matrix, rhs: np.linalg.solve(matrix, rhs) if correct_small_row else np.zeros(2),
        max_newton=3,
        tol=1e-50,
        row_acceptance=True,
    )
    assert observed and all(state[0] > 0 for state in observed)
    assert bool(converged) is correct_small_row
    # An unknown source/homogeneity cannot let the large logarithmic row hide
    # a small, independently declared nonzero physical equation.
    assert root[1] == (target if correct_small_row else 0.0)
