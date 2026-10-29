"""An unrelated large row cannot certify another equation's arithmetic."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._step import ProblemCapabilities, _at_roundoff_floor, _gmin_continue, _solve_step


class _Unknown:
    def residual(self, u, rate, acceleration, time):
        return np.array([u[0], rate[1]])

    def jacobian(self, u, rate, acceleration, time, weights):
        return np.diag([weights.alpha0.real, weights.alpha1.real])


@pytest.mark.parametrize('sparse', [False, True])
def test_a_large_dynamic_row_cannot_certify_an_unrelated_algebraic_defect(sparse):
    matrix = np.diag([1.0, 1e16])
    tangent = sp.csr_matrix(matrix) if sparse else matrix
    problem = _Unknown()
    capabilities = ProblemCapabilities.resolve(problem)
    assert not _at_roundoff_floor(problem, tangent, np.ones(2), None, capabilities, np.array([1e-6, 0.0]))
    assert _at_roundoff_floor(problem, tangent, np.ones(2), None, capabilities, np.array([1e-15, 1.0]))


@pytest.mark.parametrize(('floor', 'accepted'), [(2e-6, True), ([2e-6, 0.0], True), ([0.0, 2e-6], False)])
def test_explicit_scalar_and_row_floor_authority_is_unchanged(floor, accepted):
    class Declared(_Unknown):
        def residual_roundoff_floor(self, u, form, *, residual):
            return floor

    problem = Declared()
    actual = _at_roundoff_floor(
        problem,
        np.diag([1.0, 1e16]),
        np.ones(2),
        None,
        ProblemCapabilities.resolve(problem),
        np.array([1e-6, 0.0]),
    )
    assert actual == accepted


@pytest.mark.parametrize('declared', [False, True])
def test_gmin_continuation_uses_the_same_row_floor_authority_as_plain_newton(declared):
    class Problem:
        def residual(self, u, rate, acceleration, time):
            return np.array([u[0] ** 2 - 1.0, rate[1]])

        def jacobian(self, u, rate, acceleration, time, weights):
            return np.diag([2 * u[0], weights.alpha1.real])

    problem = Problem()
    if declared:

        def floor(_u, _form, *, residual):
            return np.zeros_like(residual)

        problem.residual_roundoff_floor = floor
    form = SimpleNamespace(
        t_eval=0.0,
        weights=DerivativeWeights(1.0, 1e16, 0.0),
        u_arg=lambda u: u,
        u_dot=lambda u: np.array([0.0, 1e16 * (u[1] - 1.0)]),
        u_ddot=np.zeros_like,
    )
    _state, converged, _residual, on_floor = _gmin_continue(
        problem,
        form,
        np.array([1.0001, 1.0]),
        solve=np.linalg.solve,
        max_newton=2,
        tol=1e-12,
        tol_rel=None,
        residual_0=0.001,
        gmin_stages=(1e16,),
    )
    assert not converged
    assert not on_floor


@pytest.mark.parametrize('sparse', [False, True])
def test_backtracking_can_select_a_trial_already_admitted_by_every_existing_row_floor(sparse):
    """A quantized high-conductance KCL must not strand a correctable small KVL."""
    target, conductance, coupling = 2.0**-18, 1e6, 1e-3
    dense = np.array([[1.0, 0.0, 0.0], [conductance, -conductance, -coupling], [0.0, 0.0, 1.0]])
    matrix = sp.csr_matrix(dense) if sparse else dense

    class Problem:
        def residual(self, u, _rate, _acceleration, _time):
            return np.array([u[0] - 1.0, conductance * (u[0] - u[1]) - 0.6 - coupling * u[2], u[2] - target])

        def jacobian(self, _u, _rate, _acceleration, _time, _weights):
            return matrix

    problem = Problem()
    form = SimpleNamespace(
        t_eval=0.0,
        weights=DerivativeWeights(1.0, 0.0, 0.0),
        u_arg=lambda u: u,
        u_dot=np.zeros_like,
        u_ddot=np.zeros_like,
    )
    seed = np.array([1.0, 1.0 - (0.6 + coupling * target) / conductance, target])
    defect = problem.residual(seed, None, None, 0.0)[1]
    assert defect != 0.0
    seed[2] += np.copysign(2.0**-55, defect)
    initial_residual = problem.residual(seed, None, None, 0.0)
    trial = seed + np.linalg.solve(dense, -initial_residual)
    trial_residual = problem.residual(trial, None, None, 0.0)
    capabilities = ProblemCapabilities.resolve(problem)
    assert np.linalg.norm(trial_residual) > np.linalg.norm(initial_residual)
    assert not _at_roundoff_floor(problem, matrix, seed, form, capabilities, initial_residual)
    assert _at_roundoff_floor(problem, matrix, trial, form, capabilities, trial_residual)
    root, converged, _iterations, _residual, _on_floor = _solve_step(
        problem,
        form,
        seed,
        solve=lambda _matrix, rhs: np.linalg.solve(dense, rhs),
        max_newton=3,
        tol=1e-30,
        backtrack_cap=8,
    )
    assert converged
    assert root[2] == target
