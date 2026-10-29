"""Trust-region radius and dogleg; CG, MINRES and Steihaug-CG against a direct solve."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.krylov import conjugate_gradient, minres, steihaug_cg
from numerics_lab.trust_region import agreement, dogleg, update_radius

_RNG = np.random.default_rng(11)
_N = 12
_M = _RNG.standard_normal((_N, _N))
_SPD = _M @ _M.T + _N * np.eye(_N)
_INDEFINITE = _SPD - 1.5 * _N * np.eye(_N)
_B = _RNG.standard_normal(_N)


def test_cg_solves_an_spd_system_matrix_free_and_sparse() -> None:
    exact = np.linalg.solve(_SPD, _B)
    for operator in (_SPD, sp.csr_matrix(_SPD), lambda v: _SPD @ v):
        result = conjugate_gradient(operator, _B, rtol=1e-12)
        assert result.converged
        np.testing.assert_allclose(result.x, exact, rtol=1e-9)


def test_cg_does_not_claim_an_indefinite_system() -> None:
    assert np.min(np.linalg.eigvalsh(_INDEFINITE)) < 0.0 < np.max(np.linalg.eigvalsh(_INDEFINITE))
    assert not conjugate_gradient(_INDEFINITE, _B, rtol=1e-12).converged


def test_minres_solves_a_symmetric_indefinite_system() -> None:
    result = minres(_INDEFINITE, _B, rtol=1e-12)
    assert result.converged
    np.testing.assert_allclose(result.x, np.linalg.solve(_INDEFINITE, _B), rtol=1e-8)
    # Planted control: a two-iteration budget is not a solve, and says so.
    assert not minres(_INDEFINITE, _B, rtol=1e-12, max_iterations=2).converged


def test_minres_agrees_with_the_oracle_iterate() -> None:
    from scipy.sparse.linalg import minres as oracle  # noqa: PLC0415 -- tests only

    ours = minres(_INDEFINITE, _B, rtol=1e-12).x
    theirs, info = oracle(_INDEFINITE, _B, rtol=1e-12)
    assert info == 0
    np.testing.assert_allclose(ours, theirs, rtol=1e-7)


def test_steihaug_reaches_newton_inside_a_large_ball() -> None:
    newton = -np.linalg.solve(_SPD, _B)
    result = steihaug_cg(_SPD, _B, radius=1e6, rtol=1e-12)
    assert result.stop == 'converged'
    np.testing.assert_allclose(result.step, newton, rtol=1e-8)


def test_steihaug_stops_on_the_boundary_of_a_small_ball() -> None:
    result = steihaug_cg(_SPD, _B, radius=1e-3)
    assert result.stop == 'boundary'
    assert math.isclose(float(np.linalg.norm(result.step)), 1e-3, rel_tol=1e-12)
    assert float(_B @ result.step) < 0.0


def test_steihaug_follows_negative_curvature_to_the_boundary() -> None:
    hessian = -np.eye(3)
    result = steihaug_cg(hessian, np.array([1.0, 0.0, 0.0]), radius=2.0)
    assert result.stop == 'negative_curvature'
    np.testing.assert_allclose(result.step, [-2.0, 0.0, 0.0])


def test_the_radius_shrinks_grows_or_holds_by_agreement() -> None:
    assert agreement(1.0, 2.0) == 0.5
    assert agreement(1.0, 0.0) == -math.inf
    assert update_radius(1.0, 0.1, 0.8, max_radius=10.0) == 0.25 * 0.8
    assert update_radius(1.0, 0.9, 1.0, max_radius=10.0) == 2.0
    assert update_radius(8.0, 0.9, 8.0, max_radius=10.0) == 10.0
    # Planted control: a good agreement INSIDE the ball is no reason to grow.
    assert update_radius(1.0, 0.9, 0.5, max_radius=10.0) == 1.0
    assert update_radius(1.0, 0.5, 1.0, max_radius=10.0) == 1.0


def test_dogleg_takes_newton_inside_and_cuts_on_the_boundary() -> None:
    newton = -np.linalg.solve(_SPD, _B)
    apply = lambda v: _SPD @ v  # noqa: E731
    np.testing.assert_allclose(dogleg(_B, apply, newton, 1e6), newton)
    for radius in (1e-4, 0.5 * float(np.linalg.norm(newton))):
        step = dogleg(_B, apply, newton, radius)
        assert math.isclose(float(np.linalg.norm(step)), radius, rel_tol=1e-10)
        model = lambda p: float(_B @ p + 0.5 * p @ _SPD @ p)  # noqa: E731
        assert model(step) < 0.0


def test_dogleg_refuses_a_nonconvex_model() -> None:
    with pytest.raises(ValueError, match='positive definite'):
        dogleg(np.ones(2), lambda v: -v, np.full(2, 10.0), 1.0)
