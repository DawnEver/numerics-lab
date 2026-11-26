"""CG, MINRES and Steihaug-CG against a direct solve."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear.krylov import conjugate_gradient, minres, steihaug_cg

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
