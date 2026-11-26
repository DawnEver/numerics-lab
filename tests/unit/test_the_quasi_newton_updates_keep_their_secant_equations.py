"""The secant pair, the two-loop recursion and Broyden: each update keeps its secant equation, and skips a degenerate pair."""

from __future__ import annotations

import numpy as np

from numerics_lab.quasi_newton import (
    broyden_apply,
    broyden_inverse_update,
    broyden_update,
    secant_pair,
    two_loop,
)

_RNG = np.random.default_rng(7)
_N = 5
_G = _RNG.standard_normal((_N, _N))
_SPD = _G @ _G.T + _N * np.eye(_N)


def _pairs(count):
    s = _RNG.standard_normal((count, _N))
    return s, s @ _SPD.T  # y = A s: curvature y . s > 0


def test_the_two_loop_keeps_the_newest_secant_equation() -> None:
    s, y = _pairs(4)
    pairs = [secant_pair(s[k], y[k]) for k in range(4)]
    np.testing.assert_allclose(two_loop(y[-1], pairs, lambda v: 0.5 * v), s[-1], rtol=1e-10)
    # Planted control: without the newest pair the equation no longer holds.
    assert not np.allclose(two_loop(y[-1], pairs[:-1], lambda v: 0.5 * v), s[-1], rtol=1e-6)


def test_a_flat_secant_pair_is_none() -> None:
    assert secant_pair(np.array([1.0, 0.0]), np.array([0.0, 1.0])) is None
    assert secant_pair(np.array([1.0, 0.0]), np.array([1.0, 1.0])) is not None


def test_broyden_matrix_free_equals_dense_and_keeps_the_secant_equation() -> None:
    a = _SPD + np.triu(_RNG.standard_normal((_N, _N)))  # unsymmetric
    h_dense = np.eye(_N)
    updates = []
    for _ in range(3):
        s = _RNG.standard_normal(_N)
        y = a @ s
        h_y = broyden_apply(y.copy(), updates)  # H0 = I
        update = broyden_update(s, h_y)
        assert update is not None
        updates.append(update)
        h_dense = broyden_inverse_update(h_dense, s, y)
        np.testing.assert_allclose(broyden_apply(y.copy(), updates), s, rtol=1e-10)
    r = _RNG.standard_normal(_N)
    np.testing.assert_allclose(broyden_apply(r.copy(), updates), h_dense @ r, rtol=1e-10)
    assert broyden_update(np.array([1.0, 0.0]), np.array([0.0, 1.0])) is None
