"""The L-BFGS pair memory: one bookkeeping, two forms of one model.

``LbfgsMemory`` keeps the last ``size`` secant pairs, skips a pair whose curvature ``s.y`` is not
positive, and scales by ``gamma = s.y / y.y`` of the newest pair. Its inverse action (the two-loop
recursion on ``gamma I``) and its compact direct form ``B = theta I - W M W^T`` (Byrd, Nocedal and
Schnabel 1994, ``theta = 1 / gamma``) must be inverses of each other, and the compact form must equal
the dense BFGS chain from ``theta I`` through the same pairs. The planted control perturbs one
stored pair and the same comparison refuses it.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.quasi_newton import LbfgsMemory, SecantPair, SingularCompactModel, bfgs_update

_RNG = np.random.default_rng(11)
_N = 6
_G = _RNG.standard_normal((_N, _N))
_SPD = _G @ _G.T + _N * np.eye(_N)


def _filled(count: int, size: int) -> LbfgsMemory:
    memory = LbfgsMemory(size)
    for _ in range(count):
        s = _RNG.standard_normal(_N)
        memory.push(s, _SPD @ s)
    return memory


def _dense_chain(memory: LbfgsMemory) -> np.ndarray:
    b = (1.0 / memory.gamma) * np.eye(_N)
    for pair in memory.pairs:
        b = bfgs_update(b, pair.s, pair.y)
    return b


@pytest.mark.parametrize(('count', 'size'), [(1, 5), (4, 5), (9, 5)])
def test_the_compact_form_is_the_dense_bfgs_chain(count: int, size: int) -> None:
    memory = _filled(count, size)
    assert len(memory.pairs) == min(count, size)
    assert np.allclose(memory.hessian(), _dense_chain(memory), rtol=1e-10, atol=1e-10)


def test_the_two_loop_inverts_the_compact_form() -> None:
    memory = _filled(4, 5)
    q = _RNG.standard_normal(_N)
    assert np.allclose(memory.inverse_apply(memory.hessian() @ q), q, rtol=1e-9, atol=1e-9)


def test_a_non_positive_curvature_pair_is_skipped_and_gamma_follows_the_newest_kept() -> None:
    memory = _filled(2, 5)
    newest = memory.pairs[-1]
    assert memory.gamma == pytest.approx(float(newest.s @ newest.y) / float(newest.y @ newest.y))
    s = _RNG.standard_normal(_N)
    assert not memory.push(s, -s)
    assert len(memory.pairs) == 2 and memory.pairs[-1] is newest
    memory.clear()
    assert memory.pairs == () and memory.gamma == 1.0 and np.array_equal(memory.hessian(), np.eye(_N))


def test_planted_control_a_perturbed_pair_breaks_the_identity() -> None:
    memory = _filled(4, 5)
    compact = memory.hessian()
    pair = memory.pairs[1]
    planted = LbfgsMemory(5)
    for p in memory.pairs:
        planted.push(p.s, p.y + (1e-3 * _RNG.standard_normal(_N) if p is pair else 0.0))
    assert not np.allclose(compact, _dense_chain(planted), rtol=1e-10, atol=1e-10)


def _middle(memory: LbfgsMemory) -> np.ndarray:
    theta = 1.0 / memory.gamma
    s = np.column_stack([pair.s for pair in memory.pairs])
    y = np.column_stack([pair.y for pair in memory.pairs])
    sy = s.T @ y
    lower = np.tril(sy, -1)
    return np.block([[-np.diag(np.diag(sy)), lower.T], [lower, theta * (s.T @ s)]])


def test_a_well_scaled_memory_keeps_the_explicit_inverse_golden() -> None:
    memory = _filled(4, 5)
    _theta, _w, m = memory.compact_factors()
    golden = np.linalg.inv(_middle(memory))
    assert np.allclose(m, golden, rtol=1e-12, atol=1e-12 * float(np.max(np.abs(golden))))


def test_a_pair_with_curvature_below_eps_of_its_y_norm_is_rejected_and_the_model_stays() -> None:
    memory = _filled(3, 5)
    reference = memory.hessian()
    u = _RNG.standard_normal(_N)
    v = _RNG.standard_normal(_N)
    v -= (v @ u) / (u @ u) * u  # y nearly orthogonal to s: s.y ~ 1e-18, far above SECANT_FLOOR
    s, y = 1e-12 * u / np.linalg.norm(u), v / np.linalg.norm(v) + 1e-6 * u / np.linalg.norm(u)
    assert float(s @ y) > 1e-30
    assert not memory.push(s, y)
    hessian = memory.hessian()
    assert np.all(np.isfinite(hessian))
    assert np.allclose(hessian, reference, rtol=1e-12, atol=1e-12)


def test_a_singular_middle_matrix_raises_the_named_error() -> None:
    memory = _filled(2, 5)
    newest = memory.pairs[-1]
    memory.pairs = (*memory.pairs, SecantPair(np.zeros(_N), newest.y, newest.rho))  # a zero step: a null row
    with pytest.raises(SingularCompactModel):
        memory.compact_factors()
    assert issubclass(SingularCompactModel, np.linalg.LinAlgError)
