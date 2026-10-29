"""The gauss half of W6 factorization reuse: :class:`FactorCache` and its march wiring.

The cache is a solve-seam object: the problem may expose ``factor_reuse_key(u, u_dot, u_ddot,
t, weights)`` returning a value that DETERMINES the step jacobian; while the key holds, the
cached factorization back-substitutes instead of re-factoring. gauss trusts the key exactly as it
trusts ``residual`` / ``jacobian`` -- key SOUNDNESS is the problem's (laplace pins it in
``tests/unit/laplace/solver/test_circuit_factor_reuse_key.py``); what gauss owns here is the mechanics: hit/miss
accounting, dense/sparse/scalar storage, and that a problem WITHOUT the seam marches untouched.
"""
# ruff: noqa: PLC0415 -- test-module lazy imports: the module under test is imported inside
# the test/fixture, keeping collection cheap and loading platform-conditional deps (win32com,
# OCP, FEMM) only on the path that needs them. Hoist a file's imports and DELETE this header;
# the population is ratcheted by tests/architecture/ratchets/test_suppression_ratchet.py.

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from numerics_lab.formulation import DerivativeWeights, IntegratorKind, TimeWindow
from numerics_lab.time_integrators import factor_reuse, integrate_transient
from numerics_lab.time_integrators.factor_reuse import FactorCache
from numerics_lab.time_integrators.march import _default_solve


class _SpringBase:
    """``u_dot = -K u`` (decaying 2-DOF), optionally with a MUTABLE stiffness for the event tests.

    ``factor_reuse_key`` is weights-plus-stiffness: everything its ``jacobian`` reads. ``K`` is a
    constructor constant unless the test mutates ``k_scale`` (the synthetic stand-in for a switch
    toggle), and the key re-reads it every call, so a mutation shows up in the key unprompted.
    """

    def __init__(self, k: NDArray[np.floating]) -> None:
        self.k = np.asarray(k, dtype=float)
        self.k_scale = 1.0

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + self.k_scale * self.k @ u

    def jacobian(self, u, u_dot, u_ddot, t, weights: DerivativeWeights):
        return weights.alpha1.real * np.eye(len(self.k)) + weights.alpha0.real * self.k_scale * self.k


class _LinearSpring(_SpringBase):
    """A spring whose complete Jacobian dependency set forms a stable reuse key."""

    def factor_reuse_key(self, u, u_dot, u_ddot, t, weights: DerivativeWeights):
        return (weights.alpha0.real, weights.alpha1.real, weights.alpha2.real, self.k_scale)


class _NoKeySpring(_SpringBase):
    """The same physics WITHOUT the reuse seam -- the pre-W6 problem shape (hamilton today)."""


def _study(dt: float = 0.05, t1: float = 1.0, integrator=IntegratorKind.BDF2) -> TimeWindow:
    return TimeWindow(t0=0.0, t1=t1, dt=dt, integrator=integrator)


def _march(problem, study, **kwargs):
    return integrate_transient(problem, study, np.array([1.0, 0.5]), **kwargs)


def test_a_uniform_linear_march_factors_once_per_distinct_step_form() -> None:
    """BDF2 on a uniform grid has exactly TWO weight sets: the BDF1 bootstrap, then steady BDF2.

    Every later step reuses. Measured pre-W6 (2026-07-22, see the W6 commit): an unfactored march
    pays one factorization per solve, i.e. ``steps`` of them; the count, not the wall-clock, is the
    machine-independent claim.
    """
    study = _study()
    cached = FactorCache()
    _march(_LinearSpring([[2.0, 0.1], [0.1, 1.0]]), study, factor_cache=cached)
    n_steps = round((study.t1 - study.t0) / study.dt)
    assert cached.solves == n_steps  # linear: one exact Newton solve per step
    assert cached.factorizations == 2, f'{cached.factorizations} factorizations for a 2-weight march'
    assert cached.factorizations + cached.reuses == cached.solves


def test_the_cached_march_is_bit_identical_to_the_uncached_march() -> None:
    """lu_factor+lu_solve and np.linalg.solve are the same LAPACK getrf/getrs pair (measured
    2026-07-22: bit-equal on 200 random 5x5 systems), so reuse must not move ONE BIT.
    """
    study = _study()
    problem = _LinearSpring([[2.0, 0.1], [0.1, 1.0]])
    plain = _march(problem, study)
    cached = _march(problem, study, factor_cache=FactorCache())
    assert plain.times == cached.times
    for a, b in zip(plain.states, cached.states, strict=True):
        assert np.array_equal(a, b), 'reuse moved the march'


def test_a_problem_without_the_seam_marches_exactly_as_before() -> None:
    """No ``factor_reuse_key`` -> the cache is never consulted and ``solve`` is called per step."""
    study = _study()
    calls = {'n': 0}

    def counting_solve(matrix, rhs):
        calls['n'] += 1
        return _default_solve(matrix, rhs)

    solution = _march(_NoKeySpring([[2.0, 0.1], [0.1, 1.0]]), study, solve=counting_solve, factor_cache=FactorCache())
    n_steps = round((study.t1 - study.t0) / study.dt)
    assert calls['n'] == n_steps, 'a keyless problem must solve per step, cache or no cache'
    assert len(solution.states) == n_steps + 1


def test_a_march_with_no_cache_argument_reuses_by_default_when_the_seam_exists() -> None:
    """Default-on: ``integrate_transient`` builds its own cache when the problem offers the seam."""
    study = _study()
    keyed = _march(_LinearSpring([[2.0, 0.1], [0.1, 1.0]]), study)
    plain = _march(_NoKeySpring([[2.0, 0.1], [0.1, 1.0]]), study)
    for a, b in zip(keyed.states, plain.states, strict=True):
        assert np.array_equal(a, b)


def test_a_state_change_is_a_new_key_and_refactors() -> None:
    """The event case: toggle the stiffness mid-march -> the key must change -> refactor."""
    problem = _LinearSpring([[2.0, 0.1], [0.1, 1.0]])
    cache = FactorCache()
    _study()
    # First half of the march.
    first = integrate_transient(problem, _study(t1=0.5), np.array([1.0, 0.5]), factor_cache=cache)
    n_first = cache.factorizations
    problem.k_scale = 3.0  # the 'switch toggle'
    integrate_transient(problem, _study(t1=0.5), np.array([1.0, 0.5]), factor_cache=cache)
    assert cache.factorizations > n_first, 'a mutated system reused a stale factorization'
    assert first is not None


def test_the_event_march_refactors_on_the_toggle_and_stays_bit_identical() -> None:
    """``integrate_transient``: the toggle flips the key, the grid event is exact."""
    problem = _LinearSpring([[2.0, 0.1], [0.1, 1.0]])
    study = _study()
    cache = FactorCache()

    def toggle(_i, _t, _u) -> None:
        problem.k_scale = 3.0

    kwargs = {'event_fns': (lambda _u, t: t - 0.5,), 'on_event': toggle}
    plain = integrate_transient(problem, study, np.array([1.0, 0.5]), **kwargs)
    problem.k_scale = 1.0
    cached = integrate_transient(problem, study, np.array([1.0, 0.5]), factor_cache=cache, **kwargs)
    assert plain.times == cached.times and plain.events == cached.events
    for a, b in zip(plain.states, cached.states, strict=True):
        assert np.array_equal(a, b)
    n_steps = round((study.t1 - study.t0) / study.dt)
    assert cache.solves >= n_steps  # the event adds landing/remainder steps
    assert cache.factorizations < cache.solves // 2, (
        f'{cache.factorizations} factorizations for {cache.solves} solves: reuse is not working'
    )


@pytest.mark.parametrize('native', ['0', '1'])
def test_the_sparse_path_back_substitutes_like_spsolve(native, monkeypatch) -> None:
    """With the native factor switched off, splu and spsolve are the same SuperLU call (bit-equal,
    measured 2026-07-22); with it on, the held factor is faer's and agrees to round-off.
    """
    monkeypatch.setenv('NUMERICS_LAB_DIRECT_NATIVE', native)
    m = csr_matrix(np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]]))
    b = np.array([1.0, 2.0, 3.0])
    cache = FactorCache()
    key = ('k', 1)
    x1 = cache.solve(key, lambda: m, b)
    x2 = cache.solve(key, lambda: m, 2.0 * b)  # a hit: same key, new rhs
    assert cache.factorizations == 1
    assert np.allclose(x2, 2.0 * x1, atol=0.0)
    from scipy.sparse.linalg import spsolve

    ref = np.asarray(spsolve(m.tocsc(), b))
    if native == '0':
        assert np.array_equal(x1, ref)
    else:
        assert np.abs(x1 - ref).max() <= 1e-14 * np.abs(ref).max()


def test_a_none_key_never_caches() -> None:
    """``None`` is the problem's 'cannot prove unchanged' -- every solve factors."""
    cache = FactorCache()
    m = np.array([[2.0, 0.0], [0.0, 1.0]])
    cache.solve(None, lambda: m, np.ones(2))
    cache.solve(None, lambda: m, np.ones(2))
    assert cache.factorizations == 2
    assert cache.reuses == 0


def test_a_repaired_miss_also_repairs_later_hits(monkeypatch) -> None:
    """Reuse must not replay an inaccurate back-substitution on another RHS."""
    matrix = csr_matrix(np.diag([1.0, 1.0e-15]))

    class InaccurateFactor:
        def solve(self, rhs):
            return np.array([rhs[0], rhs[1] * 1.0e15 * (1.0 + 1.0e-4)])

    monkeypatch.setattr(factor_reuse, 'factor_matrix', lambda _matrix: InaccurateFactor())
    cache = FactorCache()
    first = cache.solve('same', lambda: matrix, np.array([1.0, 1.0]))
    second = cache.solve('same', lambda: matrix, np.array([2.0, 2.0]))

    assert np.allclose(first, [1.0, 1.0e15], rtol=1.0e-12)
    assert np.allclose(second, [2.0, 2.0e15], rtol=1.0e-12)
    assert cache.factorizations == 1 and cache.reuses == 1
