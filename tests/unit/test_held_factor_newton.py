"""The modified (held-factor) Newton is the SAME Newton holding its factor (user ruling 2026-09-26, 3A).

What is pinned:

* `newton-held-factor` is a DECLARED registry name, residual-stopped like `newton`, NOT the default;
* on a saturating system it reaches the same root as full Newton, to the shared residual bar, while
  paying FEWER tangent factorizations than iterations (full Newton pays one per iteration);
* a held step that needs a line search is retaken with a fresh factor, so a stale factor cannot
  march a wrong direction (the legacy energy-stopped `modified-newton` had no such guard);
* on the FEA Galerkin system the system's own `factorize` seam lands the field full Newton lands.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.contracts.spec.analysis_formulation import DerivativeWeights
from motronics.contracts.spec.study_spec import TransientStudy
from motronics.core.materials.library import library
from motronics.core.materials.registry import compile_selection
from motronics.gauss.nonlinear import (
    DEFAULT_ITERATION_STRATEGY,
    ITERATION_REGISTRY,
    AssembledState,
    HeldFactorNewtonStrategy,
    NewtonStrategy,
    get_strategy,
    strategy_stops_on_residual,
)
from motronics.gauss.nonlinear._seam import _TimeBudgetedSystem
from motronics.gauss.nonlinear._strategies_newton import _residual_reference, _SeededSystem
from motronics.gauss.time_integrators import integrate_transient
from motronics.gauss.time_integrators._step import HeldFactor
from motronics.hamilton.discretizations.constitutive import binding_from_group_materials
from motronics.hamilton.solve import solve

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('wrappers', ['bare', 'seeded', 'timed', 'seeded-timed', 'timed-seeded'])
def test_transparent_wrappers_preserve_the_physical_source_reference(wrappers) -> None:
    base = _CubicSystem()
    seed = base.initial_guess()
    state = base.assemble(seed)
    system = base
    for wrapper in wrappers.split('-'):
        if wrapper == 'seeded':
            system = _SeededSystem(system, seed)
        elif wrapper == 'timed':
            system = _TimeBudgetedSystem(system, 10.0, lambda: 0.0)
    source_norm = base.rhs_norm(state)
    assert _residual_reference(system, state, source_norm * 1e-8) == source_norm
    assert _residual_reference(system, state, source_norm * 2.0) == source_norm * 2.0


def test_a_system_without_a_source_norm_retains_its_seed_reference() -> None:
    system = _TimeBudgetedSystem(object(), 10.0, lambda: 0.0)
    assert _residual_reference(system, None, 3.0) == 3.0


class _CubicSystem:
    """``r(x) = K x + c x^3 - b``: a sparse, saturating-like system with an exact sparse tangent."""

    is_linear = False

    def __init__(self, n: int = 40, c: float = 5.0) -> None:
        main = 2.0 * np.ones(n)
        off = -np.ones(n - 1)
        self.k = sp.diags([off, main, off], [-1, 0, 1], format='csr')
        self.c = c
        self.b = np.linspace(1.0, 3.0, n)

    def initial_guess(self) -> np.ndarray:
        return np.zeros(self.k.shape[0])

    def assemble(self, x, /) -> AssembledState:
        return AssembledState(system_matrix=self.k, total_energy=float(x @ (self.k @ x)))

    def residual(self, x, state, /):
        del state
        return self.k @ x + self.c * x**3 - self.b

    def rhs_norm(self, state):
        del state
        return float(np.linalg.norm(self.b))

    def tangent(self, x, state, /):
        del state
        return (self.k + sp.diags(3.0 * self.c * x**2)).tocsr()

    def solve_system(self, matrix, rhs, /):
        return sp.linalg.spsolve(sp.csc_matrix(matrix), rhs)

    def on_iteration(self, *_args) -> None:
        return None


def test_the_held_factor_newton_is_declared_residual_stopped_and_not_the_default() -> None:
    assert isinstance(ITERATION_REGISTRY['newton-held-factor'], HeldFactorNewtonStrategy)
    assert strategy_stops_on_residual('newton-held-factor')
    assert DEFAULT_ITERATION_STRATEGY == 'newton'
    assert get_strategy('newton').refactor_every is None, 'full Newton must stay full Newton'


def test_a_refactor_period_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match='refactor_every'):
        NewtonStrategy(refactor_every=0)


def test_it_reaches_the_newton_root_on_fewer_factorizations() -> None:
    rtol_pct = 1e-8  # 1e-10 as a fraction: both must reach the same root to that scale
    full = NewtonStrategy()
    held = HeldFactorNewtonStrategy()
    rf = full.solve(_CubicSystem(), max_iter=100, rtol_energy_pct=rtol_pct)
    rh = held.solve(_CubicSystem(), max_iter=100, rtol_energy_pct=rtol_pct)
    assert rf.converged and rh.converged
    np.testing.assert_allclose(rh.x, rf.x, rtol=1e-8, atol=1e-10)
    assert full.factorizations == rf.iterations, 'full Newton factors once per iteration'
    assert held.factorizations < rh.iterations, (held.factorizations, rh.iterations)


def test_a_stale_factor_that_needs_a_line_search_is_refreshed_not_marched() -> None:
    """With an absurd hold (k = 1000) only the stall rules can refresh -- and they must."""
    strategy = NewtonStrategy(refactor_every=1000, stall_ratio=0.5)
    result = strategy.solve(_CubicSystem(c=50.0), max_iter=200, rtol_energy_pct=1e-8)
    assert result.converged
    assert strategy.factorizations > 1, 'a hard saturation must have forced at least one refresh'


# The FEA Galerkin seam: a 3x3 node grid, a conductor driving saturating steel (the same fixture
# as tests/unit/hamilton/discretizations/fem/test_fea_newton.py).
_VERT = np.array([[ix * 0.01, iy * 0.01, 0.0] for iy in range(3) for ix in range(3)])
_TRIS = np.array(
    [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]], dtype=np.intp
)
_GROUP = np.array([0, 0, 1, 1, 1, 1, 1, 1])
_GML = [('conductor', 'cu'), ('steel', 'synthetic_bh_steel')]
_MIU = {'cu': {'material_type': 'conductor'}, 'synthetic_bh_steel': {'material_type': 'steel'}}


def _fea(strategy: str):
    return solve(
        'fea',
        ele_mat=_TRIS,
        ele_group_mat=_GROUP,
        vert_mat=_VERT,
        binding=binding_from_group_materials(_GML, compile_selection(library(), _MIU)),
        conductor_current_density_list=np.array([3.0e8, 0.0]),
        bound_vert_ids_dict={'dirichlet': {0.0: np.array([0, 1, 2, 3, 5, 6, 7, 8])}, 'symmetry': {}},
        depth=0.005,
        iteration_strategy=strategy,
    )


def test_the_fea_factorize_seam_lands_the_newton_field() -> None:
    full, held = _fea('newton'), _fea('newton-held-factor')
    assert np.any(np.abs(full.b_norm_ele) > 1e-6), 'nothing was exercised'
    # Both stop on the same residual bar (core.newton_rtol), so they agree to that scale.
    np.testing.assert_allclose(held.a_vert, full.a_vert, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(held.energy_norm_ele.sum(), full.energy_norm_ele.sum(), rtol=1e-6)


class _SaturatingDiffusion:
    """``u_dot + K u + c u^3 = f``: a transient with a state-dependent sparse tangent."""

    def __init__(self, n: int = 30, c: float = 4.0) -> None:
        main = 2.0 * np.ones(n)
        off = -np.ones(n - 1)
        self.k = sp.diags([off, main, off], [-1, 0, 1], format='csr')
        self.c = c
        self.f = np.linspace(1.0, 4.0, n)

    def residual(self, u, u_dot, u_ddot, t):
        del u_ddot, t
        return u_dot + self.k @ u + self.c * u**3 - self.f

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        del u_dot, u_ddot, t
        alpha1 = complex(weights.alpha1).real
        return (alpha1 * sp.identity(u.size) + self.k + sp.diags(3.0 * self.c * u**2)).tocsr()


def test_the_held_factor_march_lands_the_full_newton_trajectory() -> None:
    problem = _SaturatingDiffusion()
    study = TransientStudy(t0=0.0, t1=0.2, dt=0.01)
    u0 = np.zeros(30)
    full = integrate_transient(problem, study, u0, max_newton=30, tol=1e-12, tol_rel=1e-10)
    held = integrate_transient(problem, study, u0, max_newton=30, tol=1e-12, tol_rel=1e-10, refactor_every=4)
    assert full.converged and held.converged
    np.testing.assert_allclose(np.asarray(held.states), np.asarray(full.states), rtol=1e-8, atol=1e-10)


def test_a_held_factor_is_refreshed_on_its_period_and_on_a_new_step_size() -> None:
    held = HeldFactor(refactor_every=2)
    w1, w2 = DerivativeWeights(alpha1=100.0), DerivativeWeights(alpha1=50.0)
    assert held.needs_refresh(w1)
    held.refresh(sp.identity(3, format='csr') * 2.0, w1)
    held.solve(np.ones(3))
    assert not held.needs_refresh(w1)
    assert held.needs_refresh(w2), 'a new dt changes the operator'
    held.solve(np.ones(3))
    assert held.needs_refresh(w1), 'the period is spent'
    with pytest.raises(ValueError, match='refactor_every'):
        HeldFactor(refactor_every=0)
