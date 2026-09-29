"""The ``direct`` backend solves through faer when the wheel provides it, and says so when it does not.

Three properties, each planted rather than assumed: the native path answers the same system SciPy
answers (to round-off, not bit-for-bit -- it is a different factorisation); a held factor is never
back-substituted after another solve overwrote the analysis it shares; and the SciPy fallback is
logged and counted, never silent.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from motronics.gauss.linear import native_direct, solve_linear_system
from motronics.gauss.time_integrators.factor_reuse import apply_factor, factor_matrix

needs_native = pytest.mark.skipif(
    not native_direct.native_available(), reason='motronics_native with DirectFactorization.refactor not built'
)


def _laplacian(side: int) -> sp.csr_matrix:
    one = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(side, side))
    eye = sp.identity(side)
    return (sp.kron(one, eye) + sp.kron(eye, one)).tocsr()


def _unsymmetric(n: int) -> sp.csr_matrix:
    return sp.diags([-1.0, 3.0, -0.5], [-1, 0, 1], shape=(n, n)).tocsr()


@needs_native
@pytest.mark.parametrize(
    ('build', 'kind'), [(lambda: _laplacian(20), 'faer-cholesky'), (lambda: _unsymmetric(300), 'faer-lu')]
)
def test_the_native_path_answers_what_superlu_answers_and_names_its_factorisation(build, kind):
    a = build()
    b = np.linspace(-1.0, 2.0, a.shape[0])
    before = native_direct.DIRECT_SOLVES[kind]
    x = solve_linear_system(a, b)
    assert native_direct.DIRECT_SOLVES[kind] == before + 1
    ref = spsolve(a.tocsc(), b)
    assert np.max(np.abs(x - ref)) <= 1e-12 * np.max(np.abs(ref))


@needs_native
def test_a_revalued_system_on_the_same_pattern_reuses_the_analysis_and_is_still_right():
    a = _unsymmetric(200)
    b = np.ones(200)
    solve_linear_system(a, b)
    entries = len(native_direct._cache)
    moved = a.copy()
    moved.data = moved.data * np.linspace(1.0, 2.0, moved.nnz)
    x = solve_linear_system(moved, b)
    assert len(native_direct._cache) == entries
    assert np.linalg.norm(moved @ x - b) <= 1e-12 * np.linalg.norm(b)


@needs_native
def test_a_held_factor_is_not_answered_from_storage_another_solve_overwrote():
    # The planted hazard: two operators on ONE pattern share one analysis. Holding the first's
    # factor, solving the second, then back-substituting the first must answer the FIRST.
    a = _unsymmetric(150)
    other = a.copy()
    other.data = other.data * 1.7
    b = np.linspace(0.0, 1.0, 150)
    held = factor_matrix(a)
    assert isinstance(held, native_direct.NativeFactor)
    solve_linear_system(other, b)
    x = apply_factor(held, b)
    assert np.linalg.norm(a @ x - b) <= 1e-12 * np.linalg.norm(b)


@needs_native
def test_a_held_factor_back_substitutes_a_block_of_right_hand_sides():
    a = _laplacian(10)
    rhs = np.column_stack([np.ones(100), np.arange(100.0)])
    x = apply_factor(factor_matrix(a), rhs)
    assert x.shape == (100, 2)
    assert np.linalg.norm(a @ x - rhs) <= 1e-12 * np.linalg.norm(rhs)


def test_the_superlu_fallback_is_logged_and_counted(monkeypatch, caplog):
    monkeypatch.setenv('MOTRONICS_DIRECT_NATIVE', '0')
    monkeypatch.setattr(native_direct, '_warned', set())
    a = _unsymmetric(50)
    b = np.ones(50)
    before = native_direct.DIRECT_SOLVES['superlu']
    with caplog.at_level(logging.WARNING, logger=native_direct.__name__):
        x = solve_linear_system(a, b)
    assert native_direct.DIRECT_SOLVES['superlu'] == before + 1
    assert any('falls back to SciPy SuperLU' in r.getMessage() for r in caplog.records)
    assert np.array_equal(x, spsolve(a, b))  # the fallback IS spsolve, bit for bit


def test_a_complex_operator_is_outside_the_native_reach_and_counted_as_superlu():
    a = (_unsymmetric(30) * (1.0 + 1.0j)).tocsr()
    b = np.ones(30, dtype=complex)
    before = native_direct.DIRECT_SOLVES['superlu']
    x = solve_linear_system(a, b)
    assert native_direct.DIRECT_SOLVES['superlu'] == before + 1
    assert np.linalg.norm(a @ x - b) <= 1e-12 * np.linalg.norm(b)
