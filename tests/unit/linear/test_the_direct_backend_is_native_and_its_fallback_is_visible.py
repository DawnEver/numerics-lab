"""The ``direct`` backend solves through faer when the wheel provides it, and says so when it does not.

Three properties, each planted rather than assumed: the native path answers the same system SciPy
answers (to round-off, not bit-for-bit -- it is a different factorisation); a held factor is never
back-substituted after another solve overwrote the analysis it shares; and the SciPy fallback is
logged and counted, never silent.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from numerics_lab.linear import native_direct, solve_linear_system
from numerics_lab.time_integrators.factor_reuse import apply_factor, factor_matrix

needs_native = pytest.mark.skipif(
    not native_direct.native_available(), reason='numerics_lab._native with DirectFactorization.refactor not built'
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
@pytest.mark.parametrize('move_rows', [False, True])
def test_an_equal_size_different_pattern_is_never_answered_from_the_cache(monkeypatch, move_rows):
    monkeypatch.setattr(native_direct, '_cache', OrderedDict())
    a = sp.csr_matrix([[4.0, 0.2, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])
    other = sp.csr_matrix(
        [[4.0, 0.0, 0.0], [0.2, 5.0, 0.0], [0.0, 0.0, 6.0]]
        if move_rows
        else [[4.0, 0.0, 0.2], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    )
    assert a.shape == other.shape and a.nnz == other.nnz
    rhs = np.array([1.0, 2.0, 3.0])
    held = native_direct.native_factor(a)
    first = next(iter(native_direct._cache.values()))
    answer = native_direct.native_solve(other, rhs)
    matching = [
        entry
        for entry in native_direct._cache.values()
        if np.array_equal(entry.indptr, other.indptr) and np.array_equal(entry.indices, other.indices)
    ]
    assert len(matching) == 1 and matching[0] is not first
    assert first in native_direct._cache.values(), 'a different pattern must not overwrite the held analysis'
    assert np.linalg.norm(other @ answer - rhs) < 1e-12
    assert np.linalg.norm(a @ held.solve(rhs) - rhs) < 1e-12


@needs_native
@pytest.mark.parametrize('move_rows', [False, True])
def test_the_cached_pattern_is_owned_when_the_callers_pattern_buffer_changes(monkeypatch, move_rows):
    monkeypatch.setattr(native_direct, '_cache', OrderedDict())
    a = sp.csr_matrix([[4.0, 0.2, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])
    rhs = np.array([1.0, 2.0, 3.0])
    native_direct.native_solve(a, rhs)
    first = next(iter(native_direct._cache.values()))
    original_indptr, original_indices = a.indptr.copy(), a.indices.copy()
    if move_rows:
        a.indptr[1] = 1
        a.indices[:] = [0, 0, 1, 2]
    else:
        a.indices[1] = 2
    answer = native_direct.native_solve(a, rhs)
    matching = [
        entry
        for entry in native_direct._cache.values()
        if np.array_equal(entry.indptr, a.indptr) and np.array_equal(entry.indices, a.indices)
    ]
    assert len(matching) == 1 and matching[0] is not first
    np.testing.assert_array_equal(first.indptr, original_indptr)
    np.testing.assert_array_equal(first.indices, original_indices)
    assert np.linalg.norm(a @ answer - rhs) < 1e-12


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


@needs_native
def test_unit_constraint_rows_use_the_exact_symmetric_free_block_for_every_rhs():
    a = sp.csr_matrix([[1.0, 0.0, 0.0], [-2.0, 4.0, -1.0], [0.5, -1.0, 3.0]])
    rhs = np.array([[2.0, -3.0], [1.0, 5.0], [-4.0, 2.0]])
    factor = native_direct.native_factor(a)
    assert factor._csr.shape == (2, 2)
    assert factor.kind == 'cholesky'
    assert np.max(np.abs(a @ factor.solve(rhs) - rhs)) < 1e-12
    assert np.max(np.abs(a @ factor.solve(rhs[:, 0]) - rhs[:, 0])) < 1e-12
    a.data[:] *= 7.0
    original = sp.csr_matrix([[1.0, 0.0, 0.0], [-2.0, 4.0, -1.0], [0.5, -1.0, 3.0]])
    assert np.max(np.abs(original @ factor.solve(rhs) - rhs)) < 1e-12


@needs_native
def test_a_reduced_held_factor_restores_its_generation_and_its_own_boundary_columns():
    a = sp.csr_matrix([[1.0, 0.0, 0.0], [-2.0, 4.0, -1.0], [0.5, -1.0, 3.0]])
    other = sp.csr_matrix([[1.0, 0.0, 0.0], [7.0, 8.0, -2.0], [-3.0, -2.0, 6.0]])
    held = native_direct.native_factor(a)
    rhs = np.array([[2.0, -3.0], [1.0, 5.0], [-4.0, 2.0]])
    second = native_direct.native_factor(other)
    assert held._entry is second._entry
    assert held._generation != held._entry.generation
    assert np.max(np.abs(a @ held.solve(rhs) - rhs)) < 1e-12
    assert np.max(np.abs(other @ second.solve(rhs) - rhs)) < 1e-12


@needs_native
@pytest.mark.parametrize('free', [np.array([[2.0, -1.0], [-0.5, 3.0]]), np.array([[1.0, 2.0], [2.0, 1.0]])])
def test_constraint_elimination_never_declares_a_nonsymmetric_or_indefinite_block_spd(free):
    a = sp.csr_matrix(np.block([[np.ones((1, 1)), np.zeros((1, 2))], [np.ones((2, 1)), free]]))
    factor = native_direct.native_factor(a)
    assert factor.kind == 'lu'
    rhs = np.array([[3.0, -2.0], [2.0, 7.0], [-1.0, 4.0]])
    assert np.max(np.abs(a @ factor.solve(rhs) - rhs)) < 1e-12


@needs_native
def test_all_unit_rows_remain_a_real_native_factor_not_an_empty_system():
    a = sp.eye(3, format='csr')
    rhs = np.arange(6.0).reshape(3, 2)
    factor = native_direct.native_factor(a)
    assert np.array_equal(factor.solve(rhs), rhs)


def test_the_superlu_fallback_is_logged_and_counted(monkeypatch, caplog):
    monkeypatch.setenv('NUMERICS_LAB_DIRECT_NATIVE', '0')
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
