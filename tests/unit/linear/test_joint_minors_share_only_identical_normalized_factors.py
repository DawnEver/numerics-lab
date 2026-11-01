"""Original minors retain their own refinements while identical factor inputs share LU."""

import weakref

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import initial_storage_border


def _joint(*, different=False, singular=False):
    matrix = np.array([
        [4.0, 0.5, -0.25, 8.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 4.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, -1.0, 0.5, 0.75, 6.0, 2.0],
        [0.5, 0.25, -1.0, -0.5, 0.5, 2.0],
    ])
    if different:
        matrix[4, 5] += 0.125
    if singular:
        matrix[5, 4:] = matrix[4, 4:]
    return sp.csr_matrix(matrix)


def _solver(matrix):
    return initial_storage_border._triangular_joint_solver(
        matrix, sp.diags([2.0, 0.0, 0.0], format='csr'), np.array([0]), np.array([0]), np.arange(3), np.array([0, 1])
    )


@pytest.mark.parametrize('different', [False, True])
def test_identical_normalized_inputs_share_only_the_factor_not_raw_rhs_scaling(monkeypatch, different):
    matrix = _joint(different=different)
    original = initial_storage_border.splu
    factors = []

    def observe(operator):
        assert operator.shape[0] <= 2
        factors.append(operator.copy())
        return original(operator)

    monkeypatch.setattr(initial_storage_border, 'splu', observe)
    solve = _solver(matrix)
    rng = np.random.default_rng(42)
    for _ in range(4):
        forcing = rng.standard_normal(6)
        before = forcing.copy()
        actual = solve(forcing)
        expected = np.linalg.solve(matrix.toarray(), forcing)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
        assert np.all(abs(matrix @ actual - forcing) <= 1e-13 * (abs(matrix) @ abs(actual) + abs(forcing)))
        np.testing.assert_array_equal(forcing, before)
    assert len(factors) == (3 if different else 2)


def test_a_genuinely_singular_different_minor_still_requires_actual_rank_authority():
    with pytest.raises(ValueError, match='independent actual equation authority'):
        _solver(_joint(singular=True))


def test_a_single_ulp_normalized_difference_is_not_reused(monkeypatch):
    matrix = _joint()
    matrix[5, 4] = np.nextafter(0.5, 1.0)
    original = initial_storage_border.splu
    shapes = []

    def observe(operator):
        shapes.append(operator.shape)
        return original(operator)

    monkeypatch.setattr(initial_storage_border, 'splu', observe)
    solve = _solver(matrix)
    assert shapes == [(1, 1), (2, 2), (2, 2)]
    forcing = np.arange(6.0)
    np.testing.assert_allclose(matrix @ solve(forcing), forcing, rtol=0, atol=1e-14)


def test_normalized_comparison_keys_do_not_escape_with_the_solve_closure(monkeypatch):
    original = initial_storage_border.splu
    refs = []

    def observe(operator):
        refs.append(weakref.ref(operator))
        return original(operator)

    monkeypatch.setattr(initial_storage_border, 'splu', observe)
    solve = _solver(_joint())
    assert len(refs) == 2 and all(reference() is None for reference in refs)
    forcing = np.arange(6.0)
    np.testing.assert_allclose(_joint() @ solve(forcing), forcing, rtol=0, atol=1e-14)
