"""Bulk elimination preserves coefficients without a bulk-by-whole-border slab."""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import storage_border


def test_engine_is_loaded_from_this_source_tree():
    expected = Path(__file__).resolve().parents[3] / 'python/numerics_lab/linear/storage_border.py'
    assert Path(storage_border.__file__).resolve() == expected


def test_large_logical_border_uses_only_vector_rhs_and_sparse_products(monkeypatch):
    bulk_size, border_size, size = 20000, 97, 100003
    bulk = np.arange(bulk_size)
    border = np.arange(bulk_size, bulk_size + border_size)
    mass = sp.coo_matrix(
        (
            np.concatenate((np.full(bulk_size, 2.0), np.ones(border_size))),
            (np.concatenate((bulk, border)), np.concatenate((bulk, np.arange(border_size)))),
        ),
        shape=(size, size),
    ).tocsr()
    original_factor = storage_border.splu
    calls = []

    class Factor:
        def __init__(self, matrix) -> None:
            calls.append('factor')
            self.factor = original_factor(matrix)

        def solve(self, rhs, **kwargs):
            assert rhs.ndim == 1 and rhs.shape == (bulk_size,), 'whole-border dense RHS'
            calls.append('solve')
            return self.factor.solve(rhs, **kwargs)

    monkeypatch.setattr(storage_border, 'splu', Factor)
    for cls in (sp.csr_matrix, sp.csc_matrix):
        original = cls.toarray

        def bounded(matrix, *args, _original=original, **kwargs):
            assert min(matrix.shape) <= 1, 'bulk-by-border dense conversion'
            return _original(matrix, *args, **kwargs)

        monkeypatch.setattr(cls, 'toarray', bounded)
    rows = storage_border.storage_border_rows(mass, bulk_indices=bulk, border_rows=border)
    assert sp.isspmatrix_csr(rows.algebraic_rows)
    assert rows.algebraic_rows.shape == (size - bulk_size, size)
    assert rows.algebraic_rows[rows.algebraic_rows.shape[0] - 1, border[-1]] == 1.0
    assert rows.algebraic_rows[rows.algebraic_rows.shape[0] - 1, border_size - 1] == -0.5
    np.testing.assert_array_equal(rows.stored_rows, bulk)
    defect = rows.algebraic_rows @ mass
    defect.eliminate_zeros()
    assert defect.nnz == 0
    assert calls.count('factor') == 1 and calls.count('solve') == border_size


@pytest.mark.parametrize('bulk_size', [0, 1, 5])
@pytest.mark.parametrize('border_size', [0, 1, 4])
@pytest.mark.parametrize('asymmetric', [False, True])
def test_original_dense_elimination_equivalence(bulk_size, border_size, asymmetric):
    rng = np.random.default_rng(728)
    size = bulk_size + border_size + 2
    bulk = np.arange(bulk_size)
    border = np.arange(bulk_size, bulk_size + border_size)
    mass = np.zeros((size, size))
    block = rng.integers(-3, 4, size=(bulk_size, bulk_size)).astype(float)
    mass[:bulk_size, :bulk_size] = block @ block.T + np.eye(bulk_size)
    if asymmetric and bulk_size > 1:
        mass[0, 1] += 0.25
    mass[bulk_size : bulk_size + border_size, :bulk_size] = rng.integers(-4, 5, size=(border_size, bulk_size))
    if border_size:
        mass[border, bulk_size] = np.arange(1, border_size + 1)
    sparse = sp.csr_matrix(mass)
    remaining = np.arange(bulk_size, size)
    independent, null = storage_border._exact_border_dependencies(sparse[border][:, remaining])
    if bulk_size:
        factor = storage_border.splu(sparse[bulk][:, bulk].tocsc())
        correction = factor.solve(mass[np.ix_(border, bulk)].T.copy(), trans='T').T
    else:
        correction = np.zeros((border_size, 0))
    algebraic_indices = np.arange(bulk_size + border_size, size)
    selectors = np.eye(size)
    expected = np.vstack((selectors[algebraic_indices], null @ selectors[border] - null @ correction @ selectors[bulk]))
    result = storage_border.storage_border_rows(sparse, bulk_indices=bulk, border_rows=border)
    np.testing.assert_array_equal(result.stored_rows, np.concatenate((bulk, border[independent])))
    np.testing.assert_allclose(result.algebraic_rows.toarray(), expected, rtol=1e-14, atol=1e-15)
    assert sp.isspmatrix_csr(result.algebraic_rows)


@pytest.mark.parametrize('tiny', [1e-30, -1e-200, 1e-300])
def test_every_representable_tiny_coefficient_survives(tiny):
    mass = sp.csr_matrix([[1.0, 0.0, 0.0], [tiny, 0.0, 0.0], [-tiny, 0.0, 0.0]])
    result = storage_border.storage_border_rows(mass, bulk_indices=np.array([0]), border_rows=np.array([1, 2]))
    assert result.algebraic_rows[0, 0] == -tiny
    assert result.algebraic_rows[1, 0] == tiny
    assert result.algebraic_rows.nnz == 4


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_solution_is_still_refused(monkeypatch, bad):
    class Factor:
        def solve(self, rhs, **_kwargs):
            return np.full_like(rhs, bad)

    monkeypatch.setattr(storage_border, 'splu', lambda _matrix: Factor())
    with pytest.raises(ValueError, match='finite representable operands'):
        storage_border.storage_border_rows(sp.csr_matrix([[1.0, 0.0], [2.0, 0.0]]), bulk_indices=[0], border_rows=[1])


def test_singular_bulk_is_still_refused():
    with pytest.raises(ValueError, match='not numerically invertible'):
        storage_border.storage_border_rows(sp.csr_matrix((2, 2)), bulk_indices=[0], border_rows=[1])
