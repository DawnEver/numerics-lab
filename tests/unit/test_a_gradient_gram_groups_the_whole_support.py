"""A support is its complete column tuple, not the first column's tensor-span shortcut.

The oracle emits every point's COO entries independently. Integer gradients and binary-exact
weights keep the comparison exact: no tolerance can conceal a contribution assigned to a wrong DOF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from motronics.contracts.ir.discrete_space import EvaluatedBasis
from motronics.gauss import quadrature
from motronics.gauss.spline_basis import SplineBasisError


@dataclass
class _Evaluation:
    """Only the protocol's declared members; no dense conversion or spline-specific methods."""

    values: NDArray[np.float64]
    gradients: NDArray[np.float64] | None
    hessians: NDArray[np.float64] | None
    det_jacobian: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    columns: NDArray[np.intp]
    n_basis: int


def _evaluation(columns: list[list[int]]) -> EvaluatedBasis:
    ids = np.array(columns, dtype=np.intp).reshape(-1, 2)
    count = len(ids)
    gradients = np.arange(1, count * 4 + 1, dtype=float).reshape(count, 2, 2)
    return _Evaluation(
        values=np.ones((count, 2)),
        gradients=gradients,
        hessians=None,
        det_jacobian=np.ones(count),
        jacobian=np.broadcast_to(np.eye(2), (count, 2, 2)).copy(),
        columns=ids,
        n_basis=3,
    )


def _pointwise_oracle(
    values: EvaluatedBasis,
    ids: NDArray[np.intp],
    weights: NDArray[np.float64],
    diagonal: NDArray[np.float64] | None = None,
):
    """Scatter each point and each basis pair without grouping or using the production contraction."""
    assert values.gradients is not None
    coefficients = np.ones(values.gradients.shape[2]) if diagonal is None else diagonal
    result = np.zeros((int(ids.max()) + 1, int(ids.max()) + 1))
    for q, columns in enumerate(values.columns):
        for i, local_i in enumerate(columns):
            for j, local_j in enumerate(columns):
                dot = sum(
                    float(a) * float(b) * float(coefficient)
                    for a, b, coefficient in zip(
                        values.gradients[q, i], values.gradients[q, j], coefficients, strict=True
                    )
                )
                result[ids[local_i], ids[local_j]] += weights[q] * dot
    return result


def _assembled(triplets, size: int):
    rows, columns, data = triplets
    matrix = np.zeros((size, size))
    np.add.at(matrix, (rows, columns), data)
    return matrix


@pytest.mark.parametrize('batch_points', [1, 3, 100])
def test_shared_first_columns_do_not_merge_distinct_supports(monkeypatch, batch_points: int) -> None:
    values = _evaluation([[0, 1], [0, 2], [0, 1], [0, 2]])
    ids = np.array([7, 2, 11], dtype=np.intp)
    weights = np.array([0.5, 1.0, 2.0, 4.0])
    monkeypatch.setattr(quadrature, 'GRAM_BATCH_POINTS', batch_points)
    triplets = quadrature.gradient_gram_triplets(values, ids, weights)
    assert len(triplets[2]) == 8, 'two complete supports must emit two 2-by-2 blocks'
    np.testing.assert_array_equal(_assembled(triplets, 12), _pointwise_oracle(values, ids, weights))


def test_a_protocol_record_with_one_support_needs_no_dense_conversion() -> None:
    values = _evaluation([[0, 2], [0, 2]])
    ids = np.array([7, 2, 11], dtype=np.intp)
    weights = np.array([0.5, 2.0])
    triplets = quadrature.gradient_gram_triplets(values, ids, weights)
    np.testing.assert_array_equal(_assembled(triplets, 12), _pointwise_oracle(values, ids, weights))


def test_an_empty_quadrature_emits_typed_empty_coo_arrays() -> None:
    values = _evaluation([])
    rows, columns, data = quadrature.gradient_gram_triplets(values, np.array([7, 2, 11], dtype=np.intp), np.empty(0))
    for indices in (rows, columns):
        assert indices.shape == (0,)
        assert indices.dtype == np.dtype(np.intp)
    assert data.shape == (0,)
    assert data.dtype == np.dtype(np.float64)


def test_batch_boundaries_preserve_the_existing_unique_first_column_triplets(monkeypatch) -> None:
    values = _evaluation([[1, 2], [0, 1], [1, 2], [0, 1]])
    ids = np.array([7, 2, 11], dtype=np.intp)
    weights = np.array([0.5, 1.0, 2.0, 4.0])
    monkeypatch.setattr(quadrature, 'GRAM_BATCH_POINTS', 100)
    whole = quadrature.gradient_gram_triplets(values, ids, weights)
    monkeypatch.setattr(quadrature, 'GRAM_BATCH_POINTS', 1)
    batched = quadrature.gradient_gram_triplets(values, ids, weights)
    for before, after in zip(whole, batched, strict=True):
        assert before.tobytes() == after.tobytes()
    np.testing.assert_array_equal(_assembled(batched, 12), _pointwise_oracle(values, ids, weights))


def test_missing_gradients_still_refuse_before_empty_quadrature_is_accepted() -> None:
    values = _evaluation([])
    values.gradients = None
    with pytest.raises(SplineBasisError, match='derivative order 0'):
        quadrature.gradient_gram_triplets(values, np.array([7, 2, 11], dtype=np.intp), np.empty(0))


@pytest.mark.parametrize('batch_points', [1, 3, 100])
def test_diagonal_gradient_gram_keeps_both_physical_axes_and_the_complete_support(monkeypatch, batch_points) -> None:
    values = _evaluation([[0, 1], [0, 2], [0, 1], [0, 2]])
    ids = np.array([7, 2, 11], dtype=np.intp)
    weights = np.array([0.5, 1.0, 2.0, 4.0])
    monkeypatch.setattr(quadrature, 'GRAM_BATCH_POINTS', batch_points)
    matrices = []
    for diagonal in (np.array([2.0, 0.5]), np.array([0.5, 2.0])):
        triplets = quadrature.gradient_gram_triplets(values, ids, weights, diagonal=diagonal)
        assert len(triplets[2]) == 8
        actual = _assembled(triplets, 12)
        np.testing.assert_array_equal(actual, _pointwise_oracle(values, ids, weights, diagonal))
        matrices.append(actual)
    assert not np.array_equal(*matrices), 'swapping unequal physical-axis coefficients must change this operator'


def test_an_all_ones_diagonal_preserves_the_original_gradient_triplets_byte_for_byte() -> None:
    values = _evaluation([[1, 2], [0, 1], [1, 2], [0, 1]])
    assert values.gradients is not None
    # Noninteger samples discriminate contraction reassociation as well as support numbering.
    values.gradients = np.random.default_rng(7).normal(size=values.gradients.shape)
    ids = np.array([7, 2, 11], dtype=np.intp)
    weights = np.array([0.5, 1.0, 2.0, 4.0])
    original = quadrature.gradient_gram_triplets(values, ids, weights)
    diagonal = quadrature.gradient_gram_triplets(values, ids, weights, diagonal=np.ones(2))
    for before, after in zip(original, diagonal, strict=True):
        assert before.dtype == after.dtype
        assert before.tobytes() == after.tobytes()


def _value_pointwise_oracle(values: EvaluatedBasis, ids: NDArray[np.intp], weights: NDArray[np.float64]):
    """Each value pair is integrated and scattered independently, including repeated global IDs."""
    result = np.zeros((int(ids.max()) + 1, int(ids.max()) + 1))
    for q, columns in enumerate(values.columns):
        for i, local_i in enumerate(columns):
            for j, local_j in enumerate(columns):
                result[ids[local_i], ids[local_j]] += weights[q] * values.values[q, i] * values.values[q, j]
    return result


@pytest.mark.parametrize('batch_points', [1, 3, 100])
@pytest.mark.parametrize('capacity', [0.0, 4.0])
def test_value_gram_assembles_capacity_on_support_without_gradients(monkeypatch, batch_points, capacity) -> None:
    values = _evaluation([[0, 1], [0, 2], [0, 1], [0, 2]])
    values.values = np.array([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.125, 0.875]])
    values.gradients = None
    ids = np.array([7, 2, 11], dtype=np.intp)
    measure = np.array([0.5, 1.0, 2.0, 4.0])
    weights = capacity * measure
    monkeypatch.setattr(quadrature, 'GRAM_BATCH_POINTS', batch_points)
    triplets = quadrature.value_gram_triplets(values, ids, weights)
    assert len(triplets[2]) == 8
    matrix = _assembled(triplets, 12)
    np.testing.assert_array_equal(matrix, _value_pointwise_oracle(values, ids, weights))
    assert matrix.sum() == capacity * measure.sum(), 'partition of unity integrates exactly the declared capacity'


def test_empty_value_quadrature_emits_typed_empty_coo_without_requesting_gradients() -> None:
    values = _evaluation([])
    values.gradients = None
    rows, columns, data = quadrature.value_gram_triplets(values, np.array([7, 2, 11], dtype=np.intp), np.empty(0))
    for indices in (rows, columns):
        assert indices.shape == (0,)
        assert indices.dtype == np.dtype(np.intp)
    assert data.shape == (0,)
    assert data.dtype == np.dtype(np.float64)


def test_value_gram_batching_preserves_identical_coo_bytes(monkeypatch) -> None:
    values = _evaluation([[0, 1], [0, 2], [0, 1], [0, 2]])
    values.values = np.array([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.125, 0.875]])
    ids = np.array([7, 2, 11], dtype=np.intp)
    weights = np.array([0.5, 1.0, 2.0, 4.0])
    monkeypatch.setattr(quadrature, 'GRAM_BATCH_POINTS', 100)
    whole = quadrature.value_gram_triplets(values, ids, weights)
    monkeypatch.setattr(quadrature, 'GRAM_BATCH_POINTS', 1)
    batched = quadrature.value_gram_triplets(values, ids, weights)
    for before, after in zip(whole, batched, strict=True):
        assert before.tobytes() == after.tobytes()
