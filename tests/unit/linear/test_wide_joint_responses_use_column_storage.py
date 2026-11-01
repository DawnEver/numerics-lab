"""Real wide joint equations do not retain one field-sized CSR row pointer per column."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import initial_storage_border


@pytest.mark.parametrize('active', [False, True])
def test_wide_actual_joint_response_and_error_columns_are_compact_csc(monkeypatch, active):
    size = 600
    # A positive-definite arrow has genuinely field-wide responses without an
    # exponentially decaying chain whose last response underflows at this size.
    air = np.arange(1, size)
    rows = np.r_[air, np.zeros(size - 1, dtype=int)]
    cols = np.r_[np.zeros(size - 1, dtype=int), air]
    stiffness = sp.diags(np.r_[float(size + 1), 4 * np.ones(size - 1)], format='csr')
    stiffness += sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))
    mass = sp.diags(np.r_[2.0, np.zeros(size - 1)], format='csr')
    zero = sp.csr_matrix((size, size))
    matrix = sp.vstack(
        (sp.hstack((stiffness, mass)), sp.hstack((mass[:1], zero[:1])), sp.hstack((zero[1:], stiffness[1:]))),
        format='csr',
    )
    directions = sp.hstack((sp.eye(size), sp.csr_matrix((size, 1))), format='csr')
    extra = sp.lil_matrix((2 * size, size + 1))
    if active:
        extra[0, 0] = 0.25
    extra = extra.tocsr()
    effective = matrix + sp.hstack((extra[:, :size], sp.csr_matrix((2 * size, size))))
    rng = np.random.default_rng(17)
    expected = rng.standard_normal(2 * size)
    rhs = effective @ expected
    original = sp.hstack
    records = []

    def observe(blocks, *args, **kwargs):
        blocks = list(blocks)
        if len(blocks) == size + 1 and all(block.shape[1] == 1 for block in blocks):
            records.append(({block.format for block in blocks}, sum(block.indptr.nbytes for block in blocks)))
            assert all(block.shape[0] != 2 * size for block in blocks), 'retain projected images, not full responses'
        return original(blocks, *args, **kwargs)

    monkeypatch.setattr(sp, 'hstack', observe)
    answer, unique = initial_storage_border.joint_coordinate_sparse_correction(
        matrix,
        rhs,
        mass=mass,
        algebraic_rows=sp.eye(size, format='csr')[1:],
        stored_rows=np.array([0]),
        indices=np.arange(size),
        state_directions=directions,
        state_direction_columns=extra,
    )
    np.testing.assert_allclose(answer, expected, rtol=0, atol=5e-12)
    assert unique.all()
    assert np.all(abs(effective @ answer - rhs) <= 1e-10 * (abs(effective) @ abs(answer) + abs(rhs)))
    assert 1 <= len(records) <= 2
    assert all(formats == {'csc'} and pointer_bytes <= 16 * (size + 1) for formats, pointer_bytes in records)


def test_structurally_empty_columns_skip_compute_but_tiny_support_is_retained():
    columns = sp.csc_matrix(([1e-300], ([17], [1])), shape=(10000, 3))
    calls = []

    def compute(column):
        calls.append(column)
        value = np.zeros(10000)
        value[19] = 1e-300
        return value

    result = initial_storage_border._sparse_joint_columns(columns, compute)
    assert calls == [1]
    assert result.shape == columns.shape and result.format == 'csr'
    assert result.nnz == 1 and result[19, 1] == 1e-300


def test_no_columns_require_no_computation():
    calls = []
    result = initial_storage_border._sparse_joint_columns(sp.csc_matrix((10000, 0)), calls.append)
    assert result.shape == (10000, 0) and not result.nnz and not calls


def test_a_row_pointer_view_cannot_be_misread_as_column_support():
    with pytest.raises(ValueError, match='CSC'):
        initial_storage_border._sparse_joint_columns(sp.csr_matrix((3, 4)), None)


@pytest.mark.parametrize('width', [0, 3])
def test_projected_columns_use_the_declared_image_height(width):
    support = sp.csc_matrix(np.ones((4, width)))
    result = initial_storage_border._sparse_joint_columns(
        support, lambda column: np.array([1e-300, column + 1.0]), row_count=2
    )
    assert result.shape == (2, width) and result.format == 'csr'
    if width:
        np.testing.assert_array_equal(result.toarray(), [[1e-300] * width, np.arange(1.0, width + 1.0)])


def test_actual_nonzero_homogeneous_rate_lift_is_recomputed_not_discarded():
    jacobian = sp.csr_matrix([[4.0, 0.0, 0.0], [0.0, 5.0, 1.0], [0.0, 0.0, 1.0]])
    mass = sp.diags([2.0, 0.0, 0.0], format='csr')
    zero = sp.csr_matrix((3, 3))
    matrix = sp.vstack(
        (sp.hstack((jacobian, mass)), sp.hstack((mass[:1], zero[:1])), sp.hstack((zero[1:2], jacobian[1:2]))),
        format='csr',
    )
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    directions = sp.hstack((sp.eye(3), sp.csr_matrix((3, 510))), format='csr')
    answer, unique = initial_storage_border.joint_coordinate_sparse_correction(
        matrix,
        matrix @ expected,
        mass=mass,
        algebraic_rows=sp.csr_matrix([[0.0, 1.0, 0.0]]),
        stored_rows=np.array([0]),
        indices=np.arange(2),
        state_directions=directions,
    )
    np.testing.assert_allclose(answer[:4], expected[:4], rtol=0, atol=1e-12)
    np.testing.assert_allclose(matrix @ answer, matrix @ expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(unique, [True, True, True, True, False, False])
