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
        if len(blocks) == size + 1 and all(block.shape == (2 * size, 1) for block in blocks):
            records.append(({block.format for block in blocks}, sum(block.indptr.nbytes for block in blocks)))
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
    assert len(records) == 2
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
