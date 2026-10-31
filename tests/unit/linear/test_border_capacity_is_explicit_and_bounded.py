"""The measured 1298-column border fits a bounded dense work envelope, not a new rank bar."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import border_rank, initial_storage_border


@pytest.mark.parametrize('size', [513, 1298])
def test_actual_connected_homogeneous_border_above_old_column_cap_is_solved(size):
    matrix = sp.vstack((sp.eye(size), sp.csr_matrix(np.ones((1, size)))), format='csr')
    forcing = np.zeros(size + 1)
    errors = sp.csr_matrix(matrix.shape)
    answer, directions, components = initial_storage_border._component_affine(matrix, forcing, errors, forcing)
    assert len(components) == 1 and len(components[0][0]) == size
    np.testing.assert_array_equal(answer, np.zeros(size))
    assert directions.shape == (size, 0)
    np.testing.assert_array_equal(matrix @ answer, forcing)
    direct, free = border_rank.border_affine_space(matrix.toarray(), forcing, entry_errors=errors.toarray())
    np.testing.assert_array_equal(direct, answer)
    assert free.shape == (size, 0)


def test_column_and_entry_envelopes_are_shared_and_finite():
    assert 1298 < border_rank.MAX_AFFINE_COLUMNS <= 2048
    assert 1298 * 1926 <= border_rank.MAX_AFFINE_ENTRIES <= 2048**2
    with pytest.raises(ValueError, match='finite bounded matrix'):
        border_rank.border_affine_space(np.zeros((1, border_rank.MAX_AFFINE_COLUMNS + 1)), np.zeros(1))


def test_driven_connected_border_above_old_cap_keeps_original_pivot_and_residual_authority():
    size = 513
    matrix = np.vstack((2 * np.eye(size), np.ones((1, size))))
    expected = np.where(np.arange(size) % 2, -1.0, 1.0)
    rhs = matrix @ expected
    assert np.all(rhs != 0), 'this must exercise driven elimination, not homogeneous-zero anchoring'
    answer, directions = border_rank.border_affine_space(matrix, rhs)
    np.testing.assert_array_equal(answer, expected)
    np.testing.assert_array_equal(matrix @ answer, rhs)
    assert directions.shape == (size, 0)


def test_tall_thin_input_over_entry_envelope_refuses_without_running_elimination():
    rows = border_rank.MAX_AFFINE_ENTRIES // 2 + 1
    with pytest.raises(ValueError, match='finite bounded matrix'):
        border_rank.border_affine_space(np.zeros((rows, 2)), np.zeros(rows))


def test_actual_connected_component_over_shared_envelope_is_still_refused():
    size = border_rank.MAX_AFFINE_COLUMNS + 1
    matrix = sp.vstack((sp.eye(size), sp.csr_matrix(np.ones((1, size)))), format='csr')
    with pytest.raises(NotImplementedError, match='bounded actual Schur component rank work'):
        initial_storage_border._component_affine(
            matrix, np.zeros(size + 1), sp.csr_matrix(matrix.shape), np.zeros(size + 1)
        )
