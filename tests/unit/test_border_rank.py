"""Local index constraints retain their actual authority across arbitrary row units."""

import numpy as np
import pytest

from motronics.gauss.linear.border_rank import border_affine_space


@pytest.mark.parametrize('scale', [1e-18, 1.0, 1e18])
def test_independent_weak_equation_is_not_discarded_by_a_global_rank_cutoff(scale):
    matrix = np.array([[1, 1, 0], [0, scale, 1], [0, scale, 0]], dtype=float)
    rhs = np.array([3.0, 2 * scale + 4.0, 2 * scale])
    answer, free = border_affine_space(matrix, rhs, mandatory_rows=(2,))
    assert free.shape == (3, 0)
    assert np.array_equal(answer[:2], [1.0, 2.0])
    # At enormous row units the input itself cannot retain the added 4.0.
    assert answer[2] == rhs[1] - 2 * scale


def test_actual_noncoordinate_free_direction_is_retained_not_selected_as_unique():
    matrix = np.array([[1, 1, 0], [0, 1e-18, -0.5e-18]])
    answer, free = border_affine_space(matrix, np.zeros(2))
    assert np.array_equal(answer, np.zeros(3))
    assert np.array_equal(free[:, 0], [-0.5, 0.5, 1.0])
    assert np.array_equal(matrix @ free, np.zeros((2, 1)))


def test_dependent_mandatory_storage_rows_cannot_create_pivot_authority():
    with pytest.raises(ValueError, match='mandatory border constraints'):
        border_affine_space(np.array([[1.0, 0.0], [1.0, 0.0]]), np.zeros(2), mandatory_rows=(0, 1))
