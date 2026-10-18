"""Local index constraints retain their actual authority across arbitrary row units."""

import numpy as np
import pytest

from numerics_lab.linear.border_rank import border_affine_space


@pytest.mark.parametrize('scale', [1e-18, 1.0, 1e18])
def test_independent_weak_equation_is_not_discarded_by_a_global_rank_cutoff(scale):
    matrix = np.array([[1, 1, 0], [0, scale, 1], [0, scale, 0]], dtype=float)
    rhs = np.array([3.0, 2 * scale + 4.0, 2 * scale])
    answer, free = border_affine_space(matrix, rhs, mandatory_rows=(2,))
    assert free.shape == (3, 0)
    # The bar is relative: the elimination order (and so the last ulp) is the LAPACK build's, not the claim's.
    np.testing.assert_allclose(answer[:2], [1.0, 2.0], rtol=4 * np.finfo(float).eps, atol=0.0)
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


def test_original_homogeneous_anchors_propagate_before_driven_reconstruction():
    matrix = np.array([
        [0.6963106238227914, 0, 0, -0.211041953783194, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0.6152877401519846, 0, 0, 0, 0, 0, 0],
        [0, -0.9165151389911681, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.7177405625652734, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.9756235748347674, 0, 0, -0.32444284226152514, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0.9459053029269173, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0.4000000000000001, 0, -0.7595276759684509, 0, -0.21945076948288172, 0, 0, 0, 0],
    ])
    answer, directions = border_affine_space(matrix, np.array([0, 0, -3, 0, 0, 0, 0, -0.02259923024842078, 0]))
    assert np.array_equal(answer[[0, 2, 3]], np.zeros(3))
    assert np.array_equal(directions[[0, 2, 3]], np.zeros((3, directions.shape[1])))


def test_zero_anchor_cannot_erase_a_tiny_conflicting_original_equation():
    with pytest.raises(ValueError, match='contradict'):
        border_affine_space(np.array([[1.0, 0], [1.0, 0]]), np.array([0.0, 1e-30]))


def test_tiny_driven_singleton_is_not_a_zero_anchor():
    answer, directions = border_affine_space(np.array([[1.0, 0]]), np.array([1e-30]))
    assert answer[0] == 1e-30
    assert np.array_equal(directions[:, 0], [0.0, 1.0])


def test_one_free_column_does_not_allocate_a_full_identity(monkeypatch):
    matrix = np.zeros((64, 65))
    matrix[np.arange(64), np.arange(64)] = 1

    def forbidden_identity(*_args, **_kwargs):
        pytest.fail('free directions must allocate only their actual column count')

    monkeypatch.setattr(np, 'eye', forbidden_identity)
    answer, directions = border_affine_space(matrix, np.zeros(64))
    assert np.array_equal(answer, np.zeros(65))
    assert directions.shape == (65, 1)
    assert directions[-1, 0] == 1
    assert not directions[:-1].any()


def test_only_explicit_inherited_row_allowance_can_cover_a_derived_anchor_remainder():
    matrix = np.array([[1.0, 0], [1.0, 0]])
    rhs = np.array([0.0, 3e-16])
    answer, _directions = border_affine_space(matrix, rhs, inherited_allowance=np.array([0.0, 4e-16]))
    assert answer[0] == 0
    with pytest.raises(ValueError, match='contradicts actual'):
        border_affine_space(matrix, rhs, inherited_allowance=np.array([0.0, 2e-16]))


def test_tiny_physical_source_is_not_covered_by_its_relative_evaluation_allowance():
    with pytest.raises(ValueError, match='contradicts actual'):
        border_affine_space(
            np.array([[1.0, 0], [1.0, 0]]),
            np.array([0.0, 1e-30]),
            inherited_allowance=np.array([0.0, 1e-40]),
        )


@pytest.mark.parametrize(
    'allowance',
    [
        [False, True],
        [0.0, True],
        [0.0, 1j],
        [0.0],
        [0.0, np.inf],
        [0.0, np.nan],
        [0.0, -1.0],
    ],
)
def test_inherited_allowance_requires_real_nonboolean_physical_row_bounds(allowance):
    with pytest.raises(ValueError, match='inherited border allowance'):
        border_affine_space(np.eye(2), np.zeros(2), inherited_allowance=allowance)


def test_mandatory_independent_anchor_rows_retain_their_original_authority():
    answer, directions = border_affine_space(
        np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        np.zeros(2),
        mandatory_rows=(0, 1),
    )
    assert np.array_equal(answer, np.zeros(3))
    assert np.array_equal(directions[:, 0], [0.0, 0.0, 1.0])


def test_a_computed_entry_inside_its_supplied_error_bound_is_never_a_pivot():
    # A Schur border is COMPUTED: an entry no larger than its own formation error is noise
    # left by a cancellation (an exactly dependent row), and pivoting on it invents authority.
    matrix = np.array([[1.0, 0.0], [0.0, 1e-17]])
    rhs = np.array([1.0, 1e-17])
    answer, directions = border_affine_space(matrix, rhs)
    assert answer[1] == 1.0
    errors = np.array([[0.0, 0.0], [0.0, 1e-16]])
    answer, directions = border_affine_space(matrix, rhs, entry_errors=errors)
    assert np.array_equal(answer, [1.0, 0.0])
    assert np.array_equal(directions[:, 0], [0.0, 1.0])


@pytest.mark.parametrize('errors', [np.zeros((2, 3)), np.array([[0.0, -1.0], [0.0, 0.0]]), np.full((2, 2), np.nan)])
def test_entry_errors_require_a_finite_nonnegative_aligned_bound(errors):
    with pytest.raises(ValueError, match='entry error'):
        border_affine_space(np.eye(2), np.zeros(2), entry_errors=errors)


def test_a_computed_entry_inside_its_error_bound_never_anchors_a_coordinate_at_zero():
    """dcm_pmdc_rect / dcm_pmdc_pear / dcm_spm_bldc (2026-10-17): a Schur row ``-9.1e-12 x0 - 137 x1 = 0`` whose
    ``x0`` entry is inside its 1.5e-9 formation bound. Once ``x1`` is anchored, the exact-support scan read the
    row as the singleton ``x0 = 0``, chained it through ``x0 - x2 = 0`` and refused a driven row as a contradicted
    stored state. A noise entry is a POSSIBLE zero, so it proves nothing; ``x0`` stays free and the row admits.
    """
    matrix = np.array([
        [-9.094947017729282e-12, -137.24416462050308, 0.0],
        [0.0, 4.46981229743835e-06, 0.0],
        [1.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
    ])
    rhs = np.array([0.0, 0.0, 0.0, 14.56])
    errors = np.zeros_like(matrix)
    errors[0, 0] = 1.5239131365770421e-09
    answer, _free = border_affine_space(matrix, rhs, entry_errors=errors)
    assert answer[2] == pytest.approx(14.56)
    assert answer[1] == 0.0
    # PLANTED CONTROL: the same entry, taken as EXACT, does anchor x0 -- and the driven row is a contradiction.
    with pytest.raises(ValueError, match='homogeneous anchors'):
        border_affine_space(matrix, rhs)
