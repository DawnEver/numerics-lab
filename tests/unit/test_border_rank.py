"""Local index constraints retain their actual authority across arbitrary row units."""

import time

import numpy as np
import pytest

from numerics_lab.linear.border_rank import border_affine_space, homogeneous_zero_coordinates


def test_a_computed_borders_rank_does_not_depend_on_its_forcing():
    """A six-column reduction of the actual native case_a initial border.

    Preferring an evolving zero RHS pivot lost an independent physical column:
    the same computed entries and their same bounds acquired a false null mode.
    """
    matrix = np.array([
        [6e-2, 4.4e-8, 1.1e-8, 5e-8, 2e-5, 8e-3],
        [8.58e-3, 1.058508e-8, 3.702412e-8, 2.356137e-7, 9.94e-5, 1.207678e-3],
        [2.25e-3, 2.316786e-8, 1.895425e-7, 1.222831e-6, 5.161529e-4, 3.165049e-4],
        [5e-4, 1.115594e-7, 9.805451e-7, 6.330469e-6, 2.672953e-3, 7e-5],
        [0, 2.95e-5, 2.604154e-4, 1.681790e-3, 7.17e-1, 0],
        [-0.6, -0.6, 0, -0.6, 0, 0],
    ])
    errors = 64 * np.finfo(float).eps * abs(matrix)
    _, homogeneous = border_affine_space(matrix, np.zeros(6), entry_errors=errors)
    _, driven = border_affine_space(matrix, np.array([1, 1, 1, 1, 1, 0]), entry_errors=errors)
    assert homogeneous.shape == (6, 0)
    assert driven.shape == homogeneous.shape


def test_free_directions_are_reconstructed_from_original_selected_equations():
    """The native border's mixed scales also contaminate eliminated homogeneous columns."""
    matrix = np.array([
        [2e-3, 7e-1, 1e-7, 3e-7, 2e-6, -8e2],
        [5e-4, 1e-1, 2e-7, 1e-6, 8e-6, -8e6],
        [3e-5, 8e-3, 2e-4, 1e-3, 9e-3, -8e6],
        [6e-4, 1e-1, 1e1, 3e1, 1e1, -2e1],
    ])
    _, directions = border_affine_space(matrix, np.ones(4))
    assert directions.shape == (6, 2)
    # Each original dot product's own arithmetic bar, not a mixed-unit global norm.
    bound = matrix.shape[1] * np.finfo(float).eps * (abs(matrix) @ abs(directions))
    assert np.all(abs(matrix @ directions) <= bound)


@pytest.mark.parametrize('permuted', [False, 'rows', 'columns'])
def test_unreachable_state_coordinates_of_actual_body_border_directions_are_exact_zero(permuted):
    """Original pinned-body Schur support separates state from the free rate forcing.

    Captured from the actual public field compiler's joint admission (2026-10-20,
    measure-8620). No scale threshold or numerical rank inference licenses zeros.
    """
    matrix = np.array([
        [1.4432989690721669e-5, -0.9278350515463915, 0, 0, 0, 0],
        [0, 1.824e-5, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [5517055.610463657, 27585278052.318295, 0, 1.4432989690721696e-5, -0.9278350515463915, 0],
        [-0.09278350515463916, -463.91752577319573, 0, 0, 0, 1],
    ])
    rhs = np.array([1, 2, 3, 3, 4, 5], dtype=float)
    order = np.array([4, 1, 5, 2, 0, 3])
    columns = order if permuted == 'columns' else np.arange(6)
    if permuted == 'rows':
        matrix, rhs = matrix[order], rhs[order]
    matrix = matrix[:, columns]
    _, directions = border_affine_space(matrix, rhs)
    assert directions.shape == (6, 1)
    physical = directions[np.argsort(columns)]
    np.testing.assert_array_equal(physical[[0, 1, 2, 5]], np.zeros((4, 1)))
    assert np.count_nonzero(physical[[3, 4]]) == 2
    bound = matrix.shape[1] * np.finfo(float).eps * (abs(matrix) @ abs(directions))
    assert np.all(abs(matrix @ directions) <= bound)


def test_cyclic_dense_state_dependencies_retain_genuine_free_state_directions():
    matrix = np.array([[2, 1, 1], [1, 3, -2]], dtype=float)
    _, directions = border_affine_space(matrix, np.array([3.0, 4.0]))
    assert directions.shape == (3, 1)
    assert np.all(directions != 0), 'every state lies on an actual path from the free forcing'
    np.testing.assert_allclose(matrix @ directions, 0, rtol=0, atol=1e-14)


def test_nonzero_cancellation_in_a_reachable_state_direction_is_not_snapped_to_zero():
    delta = 2.0**-40
    matrix = np.array([[1, 1, 1], [0, 1, 1 - delta]], dtype=float)
    _, directions = border_affine_space(matrix, np.array([2.0, 1.0]))
    assert directions.shape == (3, 1)
    assert directions[0, 0] != 0
    assert abs(directions[0, 0]) == delta


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
    """case_e / case_f / case_g (2026-10-17): a Schur row ``-9.1e-12 x0 - 137 x1 = 0`` whose
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


def test_a_full_rank_homogeneous_block_is_an_exact_zero_anchor_like_a_singleton():
    # Rows 0-1 are homogeneous and hold columns 1-2 at full exact rank; row 2 is driven and shares
    # column 1. Exact arithmetic says x1 = x2 = 0; the block must not inherit row 2's cancellation.
    matrix = np.array([
        [0, 0.522233074687417, -0.6123726661276055],
        [0, -0.42640151992964986, 0.5000001881467859],
        [0.4999999687499717, -0.5222329441291974, 0.0],
    ])
    remaining, consumed = homogeneous_zero_coordinates(matrix, np.array([0.0, 0.0, -0.07]))
    assert remaining.tolist() == [True, False, False]
    assert consumed == {0, 1}


def test_a_rank_deficient_homogeneous_block_proves_nothing():
    remaining, consumed = homogeneous_zero_coordinates(np.array([[1.0, -1.0], [2.0, -2.0]]), np.zeros(2))
    assert remaining.tolist() == [True, True]
    assert consumed == set()


def test_a_computed_block_anchors_only_where_its_rank_survives_its_entry_errors():
    """Exact rank proves only exact entries. A computed block anchors where its least singular value (~0.16
    here) clears the singleton's reliability factor (16 x 2) times its errors' norm, and not where a
    perturbation inside those errors could make it singular.
    """
    matrix = np.array([[1.0, 2.0], [3.0, 5.0]])
    small, large = np.full((2, 2), 1e-6), np.full((2, 2), 1e-2)
    assert homogeneous_zero_coordinates(matrix, np.zeros(2), entry_errors=small)[1] == {0, 1}
    remaining, consumed = homogeneous_zero_coordinates(matrix, np.zeros(2), entry_errors=large)
    assert remaining.tolist() == [True, True]
    assert consumed == set()


def _sparse_block(n, seed, per_row=8):
    """A sparse ``(n, n)`` float block of mixed binary exponents, the shape a ring's homogeneous anchor block takes."""
    rng = np.random.default_rng(seed)
    block = np.zeros((n, n))
    for row in range(n):
        columns = rng.choice(n, size=per_row, replace=False)
        block[row, columns] = rng.standard_normal(per_row) * 2.0 ** rng.integers(-29, 23, size=per_row)
    return block


def test_a_large_sparse_full_rank_homogeneous_block_is_judged_in_floating_time():
    """The exact verdict on a 244-column block -- the k = 1 interpolated ring's -- costs what a float pass costs.

    Rational elimination over its float entries took 52 s of that ring's 61 s fea march (MEASURED 2026-10-18,
    2.7M ``Fraction`` operations, ``math.gcd`` alone 30 s).
    """
    block = _sparse_block(244, 7)
    block[np.arange(244), np.arange(244)] += 1.0  # nonsingular: the diagonal dominates every row's tail
    start = time.perf_counter()
    remaining, consumed = homogeneous_zero_coordinates(block, np.zeros(244))
    assert time.perf_counter() - start < 2.0
    assert not remaining.any()
    assert consumed == set(range(244))


def test_an_exactly_dependent_block_of_mixed_exponents_proves_nothing():
    """A row the exact sum of two others (power-of-two multiples: no rounding) is rank deficient whatever prime
    reads it, and the verdict says so.
    """
    block = _sparse_block(40, 11, per_row=5)
    block[np.arange(40), np.arange(40)] += 1.0
    block[39] = 2.0 * block[3] + 0.5 * block[17]
    assert np.array_equal(block[39] - 2.0 * block[3], 0.5 * block[17])  # the dependence is exact in floats
    remaining, consumed = homogeneous_zero_coordinates(block, np.zeros(40))
    assert remaining.all()
    assert consumed == set()


@pytest.mark.parametrize('dependent', ['column', 'row'])
def test_a_large_exactly_dependent_block_is_refused_in_floating_time(dependent):
    """An exact dependence is CERTIFIED by exact null vectors, not found by rational elimination.

    Deficient under every prime, the verdict fell to ``Fraction`` elimination, whose cost grows like ``n^4.5``
    over float entries (MEASURED 2026-10-24: 0.37 s at 30 columns, 1.26 s at 40), and a downstream repo's bulk-storage
    knot history (``test_continuous_bulk_storage[True-True-False-False]``) sat in it for over 33 minutes.
    """
    block = _sparse_block(244, 7)
    block[np.arange(244), np.arange(244)] += 1.0
    if dependent == 'column':
        block[:, 243] = 2.0 * block[:, 3] + 0.5 * block[:, 17]
        assert np.array_equal(block[:, 243] - 2.0 * block[:, 3], 0.5 * block[:, 17])  # exact in floats
    else:
        block[243] = 2.0 * block[3] + 0.5 * block[17]
        assert np.array_equal(block[243] - 2.0 * block[3], 0.5 * block[17])
    start = time.perf_counter()
    remaining, consumed = homogeneous_zero_coordinates(block, np.zeros(244))
    assert time.perf_counter() - start < 2.0
    assert remaining.all()
    assert consumed == set()
