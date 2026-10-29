"""Actual storage/state uniqueness without a dense field-sized SVD."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import initial_rate
from numerics_lab.linear.initial_rate import (
    _independent_exact_columns,
    _independent_exact_rows,
)
from numerics_lab.linear.initial_sparse import (
    sparse_complete_initial_rate,
    sparse_initial_pair,
)
from numerics_lab.linear.storage_border import storage_border_rows


@pytest.mark.parametrize('tiny_constraint', [0.0, 1e-30])
def test_only_exact_disposable_rate_dependencies_are_quotiented(tiny_constraint):
    matrix = sp.csr_matrix([[1, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, tiny_constraint]])
    selected, affected = _independent_exact_columns(matrix, np.arange(4), dependent_start=1)
    assert selected.tolist() == ([0, 1, 2] if tiny_constraint == 0 else [0, 1, 2, 3])
    assert affected == ({1, 2, 3} if tiny_constraint == 0 else set())


def test_a_rate_dependency_on_physical_state_is_not_disposable():
    selected, affected = _independent_exact_columns(sp.csr_matrix([[1, 1]]), np.arange(2), dependent_start=1)
    assert selected.tolist() == [0, 1]
    assert not affected


def test_structurally_deficient_disposable_rates_use_the_exact_dependency_retry():
    mass = sp.csr_matrix([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    pair = sparse_initial_pair(
        sp.eye(3, format='csr'),
        [-6.0, 0.0, 0.0],
        mass,
        np.zeros(3),
        algebraic_rows=sp.eye(3, format='csr')[1:],
        stored_rows=[0],
        partial_time=np.zeros(3),
        rate_jacobian=sp.csr_matrix((3, 3)),
    )
    np.testing.assert_array_equal(pair.state, np.zeros(3))
    np.testing.assert_allclose(mass @ pair.rate, [6.0, 0.0, 0.0], rtol=1e-14)
    assert not pair.rate_is_unique.any()


def test_exact_row_basis_preserves_mandatory_tiny_independent_equations():
    matrix = sp.csr_matrix([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1e-30]])
    rows = _independent_exact_rows(matrix, np.array([3]))
    assert rows.tolist() == [3, 0, 2]


def test_exact_row_retry_never_discards_a_dependent_mandatory_storage_row():
    with pytest.raises(ValueError, match='mandatory stored-coordinate'):
        _independent_exact_rows(sp.csr_matrix([[1, 0], [2, 0], [0, 1]]), np.array([0, 1]))


@pytest.mark.parametrize('contradiction', [0.0, 1e-3])
def test_a_structural_matching_must_not_replace_actual_initial_row_independence(contradiction):
    jacobian = [
        [4, 0, 0, 0, 0, 0, -2],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, -1, 0, -1, 0],
        [0, 0, -1, 1, 0, 0, 1],
        [0, -1, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 2],
    ]
    mass = sp.csr_matrix(([0.25, 1.0], ([5, 6], [5, 0])), shape=(7, 7))
    initial = np.array([0.15, 3.0, 2.3, 2.0, -0.3, 0.3, 0.3])

    def solve():
        return sparse_initial_pair(
            jacobian,
            [contradiction, 0, 0, 0, 0, -0.7, -1.4],
            mass,
            initial,
            algebraic_rows=sp.eye(7, format='csr')[:5],
            stored_rows=[5, 6],
            partial_time=np.zeros(7),
        )

    if contradiction:
        with pytest.raises(ValueError, match='contradicts'):
            solve()
        return
    pair = solve()
    np.testing.assert_allclose(pair.state, initial, atol=1e-14, rtol=0)
    assert pair.rate[0] == pytest.approx(1.4)
    assert pair.rate[5] == pytest.approx(2.8)
    np.testing.assert_array_equal(pair.rate_is_unique, [True, True, False, False, True, True, True])


@pytest.mark.parametrize(('stored_value', 'linkage_residual'), [(0.0, 0.0), (1.0, 1e-23), (1.0, 1e-3)])
def test_identical_current_rate_columns_do_not_make_the_physical_state_free(stored_value, linkage_residual):
    # Stored field and linkage fix the winding current through its dynamic
    # resistance equation. KCL differentiates only the sum of the two currents.
    pair = sparse_initial_pair(
        [[0, 0, -1, 0], [0, 1, 1, 0], [0, 0, 2, 0], [-1, 0, 0, 1]],
        [0, 0, 0, 0],
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
        [stored_value, 0, 0, stored_value],
        algebraic_rows=[[0, 1, 0, 0], [0, 0, 0, 1]],
        stored_rows=[0, 2],
        partial_time=[0, 0, -6, 0],
    )
    assert np.array_equal(pair.state, [stored_value, 0, 0, stored_value])
    assert np.array_equal(pair.rate_is_unique, [True, False, False, True])
    assert np.isnan(pair.physical_rate[1:3]).all()

    def complete():
        return sparse_complete_initial_rate(
            [[0, 0, -1, 0], [0, 1, 1, 0], [0, 0, 2, 0], [-1, 0, 0, 1]],
            [0, 0, 0, linkage_residual],
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
            pair,
            algebraic_rows=[[0, 1, 0, 0], [0, 0, 0, 1]],
            partial_time=[0, 0, -6, 0],
            acceleration_remainder=np.zeros(4),
            stored_rows=[0, 2],
            stored_columns=[0, 3],
        )

    if linkage_residual == 1e-3:
        with pytest.raises(ValueError, match='contradicts'):
            complete()
        return
    proven = complete()
    assert proven.rate_is_unique.all()
    assert proven.rate == pytest.approx([0, -2, 2, 0])
    assert np.array_equal(proven.state, pair.state)


def test_an_ideal_source_current_rate_is_not_a_physical_zero_derivative():
    # C v_dot + g v + i_source = 0; v = declared ramp(t).
    # Source current is determined, but its outgoing rate needs ramp''.
    pair = sparse_initial_pair(
        [[3.0, 1.0], [1.0, 0.0]],
        [6.0, 0.0],
        [[4.0, 0.0], [0.0, 0.0]],
        [2.0, 0.0],
        algebraic_rows=[[0.0, 1.0]],
        stored_rows=[0],
        partial_time=[0.0, -5.0],
    )
    np.testing.assert_allclose(pair.state, [2.0, -26.0])
    assert pair.rate[0] == pytest.approx(5.0)
    np.testing.assert_array_equal(pair.rate_is_unique, [True, False])


def test_declared_curvature_proves_source_current_rate_without_an_acceleration_guess():
    jacobian, mass, algebraic = [[3.0, 1.0], [1.0, 0.0]], [[4.0, 0.0], [0.0, 0.0]], [[0.0, 1.0]]
    pair = sparse_initial_pair(
        jacobian,
        [6.0, 0.0],
        mass,
        [2.0, 0.0],
        algebraic_rows=algebraic,
        stored_rows=[0],
        partial_time=[0.0, -5.0],
    )
    proven = sparse_complete_initial_rate(
        jacobian,
        [-20.0, 0.0],
        mass,
        pair,
        algebraic_rows=algebraic,
        partial_time=[0.0, -5.0],
        acceleration_remainder=[0.0, -2.0],
    )
    np.testing.assert_allclose(proven.state, pair.state)
    np.testing.assert_allclose(proven.rate, [5.0, -23.0])
    assert proven.rate_is_unique.all()


def test_sparse_initial_pair_never_densifies_a_large_differential_field(monkeypatch):
    def refuse_dense(*_args, **_kwargs):
        pytest.fail('field-sized dense initial decomposition is forbidden')

    monkeypatch.setattr(sp.csr_matrix, 'toarray', refuse_dense)
    monkeypatch.setattr(initial_rate, 'Fraction', refuse_dense)
    n = 17000
    pair = sparse_initial_pair(
        2 * sp.eye(n, format='csr'),
        np.full(n, 2.0),
        sp.eye(n, format='csr'),
        np.ones(n),
        algebraic_rows=sp.csr_matrix((0, n)),
        stored_rows=np.arange(n),
        partial_time=np.zeros(n),
    )
    np.testing.assert_allclose(pair.state, 1.0)
    np.testing.assert_allclose(pair.rate, -2.0)
    assert pair.rate_is_unique.all()
    proven = sparse_complete_initial_rate(
        2 * sp.eye(n, format='csr'),
        np.full(n, 2.0),
        sp.eye(n, format='csr'),
        pair,
        algebraic_rows=sp.csr_matrix((0, n)),
        partial_time=np.zeros(n),
        acceleration_remainder=np.zeros(n),
    )
    np.testing.assert_allclose(proven.rate, pair.rate)
    assert proven.rate_is_unique.all()


def test_sparse_projection_does_not_guess_an_undetermined_current_split():
    with pytest.raises(ValueError, match='physical authority|nonsingular'):
        sparse_initial_pair(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [0.0, 0.0, 0.0],
            np.diag([1.0, 0.0, 0.0]),
            np.zeros(3),
            algebraic_rows=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            stored_rows=[0],
            partial_time=np.zeros(3),
        )


def test_sparse_projection_refuses_an_actual_storage_source_contradiction():
    with pytest.raises(ValueError, match='contradicts'):
        sparse_initial_pair(
            [[3.0, 1.0], [1.0, 0.0]],
            [6.0, -1.0],
            [[4.0, 0.0], [0.0, 0.0]],
            [2.0, 0.0],
            algebraic_rows=[[0.0, 1.0]],
            stored_rows=[0],
            partial_time=[0.0, -5.0],
        )


def test_a_storage_border_eliminates_the_bulk_without_dropping_cross_rate_rows():
    mass = sp.csr_matrix([
        [2.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 3.0, 0.0, 0.0, 0.0],
        [4.0, 5.0, 2.0, 4.0, 0.0],
        [6.0, 7.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    rows = storage_border_rows(mass, bulk_indices=[0, 1], border_rows=[2, 3])
    assert len(rows.stored_rows) == 3
    assert rows.algebraic_rows.shape == (2, 5)
    np.testing.assert_allclose((rows.algebraic_rows @ mass).toarray(), 0.0, atol=1e-14)
    assert np.linalg.matrix_rank(mass[rows.stored_rows].toarray()) == 3


def test_a_large_storage_bulk_only_densifies_the_explicit_small_border(monkeypatch):
    original = sp.csr_matrix.toarray

    def forbid_square(self, *args, **kwargs):
        assert min(self.shape) <= 2, 'field-sized dense storage rank decomposition'
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, 'toarray', forbid_square)
    n = 17000
    mass = sp.bmat(
        [[sp.eye(n, format='csr'), None], [sp.csr_matrix(np.ones((2, n))), sp.eye(2, format='csr')]],
        format='csr',
    )
    rows = storage_border_rows(mass, bulk_indices=np.arange(n), border_rows=np.array([n, n + 1]))
    assert rows.algebraic_rows.shape == (0, n + 2)
    assert len(rows.stored_rows) == n + 2


def test_storage_border_never_silently_discards_an_undeclared_storage_row():
    with pytest.raises(ValueError, match='omits actual mass'):
        storage_border_rows(sp.eye(3, format='csr'), bulk_indices=[0], border_rows=[1])


@pytest.mark.parametrize('drive', [100.0, -100.0, 0.123, 1e-12])
def test_original_rate_validation_retries_exact_dependencies_lost_by_row_scaling(drive):
    # The last three columns have an exact disposable dependency. Scaling
    # rows by 3, 5, 7 and 12 can make the matched singular subsystem factor.
    matrix = sp.csr_matrix([[0, 1, 2, 3], [0, 2, 3, 5], [1, 3, 4, 7], [1, 5, 7, 12], [1, 0, 0, 0]], dtype=float)
    rhs = np.array([0, drive, 2 * drive, 3 * drive, 0])
    rate, unique = initial_rate.sparse_initial_rate(matrix, rhs)
    assert rate[0] == 0.0
    assert unique.tolist() == [True, False, False, False]
    # A zero row cannot cancel bit-exactly on every LAPACK build: its residual is judged at 64 ulps of the system scale.
    np.testing.assert_allclose(matrix @ rate, rhs, rtol=1e-10, atol=64 * np.finfo(float).eps * np.abs(rhs).max())


def test_exact_rate_retry_retains_tiny_original_constraint_contradictions():
    matrix = sp.csr_matrix(
        [[0, 1, 2, 3], [0, 2, 3, 5], [1, 3, 4, 7], [1, 5, 7, 12], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
    )
    with pytest.raises(ValueError, match='original physical constraints'):
        initial_rate.sparse_initial_rate(matrix, np.array([0, 100, 200, 300, 0, 1e-30]))


def test_rate_retry_keeps_a_genuinely_tiny_independent_original_constraint():
    matrix = sp.csr_matrix(
        [
            [0, 1, 2, 3],
            [0, 2, 3, 5],
            [1, 3, 4, 7],
            [1, 5, 7, 12],
            [1, 0, 0, 0],
            [0, 0, 0, 1e-30],
        ],
        dtype=float,
    )
    rate, unique = initial_rate.sparse_initial_rate(matrix, np.array([0, 100, 200, 300, 0, 0]))
    assert unique.all()
    assert rate[0] == rate[3] == 0.0


def test_rate_validation_retry_never_runs_general_exact_elimination_on_a_large_field(monkeypatch):
    original = initial_rate._independent_exact_columns

    def bounded(matrix, selected, *, dependent_start, general=True):
        assert not general or matrix.shape[1] <= 512
        return original(matrix, selected, dependent_start=dependent_start, general=general)

    monkeypatch.setattr(initial_rate, '_independent_exact_columns', bounded)
    matrix = sp.vstack((sp.eye(600, format='csr'), sp.eye(600, format='csr')[[0]]), format='csr')
    rhs = np.zeros(601)
    rhs[-1] = 1e-30
    with pytest.raises(ValueError, match='original physical constraints'):
        initial_rate.sparse_initial_rate(matrix, rhs)


@pytest.mark.parametrize('drive', [1.0, -1.0, 1e-12])
def test_original_homogeneous_rate_equations_survive_a_scaled_factor_roundoff_retry(drive):
    # Actual homogeneous constraints fix theta and both linkage rates to zero,
    # while finite field/current rates remain driven. No coordinate is snapped.
    matrix = sp.csr_matrix(
        [
            [-2, 4, -3, -4, 4, 5],
            [9, 0, 4, -8, 2, -1],
            [-2, 11, -9, 4, 0, -1],
            [1, 0, 0, 0, 1, 0],
            [5, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0.5, 1],
            [0, 1, 1, 0, -1, 0],
        ],
        dtype=float,
    )
    rhs = drive * np.array([-5, -28, 32, 0, 0, 0, 0])
    rate, unique = initial_rate.sparse_initial_rate(matrix, rhs)
    assert unique.all()
    assert rate[0] == rate[4] == rate[5] == 0.0
    np.testing.assert_allclose(rate, drive * np.array([0, 1, -1, 3, 0, 0]), rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(matrix @ rate, rhs, rtol=1e-10, atol=0.0)
