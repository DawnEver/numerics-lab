"""Finite-Q first pairs use the actual-rate derivative, including a changing nullspace."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import initial_rate as sparse_module
from numerics_lab.linear.initial_rate import sparse_initial_rate
from numerics_lab.linear.initial_sparse import SparseInitialPair
from numerics_lab.linear.storage_border import StorageRows, storage_border_columns, storage_border_rows
from numerics_lab.nonlinear import finite_storage as finite_module
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('contradiction', [0.0, 1e-3])
def test_state_particular_keeps_original_physical_row_accuracy_before_issuing_boundary_freedom(contradiction):
    """Dependent physical rows may differ by their actually performed cancellation.

    Q fixes nonzero z/y; x has two algebraically identical physical equations,
    while q must change and the last boundary state remains genuinely free.
    Solving a correction must not demand 1e-10 of an already tiny RHS instead
    of the unchanged original equations' own physical accuracy.
    """
    initial = np.array([0.6, 6.8e-6, 0.6 + 6.8e-6, 0.0, 0.0])
    mass = sp.diags([1.0, 1.0, 0.0, 0.0, 0.0], format='csr')
    jacobian = sp.csr_matrix([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, -1, 1, 0, 0], [-1, -1, 1, 0, 0], [0, 0, 0, 1, 0]])
    storage = StorageRows(np.array([0, 1]), sp.eye(5, format='csr')[[2, 3, 4]])

    def residual(state, rate):
        z, y, x, q, _free_boundary = state
        return np.array([rate[0], rate[1], (x - z) - y, x - (z + y) + contradiction, q - 1.0])

    assert residual(initial, np.zeros(5))[2] != 0.0, 'the performed cancellation is nonvacuous'
    arguments = (
        initial,
        mass @ initial,
        residual,
        lambda _state, _rate: jacobian,
        lambda _state, _rate: mass,
        lambda state: mass @ state,
        lambda _state, _rate: storage,
        8,
        1e-10,
        lambda _state, _rate, _storage: np.array([0, 1]),
        np.array([0, 1, 2]),
    )
    if contradiction:
        with pytest.raises(ValueError, match='contradict'):
            finite_module._actual_finite_storage_state(*arguments)
        return
    with pytest.raises(finite_module.StorageRateAuthorityRequired) as caught:
        finite_module._actual_finite_storage_state(*arguments)
    state, directions = caught.value.state, caught.value.directions.toarray()
    np.testing.assert_array_equal(mass @ state, mass @ initial)
    assert state[3] == 1.0
    scale = abs(jacobian) @ abs(state) + abs(residual(state, np.zeros(5)))
    assert np.all(abs(residual(state, np.zeros(5))) <= 1e-10 * scale)
    assert np.count_nonzero(directions[:4]) == 0
    assert directions[4, 0] != 0.0


@pytest.mark.parametrize('coupling', [1.0, 3.0])
def test_surviving_boundary_state_freedom_keeps_the_actual_mixed_time_state_column(coupling):
    """F=(x'-y, x+t*c*y-1-2*t) at t=0 has Q=x and y=2/(1+c).

    The differentiated algebraic row is x'+c*y-2, not x'-2.
    Dropping its state column makes the admission alternate or diverge despite
    an exactly representable original state/rate pair. No tolerance is at issue.
    """
    mass = sp.diags([1.0, 0.0], format='csr')
    jacobian = sp.csr_matrix([[0.0, -1.0], [1.0, 0.0]])
    pair = finite_storage_actual_rate_initial_pair(
        np.array([1.0, 0.0]),
        residual_at=lambda state, rate: np.array([rate[0] - state[1], state[0] - 1.0]),
        jacobian_at=lambda _state, _rate: jacobian,
        mass_at=lambda _state, _rate: mass,
        storage_at=lambda state: mass @ state,
        partial_time_at=lambda state, _rate: np.array([0.0, coupling * state[1] - 2.0]),
        storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]])),
        rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
        rate_constraint_state_columns_at=lambda _state, _rate, directions: (
            sp.csr_matrix([[0.0, 0.0], [0.0, coupling]]) @ directions
        ),
        stored_columns_at=lambda _state, _rate, _storage: np.array([0]),
        sparse_state_indices=np.array([0]),
        max_newton=8,
    )
    expected = 2.0 / (1.0 + coupling)
    np.testing.assert_array_equal(pair.state, [1.0, expected])
    assert pair.rate[0] == expected
    assert pair.rate_is_unique[0]
    assert not pair.rate_is_unique[1], 'no second differentiated row owns the algebraic rate'
    assert pair.rate[0] + coupling * pair.state[1] - 2.0 == 0.0


def test_inconsistent_original_mixed_boundary_equations_are_still_refused():
    mass = sp.diags([1.0, 0.0], format='csr')
    jacobian = sp.csr_matrix([[0.0, -1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match='contradict|inconsistent|did not converge|physical state authority'):
        finite_storage_actual_rate_initial_pair(
            np.array([1.0, 0.0]),
            residual_at=lambda state, rate: np.array([rate[0] - state[1], state[0] - 1.0]),
            jacobian_at=lambda _state, _rate: jacobian,
            mass_at=lambda _state, _rate: mass,
            storage_at=lambda state: mass @ state,
            partial_time_at=lambda state, _rate: np.array([0.0, -state[1] - 2.0]),
            storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]])),
            rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
            rate_constraint_state_columns_at=lambda _state, _rate, directions: (
                sp.csr_matrix([[0.0, 0.0], [0.0, -1.0]]) @ directions
            ),
            stored_columns_at=lambda _state, _rate, _storage: np.array([0]),
            sparse_state_indices=np.array([0]),
            max_newton=8,
        )


def test_mixed_callback_without_an_issued_bounded_state_range_is_not_silently_ignored():
    mass = sp.diags([1.0, 0.0], format='csr')
    jacobian = sp.csr_matrix([[0.0, -1.0], [1.0, 0.0]])
    with pytest.raises(NotImplementedError, match='mixed|state.range|bounded'):
        finite_storage_actual_rate_initial_pair(
            np.array([1.0, 0.0]),
            residual_at=lambda state, rate: np.array([rate[0] - state[1], state[0] - 1.0]),
            jacobian_at=lambda _state, _rate: jacobian,
            mass_at=lambda _state, _rate: mass,
            storage_at=lambda state: mass @ state,
            partial_time_at=lambda state, _rate: np.array([0.0, state[1] - 2.0]),
            storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]])),
            rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
            rate_constraint_state_columns_at=lambda _state, _rate, directions: (
                sp.csr_matrix([[0.0, 0.0], [0.0, 1.0]]) @ directions
            ),
            max_newton=8,
        )


@pytest.mark.parametrize('supplied', [np.zeros((2, 2)), np.array([[np.nan], [0.0]]), np.zeros((2, 1), dtype=bool)])
def test_mixed_boundary_state_callback_cannot_invent_finite_real_directional_support(supplied):
    mass = sp.diags([1.0, 0.0], format='csr')
    jacobian = sp.csr_matrix([[0.0, -1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match='mixed boundary state columns'):
        finite_storage_actual_rate_initial_pair(
            np.array([1.0, 0.0]),
            residual_at=lambda state, rate: np.array([rate[0] - state[1], state[0] - 1.0]),
            jacobian_at=lambda _state, _rate: jacobian,
            mass_at=lambda _state, _rate: mass,
            storage_at=lambda state: mass @ state,
            partial_time_at=lambda state, _rate: np.array([0.0, state[1] - 2.0]),
            storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]])),
            rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
            rate_constraint_state_columns_at=lambda _state, _rate, _directions: supplied,
            stored_columns_at=lambda _state, _rate, _storage: np.array([0]),
            sparse_state_indices=np.array([0]),
            max_newton=8,
        )


@pytest.mark.parametrize('nonlinear', [False, True])
def test_original_nonzero_algebraic_particular_is_solved_before_issuing_surviving_boundary_range(nonlinear):
    """Q fixes x, the field equation fixes z=1, and time derivatives select y.

    At the zero seed z is NOT admitted. Its independent field correction must
    not be confused with the surviving current direction (only y), or with a
    missing state-range authority. This is the actual signed-PM initial lane.
    """
    mass = sp.diags([1.0, 0.0, 0.0], format='csr')

    def jacobian(state, _rate):
        return sp.csr_matrix([[0.0, 0.0, -1.0], [0.0, 1 + 3 * nonlinear * state[1] ** 2, 0.0], [1.0, 0.0, 0.0]])

    pair = finite_storage_actual_rate_initial_pair(
        np.zeros(3),
        residual_at=lambda state, rate: np.array([
            rate[0] - state[2],
            state[1] + nonlinear * state[1] ** 3 - (1 + nonlinear),
            state[0],
        ]),
        jacobian_at=jacobian,
        mass_at=lambda _state, _rate: mass,
        storage_at=lambda state: mass @ state,
        partial_time_at=lambda state, _rate: np.array([0.0, 0.0, state[2] - 2.0]),
        storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.eye(3, format='csr')[[1, 2]]),
        rate_constraint_jacobian_at=jacobian,
        rate_constraint_state_columns_at=lambda state, rate, directions: (
            sp.diags([0.0, 6 * nonlinear * state[1] * rate[1], 1.0], format='csr') @ directions
        ),
        stored_columns_at=lambda _state, _rate, _storage: np.array([0]),
        sparse_state_indices=np.array([0, 1]),
        max_newton=12,
    )
    np.testing.assert_allclose(pair.state, [0.0, 1.0, 1.0], rtol=1e-10, atol=0)
    assert pair.rate[0] == 1.0 and pair.rate[1] == 0.0
    np.testing.assert_array_equal(mass @ pair.state, np.zeros(3))
    assert not pair.rate_is_unique[2]


def test_final_rank_proof_does_not_readmit_mutual_zero_charge_under_nonzero_rate_forcing():
    stiffness = sp.diags((-np.ones(3), 2 * np.ones(4), -np.ones(3)), (-1, 0, 1), format='csr')
    incidence = sp.csr_matrix([[1.0, 0], [1.0, 1.0], [0, 1.0], [0, 0]])
    jacobian = sp.bmat(
        [
            [stiffness, -incidence, None],
            [None, sp.diags([4.0, 2.0]), sp.csr_matrix((2, 2))],
            [-incidence.T, None, sp.eye(2)],
        ],
        format='csr',
    )
    mass = sp.csr_matrix((np.ones(2), ([4, 5], [6, 7])), shape=(8, 8))
    forcing = np.array([0.0, 0, 0, 0, 24.0, 0, 0, 0])
    storage = StorageRows(np.array([4, 5]), sp.eye(8, format='csr')[[0, 1, 2, 3, 6, 7]])
    pair = finite_storage_actual_rate_initial_pair(
        np.zeros(8),
        residual_at=lambda state, rate: jacobian @ state + mass @ rate - forcing,
        jacobian_at=lambda _state, _rate: jacobian,
        mass_at=lambda _state, _rate: mass,
        partial_time_at=lambda _state, _rate: np.zeros(8),
        storage_at=lambda state: mass @ state,
        storage_rows_at=lambda _state, _rate: storage,
        stored_columns_at=lambda _state, _rate, _storage: np.array([6, 7]),
        rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
    )
    np.testing.assert_array_equal(pair.state, np.zeros(8))
    np.testing.assert_array_equal(mass @ pair.state, np.zeros(8))
    np.testing.assert_allclose(mass @ pair.rate, forcing, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(storage.algebraic_rows @ jacobian @ pair.rate, 0.0, atol=1e-13)
    assert pair.rate_is_unique.all()


@pytest.mark.parametrize('mass', [np.array([[1e308]]), np.array([[1.0, 0], [1.0, 1e-30]])])
def test_actual_border_preserves_every_finite_independent_storage_row(mass):
    result = storage_border_rows(
        sp.csr_matrix(mass), bulk_indices=np.array([], dtype=int), border_rows=np.arange(len(mass))
    )
    np.testing.assert_array_equal(result.stored_rows, np.arange(len(mass)))
    assert result.algebraic_rows.shape == (0, len(mass))


def test_storage_border_densifies_only_actual_nonzero_border_support(monkeypatch):
    original = sp.csr_matrix.toarray

    def bounded(matrix, *args, **kwargs):
        assert min(matrix.shape) == 0 or max(matrix.shape) <= 2
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, 'toarray', bounded)
    mass = sp.csr_matrix(([2.0, 3.0, 0.0], ([0, 1, 1], [0, 16999, 100])), shape=(17000, 17000))
    rows = storage_border_rows(mass, bulk_indices=np.array([0]), border_rows=np.array([1]))
    columns = storage_border_columns(mass, bulk_indices=np.array([0]), stored_rows=rows.stored_rows)
    np.testing.assert_array_equal(columns, [0, 16999])


@pytest.mark.parametrize(('bulk', 'rows'), [([0.0], [0]), ([0], [0, 0]), ([1], [0]), ([-1], [0]), ([0], [2])])
def test_storage_column_authority_refuses_invalid_indices(bulk, rows):
    with pytest.raises(ValueError, match='indices'):
        storage_border_columns(sp.eye(2), bulk_indices=np.asarray(bulk), stored_rows=np.asarray(rows))


def test_storage_column_authority_refuses_nonfinite_mass_and_handles_empty_rank():
    with pytest.raises(ValueError, match='finite'):
        storage_border_columns(sp.csr_matrix([[np.nan]]), bulk_indices=np.array([0]), stored_rows=np.array([0]))
    np.testing.assert_array_equal(
        storage_border_columns(
            sp.csr_matrix((2, 2)), bulk_indices=np.array([], dtype=int), stored_rows=np.array([], dtype=int)
        ),
        np.array([], dtype=int),
    )


def test_frozen_state_rate_equations_preserve_exact_small_constraints_and_free_rate_support():
    rate, unique = sparse_initial_rate(
        sp.csr_matrix([[1.0, 0, 0], [0, 1e-30, 1e-30]]),
        np.array([2.0, 3e-30]),
    )
    np.testing.assert_allclose(rate[0], 2.0)
    np.testing.assert_allclose(rate[1] + rate[2], 3.0)
    np.testing.assert_array_equal(unique, [True, False, False])
    with pytest.raises(ValueError, match='contradict'):
        sparse_initial_rate(sp.csr_matrix([[1.0], [1.0]]), np.array([0.0, 1e-30]))


def test_actual_rate_tangent_sparse_storage_elimination_keeps_zero_charge_coordinates_exact():
    pair = finite_module.sparse_initial_pair(
        sp.csr_matrix([[3.0, -1.0, 0], [0, 5.0, 0], [-1.0, 0, 1.0]]),
        np.array([0.0, -7.0, 0.0]),
        sp.csr_matrix([[2.0, 0, 0], [0, 0, 1.0], [0, 0, 0]]),
        np.zeros(3),
        algebraic_rows=sp.csr_matrix([[0.0, 0, 1.0]]),
        stored_rows=np.array([0, 1]),
        stored_columns=np.array([0, 2]),
        partial_time=np.zeros(3),
        rate_jacobian=sp.csr_matrix([[3.0, -1.0, 0], [0, 5.0, 0], [-1.0, 0, 1.0]]),
    )
    np.testing.assert_array_equal(pair.state[[0, 2]], [0.0, 0.0])
    assert pair.state[1] == pytest.approx(7 / 5.5)


def test_source_clamped_capacitor_state_uses_actual_source_rate_without_inventing_current_rate():
    # Q=(v,0), F=(v_dot-i,v-t): source V=t fixes i=1 only
    # through the first differentiated source equation. Its i_dot is not
    # supplied by these first-pair equations and must remain unavailable.
    mass = sp.csr_matrix([[1.0, 0], [0, 0]])
    jacobian = sp.csr_matrix([[0.0, -1.0], [1.0, 0]])
    pair = finite_storage_actual_rate_initial_pair(
        np.zeros(2),
        residual_at=lambda state, rate: np.array([rate[0] - state[1], state[0]]),
        jacobian_at=lambda _state, _rate: jacobian,
        mass_at=lambda _state, _rate: mass,
        storage_at=lambda state: np.array([state[0], 0.0]),
        partial_time_at=lambda _state, _rate: np.array([0.0, -1.0]),
        storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]])),
        rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
        stored_columns_at=lambda _state, _rate, _storage: np.array([0]),
    )
    np.testing.assert_array_equal(pair.state, [0.0, 1.0])
    np.testing.assert_array_equal(pair.rate, [1.0, 0.0])
    np.testing.assert_array_equal(pair.rate_is_unique, [True, False])


def test_normal_field_sized_sparse_rate_path_does_not_rationalize_or_densify(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('ordinary sparse rate admission must not use dense or rational elimination')

    monkeypatch.setattr(sparse_module, 'Fraction', forbidden)
    monkeypatch.setattr(sp.csr_matrix, 'toarray', forbidden)
    rate, unique = sparse_initial_rate(sp.eye(17000, format='csr'), np.ones(17000))
    np.testing.assert_array_equal(rate, np.ones(17000))
    assert unique.all()


@pytest.mark.parametrize('options', [{'tolerance': np.nan}, {'tolerance': np.inf}, {'max_newton': 1.5}])
def test_nonfinite_or_nonintegral_iteration_controls_are_refused_before_evaluation(options):
    def forbidden(*_args):
        pytest.fail('invalid controls must be rejected before physical evaluation')

    with pytest.raises(ValueError, match='iteration controls'):
        finite_storage_actual_rate_initial_pair(
            np.zeros(1),
            residual_at=forbidden,
            jacobian_at=forbidden,
            mass_at=forbidden,
            storage_at=forbidden,
            partial_time_at=forbidden,
            storage_rows_at=forbidden,
            rate_constraint_jacobian_at=forbidden,
            **options,
        )


@pytest.mark.parametrize('charge', [np.array([np.nan]), np.zeros(2), np.array([[0.0]])])
def test_invalid_full_charge_witness_is_refused(charge):
    with pytest.raises(ValueError, match='finite.*charge.*shape'):
        finite_storage_actual_rate_initial_pair(
            np.zeros(1),
            residual_at=lambda _u, _v: np.zeros(1),
            jacobian_at=lambda _u, _v: sp.eye(1),
            mass_at=lambda _u, _v: sp.eye(1),
            storage_at=lambda _u: charge,
            partial_time_at=lambda _u, _v: np.zeros(1),
            storage_rows_at=lambda _u, _v: StorageRows(np.array([0]), sp.csr_matrix((0, 1))),
            rate_constraint_jacobian_at=lambda _u, _v: sp.eye(1),
        )


@pytest.mark.parametrize(
    ('owner', 'bad_value'),
    [
        ('mass_at', sp.csr_matrix([[np.nan]])),
        ('jacobian_at', sp.eye(2)),
        ('residual_at', np.zeros(2)),
        ('partial_time_at', np.array([np.inf])),
        ('rate_constraint_jacobian_at', sp.eye(2)),
    ],
)
def test_invalid_actual_equation_owner_shapes_and_nonfinite_values_are_refused(owner, bad_value):
    callbacks = {
        'residual_at': lambda _u, _v: np.zeros(1),
        'jacobian_at': lambda _u, _v: sp.eye(1),
        'mass_at': lambda _u, _v: sp.eye(1),
        'storage_at': lambda u: u.copy(),
        'partial_time_at': lambda _u, _v: np.zeros(1),
        'storage_rows_at': lambda _u, _v: StorageRows(np.array([0]), sp.csr_matrix((0, 1))),
        'rate_constraint_jacobian_at': lambda _u, _v: sp.eye(1),
    }
    callbacks[owner] = lambda _u, _v: bad_value
    with pytest.raises(ValueError, match='finite actual.*shape'):
        finite_storage_actual_rate_initial_pair(np.zeros(1), **callbacks)


def test_rate_uniqueness_is_proved_at_the_admitted_state_not_the_search_seed():
    # F=x-1+t*(x-1)*v at t=0; Q=t*(x-1)^2/2. The differentiated
    # equation is x*v=0: its rate rank changes from zero to one on admission.
    pair = finite_storage_actual_rate_initial_pair(
        np.array([0.0]),
        residual_at=lambda state, _rate: state - 1,
        jacobian_at=lambda _state, _rate: sp.eye(1, format='csr'),
        mass_at=lambda _state, _rate: sp.csr_matrix((1, 1)),
        storage_at=lambda _state: np.zeros(1),
        partial_time_at=lambda state, rate: (state - 1) * rate,
        storage_rows_at=lambda _state, _rate: StorageRows(np.array([], dtype=int), sp.eye(1, format='csr')),
        rate_constraint_jacobian_at=lambda state, _rate: sp.diags(state),
    )
    np.testing.assert_array_equal(pair.state, [1.0])
    np.testing.assert_array_equal(pair.rate, [0.0])
    assert pair.rate_is_unique.all()


def test_rate_equations_keep_explicit_admitted_state_allowance_row_local():
    matrix = sp.csr_matrix([[1.0], [0.0], [1.0]])
    rate, unique = sparse_initial_rate(
        matrix, np.array([2.0, 5e-17, 2.0]), residual_allowance=np.array([0.0, 1e-16, 0.0])
    )
    np.testing.assert_array_equal(rate, [2.0])
    assert unique.all()
    with pytest.raises(ValueError, match='contradict'):
        sparse_initial_rate(matrix, np.array([2.0, 1e-12, 2.0]), residual_allowance=np.array([0.0, 1e-16, 0.0]))


def test_finite_rate_equations_refuse_an_unrepresentable_solution():
    with pytest.raises(ValueError, match='finite'):
        sparse_module.sparse_initial_rate(sp.csr_matrix([[1e-308]]), np.array([2.0]))


def test_changing_storage_nullspace_retains_the_actual_curvature_in_the_rate_equation():
    def residual(state, rate):
        x, y = state
        return np.array([3 * x + rate[0] - y, y + 2 * x * rate[0] - 0.48])

    def jacobian(_state, rate):
        return sp.csr_matrix([[3, -1], [2 * rate[0], 1]])

    def mass(state, _rate):
        return sp.csr_matrix([[1, 0], [2 * state[0], 0]])

    def rows(state, rate):
        return storage_border_rows(mass(state, rate), bulk_indices=np.array([0]), border_rows=np.array([1]))

    pair = finite_storage_actual_rate_initial_pair(
        np.array([0.4, 0.0]),
        residual_at=residual,
        jacobian_at=jacobian,
        mass_at=mass,
        storage_at=lambda state: np.array([state[0], state[0] ** 2]),
        partial_time_at=lambda _state, _rate: np.zeros(2),
        storage_rows_at=rows,
        rate_constraint_jacobian_at=lambda _state, rate: sp.csr_matrix([[3, -1], [4 * rate[0], 1]]),
    )
    np.testing.assert_allclose(pair.state, [0.4, 0.8], atol=1e-13)
    np.testing.assert_allclose(pair.rate, [-0.4, -1.28 / 1.8], atol=1e-13)
    assert pair.rate_is_unique.all()
    np.testing.assert_allclose(residual(pair.state, pair.rate), 0, atol=1e-13)
    np.testing.assert_allclose(
        rows(pair.state, pair.rate).algebraic_rows @ jacobian(pair.state, pair.rate) @ pair.rate, 0, atol=1e-13
    )


def test_all_finite_charge_witnesses_are_preserved_not_only_the_selected_square(monkeypatch):
    # Stored row1=x² alone admits x=-1 while the original Q0=x explicitly forbids it.
    wrong_branch = SparseInitialPair(np.array([-1.0, 0.0]), np.zeros(2), np.ones(2, dtype=bool))
    # A planted numerical candidate satisfies F and the selected square charge;
    # physical admission must independently reject the other ORIGINAL witness.
    monkeypatch.setattr(finite_module, '_actual_finite_storage_state', lambda *_args, **_kwargs: wrong_branch.state)
    with pytest.raises(ValueError, match='finite.*storage'):
        finite_storage_actual_rate_initial_pair(
            np.array([1.0, 0.0]),
            residual_at=lambda state, rate: np.array([state[0] + 1 + rate[0], 2 * state[0] * rate[0] + state[1]]),
            jacobian_at=lambda _state, rate: sp.csr_matrix([[1, 0], [2 * rate[0], 1]]),
            mass_at=lambda state, _rate: sp.csr_matrix([[1, 0], [2 * state[0], 0]]),
            storage_at=lambda state: np.array([state[0], state[0] ** 2]),
            partial_time_at=lambda _state, _rate: np.zeros(2),
            storage_rows_at=lambda state, _rate: StorageRows(np.array([1]), sp.csr_matrix([[-2 * state[0], 1.0]])),
            rate_constraint_jacobian_at=lambda _state, rate: sp.csr_matrix([[1, 0], [4 * rate[0], 1]]),
        )


def _rl_pair(seed, *, source, visited=None):
    """The RL loop psi' + R i = v, psi = L i: one stored linkage, one algebraic current (L/R = 3e-15 s)."""
    visited = [] if visited is None else visited
    resistance, inductance = 3.0, 1.0e-14
    jacobian = sp.csr_matrix([[0.0, resistance], [1.0, -inductance]])
    mass = sp.csr_matrix([[1.0, 0.0], [0.0, 0.0]])
    forcing = np.array([source, 0.0])
    return finite_storage_actual_rate_initial_pair(
        np.asarray(seed, dtype=float),
        residual_at=lambda state, rate: visited.append(np.array(state)) or jacobian @ state + mass @ rate - forcing,
        jacobian_at=lambda _state, _rate: jacobian,
        mass_at=lambda _state, _rate: mass,
        partial_time_at=lambda _state, _rate: np.zeros(2),
        storage_at=lambda state: mass @ state,
        storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.eye(2, format='csr')[[1]]),
        rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
        target_storage=np.zeros(2),
    )


def test_a_homogeneous_system_with_zero_storage_is_admitted_at_exact_rest():
    # The left limit of a pulse that ended: 1.6e-11 A on a source now exactly zero. The answer is
    # the exact zero, never a walk into gradual underflow (2.8e-309 A on the moving band, 2026-10-14).
    visited = []
    pair = _rl_pair([0.0, 1.6e-11], source=0.0, visited=visited)
    # The search STARTS at rest: the left-limit seed is never an iterate.
    assert all(not state.any() for state in visited)
    np.testing.assert_array_equal(pair.state, np.zeros(2))
    np.testing.assert_array_equal(pair.rate, np.zeros(2))
    assert pair.rate_is_unique.all()


def test_PLANTED_CONTROL_a_forced_system_is_not_taken_for_rest():
    visited = []
    pair = _rl_pair([0.0, 1.6e-11], source=1.0, visited=visited)
    assert any(state.any() for state in visited), 'a forced system must search from its own seed'
    np.testing.assert_array_equal(pair.state, np.zeros(2))
    np.testing.assert_allclose(pair.rate[0], 1.0, rtol=1e-12)
