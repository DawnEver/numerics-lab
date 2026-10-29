"""Owned relative-rate coordinates must not inherit a rounded raw matrix rank."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.formulation import InitialJointEvaluation, ResidualAssemblyBound
from numerics_lab.linear import initial_rate
from numerics_lab.linear.storage_border import StorageRows
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(('defect', 'known'), [(2.0**-52, True), (2.0**-52, False), (1e-3, True)])
def test_joint_chart_and_original_accelerated_derivative_keep_separate_owned_bounds(defect, known):
    options, owner = _problem()
    original = options['differentiated_residual_at']
    calls = []

    def chart(state, rate):
        return replace(
            owner(state, rate),
            differentiated_bound=ResidualAssemblyBound(np.zeros(4), np.ones(4, dtype=bool)),
        )

    def physical_evaluation(state, rate, acceleration):
        calls.append((state.copy(), rate.copy(), acceleration.copy()))
        value = original(state, rate, acceleration)
        value[1] += defect
        values = np.zeros(4)
        values[1] = 1e-15
        return SimpleNamespace(residual=value, bound=ResidualAssemblyBound(values, np.full(4, known)))

    def must_not_duplicate(*_args):
        pytest.fail('physical differentiation evaluation owns its value and actual acceleration together')

    options['differentiated_residual_at'] = must_not_duplicate
    options['differentiated_residual_evaluation_at'] = physical_evaluation
    if known and defect < 1e-15:
        pair = finite_storage_actual_rate_initial_pair(np.zeros(4), joint_coordinates_at=chart, **options)
        np.testing.assert_array_equal(pair.state, [0, 1, 0, 0])
        np.testing.assert_array_equal(pair.rate, [1, 0, 0, 1])
        assert calls
    else:
        with pytest.raises(ValueError, match='converge|contradict|inconsistent'):
            finite_storage_actual_rate_initial_pair(np.zeros(4), joint_coordinates_at=chart, **options)


@pytest.mark.parametrize('invalid', ['shape', 'missing_value'])
def test_joint_differentiated_bound_requires_the_same_aligned_performed_value(invalid):
    _options, owner = _problem()
    sample = owner(np.zeros(4), np.zeros(4))
    bound = ResidualAssemblyBound(
        np.zeros(3 if invalid == 'shape' else 4), np.ones(3 if invalid == 'shape' else 4, bool)
    )
    with pytest.raises(ValueError, match='differentiat|align|shape|value'):
        replace(
            sample,
            differentiated_value=None if invalid == 'missing_value' else sample.differentiated_value,
            differentiated_bound=bound,
        )


@pytest.mark.parametrize(
    'name',
    [
        'residual_state_columns',
        'residual_rate_columns',
        'differentiated_state_columns',
        'differentiated_rate_columns',
        'physical_rate_state_map',
        'physical_rate_rate_map',
    ],
)
@pytest.mark.parametrize('sparse', [True, False])
def test_joint_columns_cannot_discard_complex_owned_authority(name, sparse):
    options, owner = _problem()

    def chart(state, rate):
        sample = owner(state, rate)
        matrix = getattr(sample, name).astype(complex) + sp.csr_matrix(([1j], ([0], [0])), shape=(4, 4))
        return replace(sample, **{name: matrix if sparse else matrix.toarray()})

    with pytest.raises(ValueError, match='real'):
        finite_storage_actual_rate_initial_pair(np.zeros(4), joint_coordinates_at=chart, **options)


def _problem(*, malicious=False, missing=False, tiny=False, raw=False):
    m, w = 3.0, 2.0
    mass = sp.csr_matrix([[m, 0, 0, 0], [-w, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    physical_j = sp.csr_matrix([[0, -m, -1, 0], [0, w, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
    raw_j = physical_j.copy()
    raw_j[0, 1] = -np.nextafter(m, np.inf)
    algebraic = sp.csr_matrix([[w, m, 0, 0], [0, 0, 0, 1]])

    def residual(u, v):
        relative = v[0] - u[1]
        return np.array([m * relative - u[2], -w * relative, v[2] + v[3] - 1, u[3] - u[0] + (1e-30 if tiny else 0)])

    def chart(u, r):
        v = r.copy()
        v[0] += u[1]
        state_map = sp.csr_matrix(([1.0], ([0], [1])), shape=(4, 4))
        joint_j = sp.csr_matrix([[0, 0, -1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]])
        g_state = sp.csr_matrix(([-1.0], ([3], [1])), shape=(4, 4))
        if malicious:
            v[3] = 0.0
        return InitialJointEvaluation(
            v,
            joint_j,
            mass,
            None if missing else g_state,
            joint_j,
            state_map,
            sp.eye(4, format='csr'),
            physical_j @ v + mass @ (state_map @ v),
            state_map @ v,
        )

    options = {
        'residual_at': residual,
        'jacobian_at': lambda _u, _v: raw_j,
        'mass_at': lambda _u, _v: mass,
        'storage_at': lambda u: mass @ u,
        'partial_time_at': lambda _u, _v: np.zeros(4),
        'storage_rows_at': lambda _u, _v: StorageRows(np.array([0, 2]), algebraic),
        'stored_columns_at': lambda _u, _v, _s: np.array([0, 3]),
        'rate_constraint_jacobian_at': lambda _u, _v: physical_j,
        'differentiated_residual_at': lambda _u, v, a: physical_j @ v + mass @ a,
    }
    # Physical jets are separately owned, not reconstructed from the bad raw joint rank.
    options['jacobian_at'] = lambda _u, _v: raw_j if raw else physical_j
    return options, chart


def test_actual_signed_relative_graph_admits_source_rate_and_transports_physical_uniqueness():
    options, chart = _problem()
    pair = finite_storage_actual_rate_initial_pair(np.zeros(4), joint_coordinates_at=chart, **options)
    np.testing.assert_array_equal(pair.state, [0, 1, 0, 0])
    np.testing.assert_array_equal(pair.rate, [1, 0, 0, 1])
    np.testing.assert_array_equal(pair.rate_is_unique, [True, False, True, True])
    np.testing.assert_array_equal(options['residual_at'](pair.state, pair.rate), np.zeros(4))
    np.testing.assert_array_equal(options['storage_at'](pair.state), np.zeros(4))


def test_joint_acceptance_does_not_resample_a_future_source_for_its_reference_scale():
    options, chart = _problem()
    original = options['residual_at']
    calls = []

    def residual(state, rate):
        calls.append((state.copy(), rate.copy()))
        if len(calls) > 2:
            pytest.fail('acceptance resampled the source instead of its actual residual snapshot')
        return original(state, rate)

    options['residual_at'] = residual
    pair = finite_storage_actual_rate_initial_pair(np.zeros(4), joint_coordinates_at=chart, **options)
    assert len(calls) == 2
    np.testing.assert_array_equal(pair.rate, [1, 0, 0, 1])


def test_one_ulp_raw_matrix_rank_is_not_the_original_relative_graph_authority():
    options, chart = _problem(raw=True)
    with pytest.raises(ValueError, match='initial|constraints|equations'):
        finite_storage_actual_rate_initial_pair(np.zeros(4), **options)
    pair = finite_storage_actual_rate_initial_pair(np.zeros(4), joint_coordinates_at=chart, **options)
    np.testing.assert_array_equal(pair.state, [0, 1, 0, 0])
    np.testing.assert_array_equal(pair.rate, [1, 0, 0, 1])


def test_nonidentity_independent_rate_map_transports_unknown_rates_to_physical_rows():
    options, owner = _problem()
    permutation = sp.eye(4, format='csr')[[0, 3, 2, 1]]

    def chart(state, independent):
        sample = owner(state, permutation @ independent)
        return InitialJointEvaluation(
            sample.physical_rate,
            sample.residual_state_columns,
            sample.residual_rate_columns @ permutation,
            sample.differentiated_state_columns,
            sample.differentiated_rate_columns @ permutation,
            sample.physical_rate_state_map,
            permutation,
            sample.differentiated_value,
            sample.physical_acceleration_offset,
        )

    pair = finite_storage_actual_rate_initial_pair(np.zeros(4), joint_coordinates_at=chart, **options)
    np.testing.assert_array_equal(pair.rate, [1, 0, 0, 1])
    np.testing.assert_array_equal(pair.rate_is_unique, [True, False, True, True])


@pytest.mark.parametrize('bad', [np.nan, np.ones((2, 2)), np.array([1.0 + 2.0j])])
def test_invalid_joint_physical_rate_is_refused(bad):
    with pytest.raises(ValueError, match='finite physical rate vector'):
        InitialJointEvaluation(bad, None, None, None, None, None, None)


def test_joint_physical_rate_vector_is_copied_and_readonly():
    original = np.array([1.0])
    sample = InitialJointEvaluation(original, None, None, None, None, None, None)
    original[0] = 2.0
    np.testing.assert_array_equal(sample.physical_rate, [1.0])
    with pytest.raises(ValueError, match='read-only'):
        sample.physical_rate[0] = 3.0


@pytest.mark.parametrize('name', ['differentiated_value', 'physical_acceleration_offset'])
@pytest.mark.parametrize('bad', [[True], [1.0 + 1j], [np.inf], [1.0, 2.0]])
def test_joint_lifted_vectors_reject_nonphysical_values(name, bad):
    with pytest.raises(ValueError, match='physical'):
        InitialJointEvaluation(np.zeros(1), None, None, None, None, None, None, **{name: bad})


@pytest.mark.parametrize('name', ['differentiated_value', 'physical_acceleration_offset'])
def test_joint_lifted_vectors_are_copied_readonly(name):
    original = np.ones(1)
    sample = InitialJointEvaluation(np.zeros(1), None, None, None, None, None, None, **{name: original})
    original[0] = 3.0
    np.testing.assert_array_equal(getattr(sample, name), [1.0])
    with pytest.raises(ValueError, match='read-only'):
        getattr(sample, name)[0] = 4.0


def test_large_sparse_joint_chart_never_densifies_or_exact_rationalizes(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('joint sparse bulk must not densify or rationalize field columns')

    monkeypatch.setattr(sp.csr_matrix, 'toarray', forbidden)
    monkeypatch.setattr(initial_rate, 'Fraction', forbidden)
    n = 2000
    identity, zero = sp.eye(n, format='csr'), sp.csr_matrix((n, n))

    def chart(_u, r):
        return InitialJointEvaluation(r, identity, identity, zero, identity, zero, identity, r, np.zeros(n))

    pair = finite_storage_actual_rate_initial_pair(
        np.zeros(n),
        joint_coordinates_at=chart,
        differentiated_residual_at=lambda _u, v, a: v + a,
        residual_at=lambda u, v: u + v - 1,
        jacobian_at=lambda _u, _v: identity,
        mass_at=lambda _u, _v: identity,
        storage_at=lambda u: u.copy(),
        partial_time_at=lambda _u, _v: np.zeros(n),
        storage_rows_at=lambda _u, _v: StorageRows(np.arange(n), sp.csr_matrix((0, n))),
        rate_constraint_jacobian_at=lambda _u, _v: identity,
    )
    np.testing.assert_array_equal(pair.state, np.zeros(n))
    np.testing.assert_array_equal(pair.rate, np.ones(n))
    assert pair.rate_is_unique.all()


@pytest.mark.parametrize('option', ['missing', 'malicious', 'tiny'])
def test_unowned_or_false_joint_certificate_cannot_override_original_physics(option):
    options, chart = _problem(**{option: True})
    with pytest.raises((ValueError, NotImplementedError)):
        finite_storage_actual_rate_initial_pair(np.zeros(4), joint_coordinates_at=chart, **options)
