"""A trial limit bounds Newton's proposals, never changes the residual or its acceptance bar."""

import math
from pathlib import Path

import numpy as np
import pytest

import numerics_lab
import numerics_lab.time_integrators._step as step_module
from numerics_lab.formulation import DerivativeWeights, IntegratorKind
from numerics_lab.line_search import TrialNotEvaluable, directional_line_search
from numerics_lab.time_integrators import get_integrator
from numerics_lab.time_integrators._forms import StepForm
from numerics_lab.time_integrators._step import take_step
from numerics_lab.time_integrators._trials import project_newton_trial

pytestmark = pytest.mark.unit


class LimitedProblem:
    def __init__(self) -> None:
        self.allowed = (0.0, 0.0)
        self.proposals = []

    def residual(self, u, rate, acceleration, time):
        assert self.allowed[0] - 1e-14 <= u[0] <= self.allowed[1] + 1e-14, 'an unlimited trial reached the law'
        return u - 10.0

    def jacobian(self, u, rate, acceleration, time, weights):
        return np.eye(1) * weights.alpha0.real

    def limit_newton_trial(self, previous, proposed, time):
        limited = previous + np.clip(proposed - previous, -1.0, 1.0)
        self.proposals.append((previous.copy(), proposed.copy(), limited.copy()))
        self.allowed = (min(previous[0], limited[0]), max(previous[0], limited[0]))
        return limited


def _advance(problem, **options):
    assert Path(numerics_lab.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2] / 'python')
    scheme = get_integrator(IntegratorKind.BDF1)
    initial = options.pop('initial', np.zeros(getattr(problem, 'size', 1)))
    return take_step(
        problem, scheme, scheme.start(initial, 0.0), 1.0, 1.0,
        solve=np.linalg.solve, max_newton=options.pop('max_newton', 20), tol=1e-13, **options,
    )  # fmt: skip


@pytest.mark.parametrize('search', ['backtrack', 'directional'])
@pytest.mark.parametrize('cap', [0, 8])
def test_every_search_limits_the_trial_before_evaluating_the_law(search, cap):
    problem = LimitedProblem()
    outcome = _advance(problem, line_search=search, backtrack_cap=cap)
    assert outcome.converged
    np.testing.assert_array_equal(outcome.u, [10.0])
    assert len(problem.proposals) >= 10
    np.testing.assert_array_equal(problem.proposals[0][1], [10.0])
    np.testing.assert_array_equal(problem.proposals[0][2], [1.0])


def test_a_native_kernel_cannot_bypass_a_limit_it_does_not_implement():
    class Native(LimitedProblem):
        def native_step(self, *args, **kwargs):
            msg = 'the native kernel has no trial-limit contract'
            raise AssertionError(msg)

    assert _advance(Native()).converged


def test_the_gmin_continuation_also_limits_its_proposals():
    problem = LimitedProblem()
    outcome = _advance(problem, max_newton=1, gmin_stages=tuple(10.0**-k for k in range(1, 11)))
    assert outcome.converged
    np.testing.assert_allclose(outcome.u, [10.0], rtol=0, atol=1e-12)
    assert len(problem.proposals) >= 10


@pytest.mark.parametrize('bad', [np.zeros(2), np.array([np.nan]), np.array([1j])])
def test_a_limit_must_return_a_same_shape_finite_real_state(bad):
    class Invalid(LimitedProblem):
        def limit_newton_trial(self, previous, proposed, time):
            return bad

    with pytest.raises(ValueError, match='same-shape finite real'):
        _advance(Invalid())


class AffineLimitedProblem:
    size = 2

    def __init__(self) -> None:
        self.evaluated = []

    def residual(self, u, rate, acceleration, time):
        self.evaluated.append(u.copy())
        return u - np.array([1.0, 10.0])

    def jacobian(self, u, rate, acceleration, time, weights):
        return np.eye(2) * weights.alpha0.real

    def project_affine_state(self, u, time):
        out = u.copy()
        out[0] = 1.0
        return out

    def affine_state_is_admitted(self, u, time):
        return bool(u[0] == 1.0)

    def limit_newton_trial(self, previous, proposed, time):
        assert self.affine_state_is_admitted(proposed, time), 'projection must precede the limiter'
        out = proposed.copy()
        out[1] = previous[1] + np.clip(out[1] - previous[1], -1.0, 1.0)
        return out


def test_gmin_projects_affine_constraints_before_limiting_the_trial():
    problem = AffineLimitedProblem()
    result = _advance(problem, max_newton=1, gmin_stages=tuple(10.0**-k for k in range(1, 11)))
    assert result.converged
    np.testing.assert_allclose(result.u, [1.0, 10.0], rtol=0, atol=1e-12)


def test_directional_search_starts_on_the_affine_plane_without_projecting_the_first_linearization(monkeypatch):
    problem, searches = AffineLimitedProblem(), []
    search = step_module.directional_line_search

    def checked(residual, previous, direction, r0, **options):
        assert problem.affine_state_is_admitted(previous, 1.0), 'energy search needs a feasible starting ray'
        searches.append(previous.copy())
        return search(residual, previous, direction, r0, **options)

    monkeypatch.setattr(step_module, 'directional_line_search', checked)
    assert _advance(problem, line_search='directional').converged
    np.testing.assert_array_equal(problem.evaluated[0], [0.0, 0.0])
    assert searches


@pytest.mark.parametrize('weight', [0.25, 1.0, -0.25])
def test_the_limit_sees_the_physical_stage_and_not_the_endpoint_coordinate(weight):
    zero = np.zeros(1)
    form = StepForm(DerivativeWeights(weight, 0, 0), np.array([2.0]), zero, zero, 1.0)
    problem = LimitedProblem()
    endpoint, limited = project_newton_trial(
        problem, form, zero, np.array([100.0 / weight]), affine_projection=False, newton_limit=True
    )
    np.testing.assert_allclose(form.u_arg(endpoint), [3.0], rtol=0, atol=1e-14)
    assert limited
    np.testing.assert_array_equal(problem.proposals[0][0], [2.0])
    np.testing.assert_array_equal(problem.proposals[0][1], [102.0])


def test_a_finite_physical_limit_must_also_have_a_finite_lifted_endpoint():
    class Limit:
        def limit_newton_trial(self, previous, proposed, time):
            return np.ones(1)

    zero = np.zeros(1)
    form = StepForm(DerivativeWeights(1e-320, 0, 0), zero, zero, zero, 1.0)
    with pytest.raises(ValueError, match='finite endpoint'):
        project_newton_trial(Limit(), form, zero, np.ones(1), affine_projection=False, newton_limit=True)


def test_the_exact_exponential_converges_without_a_permanent_constitutive_cap():
    class Exponential:
        def residual(self, u, rate, acceleration, time):
            return np.array([math.expm1(float(u[0])) - 1000.0])

        def jacobian(self, u, rate, acceleration, time, weights):
            return np.array([[math.exp(float(u[0])) * weights.alpha0.real]])

        def limit_newton_trial(self, previous, proposed, time):
            return previous + np.clip(proposed - previous, -2.0, 2.0)

    problem = Exponential()
    with pytest.raises(OverflowError):
        problem.residual(np.array([1000.0]), None, None, 1.0)
    result = _advance(problem)
    assert result.converged
    np.testing.assert_allclose(result.u, [math.log1p(1000.0)], rtol=0, atol=2e-12)


class _IllConditioned:
    size = 2
    matrix = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]])

    def residual(self, u, rate, acceleration, time):
        return self.matrix @ u - np.array([1.0, -1.0])

    def jacobian(self, u, rate, acceleration, time, weights):
        return weights.alpha0.real * self.matrix

    def limit_newton_trial(self, previous, proposed, time):
        return previous + np.clip(proposed - previous, -1e-20, 1e-20)


def test_a_tiny_restricted_step_cannot_borrow_the_unexecuted_full_steps_roundoff_floor():
    result = _advance(_IllConditioned(), max_newton=4)
    assert not result.converged, 'the restricted iterate still has an order-one physical residual'
    assert result.residual > 1.0


def test_an_unevaluable_probe_cannot_replace_the_retained_iterates_floor_provenance():
    class RejectsOtherLawDomain(_IllConditioned):
        def residual(self, u, rate, acceleration, time):
            if np.max(np.abs(u)) > 2e-20:
                msg = 'a different constitutive law cannot evaluate this trial'
                raise TrialNotEvaluable(msg)
            return super().residual(u, rate, acceleration, time)

        def limit_newton_trial(self, previous, proposed, time):
            return proposed if np.any(previous != 0.0) else super().limit_newton_trial(previous, proposed, time)

    result = _advance(RejectsOtherLawDomain(), max_newton=4)
    assert not result.converged
    assert result.residual > 1.0


@pytest.mark.parametrize('offset', [0.0, 2.0])
@pytest.mark.parametrize('weight', [1.0, 0.25])
def test_lifting_a_small_bound_does_not_cancel_against_the_discarded_large_proposal(offset, weight):
    zero = np.zeros(1)
    form = StepForm(DerivativeWeights(weight, 0, 0), np.array([offset]), zero, zero, 1.0)
    endpoint, limited = project_newton_trial(
        LimitedProblem(), form, zero, np.array([1e20]), affine_projection=False, newton_limit=True
    )
    np.testing.assert_array_equal(form.u_arg(endpoint), [offset + 1.0])
    assert limited


@pytest.mark.parametrize('limited', [False, True])
def test_affine_projection_preserves_untouched_endpoint_rows_with_nonbinary_weights_and_history(limited):
    class Projected:
        def project_affine_state(self, state, time):
            state[0] = 1.0
            return state

        def limit_newton_trial(self, previous, proposed, time):
            return proposed

    zero = np.zeros(2)
    form = StepForm(DerivativeWeights(0.3, 0, 0), np.array([0.7, 0.7]), zero, zero, 1.0)
    proposed = np.array([0.0, 0.12345678901234568])
    endpoint, changed = project_newton_trial(
        Projected(), form, zero, proposed, affine_projection=True, newton_limit=limited
    )
    expected = proposed + (np.array([1.0, form.u_arg(proposed)[1]]) - form.u_arg(proposed)) / 0.3
    np.testing.assert_array_equal(endpoint, expected)
    assert endpoint[1].tobytes() == proposed[1].tobytes()
    assert form.u_arg(endpoint)[0] == 1.0
    assert not changed, 'affine projection is not evidence of limiter alteration'


def test_directional_search_keeps_the_representable_bounded_endpoint_after_a_huge_correction():
    class PositiveDomain:
        def __init__(self) -> None:
            self.seen = []

        def residual(self, u, rate, acceleration, time):
            self.seen.append(u.copy())
            if u[0] <= 0.0:
                msg = 'the logarithm requires a positive trial'
                raise TrialNotEvaluable(msg)
            return np.array([math.log(float(u[0]))])

        def jacobian(self, u, rate, acceleration, time, weights):
            return np.array([[weights.alpha0.real / u[0]]])

        def limit_newton_trial(self, previous, proposed, time):
            return np.maximum(proposed, 1.0)

    problem = PositiveDomain()
    outcome = _advance(problem, initial=np.array([1e20]), line_search='directional', max_newton=4)
    assert outcome.converged
    np.testing.assert_array_equal(outcome.u, [1.0])
    np.testing.assert_array_equal(problem.seen[1], [1.0])


def test_an_identity_limit_keeps_the_unrestricted_solver_and_its_floor_evidence():
    class Plain:
        size = 2
        matrix = _IllConditioned.matrix
        residual = _IllConditioned.residual
        jacobian = _IllConditioned.jacobian

    class Identity(Plain):
        def limit_newton_trial(self, previous, proposed, time):
            return proposed

    plain, limited = _advance(Plain()), _advance(Identity())
    assert plain.converged
    np.testing.assert_array_equal(limited.u, plain.u)
    assert (limited.converged, limited.on_floor, limited.residual) == (plain.converged, plain.on_floor, plain.residual)


@pytest.mark.parametrize('has_limit', [False, True])
def test_no_accepted_trial_means_no_new_linear_solve_floor_evidence(has_limit):
    class Domain:
        size = 2
        matrix = _IllConditioned.matrix
        jacobian = _IllConditioned.jacobian

        def residual(self, u, rate, acceleration, time):
            if np.any(u != 0.0):
                msg = 'no nonzero state is in this law domain'
                raise TrialNotEvaluable(msg)
            return self.matrix @ u - np.array([1.0, -1.0])

    if has_limit:

        class Problem(Domain):
            def limit_newton_trial(self, previous, proposed, time):
                return proposed

    else:
        Problem = Domain
    result = _advance(Problem(), max_newton=4)
    assert not result.converged
    assert result.residual > 1.0


@pytest.mark.parametrize(
    ('start', 'direction', 'endpoint'),
    [([0.0], [2.0], [1.0]), ([0.0], [np.inf], [1.0]), ([np.finfo(float).max], [-np.inf], [-np.finfo(float).max])],
)
def test_an_explicit_endpoint_cannot_declare_a_different_or_nonfinite_direction(start, direction, endpoint):
    with pytest.raises(ValueError, match='direction'):
        directional_line_search(lambda _u: np.zeros(1), np.array(start), np.array(direction), np.ones(1),
                                endpoint=np.array(endpoint))  # fmt: skip
