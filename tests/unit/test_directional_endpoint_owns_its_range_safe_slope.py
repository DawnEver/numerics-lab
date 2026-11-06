"""UNVERIFIED finite endpoint geometry and range-only exact slope fallback controls."""

from fractions import Fraction

import numpy as np
import pytest

from numerics_lab.line_search import TrialNotEvaluable, _directional_dot, directional_line_search

pytestmark = pytest.mark.unit


def test_extreme_endpoint_difference_retains_tiny_direction_after_large_terms_cancel():
    maximum, tiny = np.finfo(float).max, np.nextafter(0.0, 1.0)
    start, endpoint = np.array([-maximum, -maximum, 0.0]), np.array([maximum, maximum, tiny])
    r0 = np.array([1.0, -1.0, -maximum])
    calls = []

    def residual(point):
        calls.append(point.copy())
        return -r0 if point[0] > 0.0 else r0

    with np.errstate(over='raise', invalid='raise'):
        result = directional_line_search(residual, start, None, r0, endpoint=endpoint)
    assert len(calls) > 1 and result[0] <= 0.0
    assert np.isfinite(result).all() and not np.array_equal(result, endpoint)


def test_mixed_finite_direction_keeps_the_original_tiny_but_meaningful_dot_component():
    direction = np.array([1e308, 1e-320])
    r0 = np.array([0.0, -1e308])
    assert _directional_dot(r0, direction) == float(np.dot(r0, direction)) < 0.0
    calls = []

    def residual(point):
        calls.append(point.copy())
        return -r0 if point[0] > 5e307 else r0

    result = directional_line_search(residual, np.zeros(2), direction, r0)
    assert len(calls) > 1 and result[0] <= 5e307


def test_overflowed_products_cancel_without_erasing_the_small_slope():
    maximum = np.finfo(float).max
    direction, residual = np.array([maximum, maximum, 1.0]), np.array([maximum, -maximum, -1.0])
    with np.errstate(over='raise', invalid='raise'):
        assert _directional_dot(residual, direction) == Fraction(-1)


def test_nonfinite_norms_use_exact_comparison_without_changing_the_smooth_finish():
    maximum = np.finfo(float).max
    r0, r1 = np.array([-maximum, maximum]), np.array([maximum / 4.0, maximum / 4.0])
    endpoint = np.array([1.0, 0.0])
    with np.errstate(over='raise', invalid='raise'):
        result = directional_line_search(lambda _point: r1, np.zeros(2), None, r0, endpoint=endpoint)
    np.testing.assert_array_equal(result, endpoint)


@pytest.mark.parametrize('bad', [None, np.nan, np.inf])
def test_non_descent_cannot_return_an_unevaluated_or_nonfinite_endpoint(bad):
    def residual(point):
        if point[0] > 0.5:
            if bad is None:
                msg = 'this endpoint has no constitutive value'
                raise TrialNotEvaluable(msg)
            return np.array([bad])
        return np.ones(1)

    result = directional_line_search(residual, np.zeros(1), None, np.ones(1), endpoint=np.ones(1))
    assert 0.0 < result[0] <= 0.5
    np.testing.assert_array_equal(residual(result), np.ones(1))


def test_no_evaluable_non_descent_trial_is_still_a_refusal_not_a_full_step():
    def residual(_point):
        msg = 'no trial can be evaluated'
        raise TrialNotEvaluable(msg)

    with pytest.raises(TrialNotEvaluable, match='no trial'):
        directional_line_search(residual, np.zeros(1), None, np.ones(1), endpoint=np.ones(1))
