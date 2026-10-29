"""`directional_line_search` minimizes a convex step's ENERGY along the Newton ray using residuals only.

THE CASE IT EXISTS FOR: a residual that is the gradient of a convex but KINKED potential (a play-model
steel). The tangent taken on one side of a kink overshoots by orders of magnitude, so the full step
RAISES the energy; the search must return a point on the LOW side of the line minimum, where
``phi'(s) = R(u + s step) . step <= 0`` and so, by convexity, the energy fell.
"""

import numpy as np
import pytest

from numerics_lab.line_search import TrialNotEvaluable, directional_line_search

STIFF, SOFT, KNEE = 1e3, 1.0, 1.0


def _energy(u):
    """Piecewise quadratic, convex: SOFT up to the knee, STIFF beyond it (per component)."""
    a = np.abs(u)
    inside = 0.5 * SOFT * np.minimum(a, KNEE) ** 2
    beyond = SOFT * KNEE * np.maximum(a - KNEE, 0.0) + 0.5 * STIFF * np.maximum(a - KNEE, 0.0) ** 2
    return float(np.sum(inside + beyond - 0.3 * u))


def _residual(u):
    a = np.abs(u)
    g = np.where(a <= KNEE, SOFT * a, SOFT * KNEE + STIFF * (a - KNEE))
    return np.sign(u) * g - 0.3


def test_an_overshooting_soft_side_step_lands_below_the_start_energy():
    u = np.array([0.0, 0.0])
    r = _residual(u)
    step = -r / SOFT * 10.0  # a tangent from the soft side, scaled to overshoot past the knee
    assert _energy(u + step) > _energy(u)  # THE PLANTED CONTROL: the full step is worse
    out = directional_line_search(_residual, u, step, r)
    assert _energy(out) < _energy(u)
    assert float(np.dot(_residual(out), step)) <= 0.0  # the low side of the line minimum


def test_a_full_step_that_still_descends_is_taken_whole():
    u = np.array([0.0])
    r = _residual(u)
    step = -0.5 * r / SOFT
    np.testing.assert_array_equal(directional_line_search(_residual, u, step, r), u + step)


def test_a_direction_that_does_not_descend_is_taken_whole():
    u = np.array([0.0])
    r = _residual(u)
    step = r.copy()  # uphill: nothing to minimize along it
    np.testing.assert_array_equal(directional_line_search(_residual, u, step, r), u + step)


def _uncertifiable_beyond(limit):
    """The residual, UNEVALUABLE past ``|u| > limit`` -- a law whose certified inverse refuses there."""

    def residual(u):
        if np.max(np.abs(u)) > limit:
            msg = 'the law cannot certify this trial'
            raise TrialNotEvaluable(msg)
        return _residual(u)

    return residual


def test_an_uncertifiable_trial_is_rejected_and_the_step_shrinks():
    """A line-search TRIAL is speculative: one the problem cannot certify is rejected like an overshoot, never raised
    (ruling 2026-10-19). Only the point the search RETURNS was evaluated -- and so certified.
    """
    u = np.array([0.0, 0.0])
    r = _residual(u)
    step = -r / SOFT * 10.0  # the full step (|u| = 3) lies past the uncertifiable limit
    residual = _uncertifiable_beyond(0.5)
    out = directional_line_search(residual, u, step, r)
    assert np.max(np.abs(out)) <= 0.5
    assert _energy(out) < _energy(u)
    residual(out)  # the returned point evaluates


def test_planted_control_a_step_with_no_certifiable_trial_raises():
    u = np.array([0.0, 0.0])
    r = _residual(u)
    with pytest.raises(TrialNotEvaluable):
        directional_line_search(_uncertifiable_beyond(0.0), u, -r / SOFT * 10.0, r)


def test_a_full_step_just_past_the_minimum_that_lowers_the_residual_is_taken_whole():
    """A Newton step on a SMOOTH law overshoots its line minimum by a sliver near the root (``0 < phi'(1)``
    well inside the curvature band) and lowers ``|R|``: it is taken whole. Searching it moved two marches of
    one problem onto different in-tolerance iterates (MEASURED 2026-10-20, composite vs pure sdm: s =
    0.9997, 1.1e-3 apart on a held slab) and cost the step its quadratic finish.
    """
    u = np.array([0.0])
    r = _residual(u)
    step = -1.01 * r / SOFT  # one percent past the soft side's exact root
    assert float(np.dot(_residual(u + step), step)) > 0.0  # THE PLANTED CONTROL: it IS past the minimum
    np.testing.assert_array_equal(directional_line_search(_residual, u, step, r), u + step)
