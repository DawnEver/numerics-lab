"""`directional_line_search` minimizes a convex step's ENERGY along the Newton ray using residuals only.

THE CASE IT EXISTS FOR: a residual that is the gradient of a convex but KINKED potential (a play-model
steel). The tangent taken on one side of a kink overshoots by orders of magnitude, so the full step
RAISES the energy; the search must return a point on the LOW side of the line minimum, where
``phi'(s) = R(u + s step) . step <= 0`` and so, by convexity, the energy fell.
"""

import numpy as np

from numerics_lab.time_integrators import directional_line_search

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
