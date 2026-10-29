"""The public line searches: backtracking, Armijo, strong Wolfe -- each step meets its stated condition."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import rosen, rosen_der

from numerics_lab.line_search import (
    WolfeSearchFailed,
    armijo,
    backtracking,
    strong_wolfe,
    sufficient_decrease,
)


def test_backtracking_takes_the_first_accepted_step() -> None:
    seen = []
    result = backtracking(lambda s: seen.append(s) or s, lambda s, _v: s <= 0.3, tries=5)
    assert result.accepted
    assert result.step == 0.25
    assert seen == [1.0, 0.5, 0.25]
    assert result.first == 1.0


def test_an_unevaluable_trial_is_rejected_and_shrinks() -> None:
    result = backtracking(lambda s: None if s > 0.5 else s, lambda _s, _v: True, tries=4)
    assert result.accepted
    assert result.step == 0.5
    assert result.first is None


def test_an_exhausted_search_reports_the_next_untried_step_and_the_last_trial() -> None:
    result = backtracking(lambda s: s, lambda _s, _v: False, tries=3)
    assert not result.accepted
    assert result.step == 0.125
    assert result.trial == 0.25


def test_armijo_accepts_only_sufficient_decrease() -> None:
    # phi(s) = (1 - s)^2 * 4 - 2 along a quadratic; phi(0) = 2, phi'(0) = -8.
    phi = lambda s: 4.0 * (1.0 - s) ** 2 - 2.0  # noqa: E731
    holds = sufficient_decrease(2.0, -8.0, c1=0.5)
    assert holds(0.5, phi(0.5))
    # Planted control: just past the full step the decrease falls short of c1 times the prediction.
    assert not holds(1.0 + 1e-9, phi(1.0 + 1e-9))
    assert not holds(0.1, float('nan'))
    result = armijo(phi, 2.0, -8.0, c1=0.5)
    assert result.accepted
    assert phi(result.step) <= 2.0 + 0.5 * result.step * -8.0


def test_armijo_refuses_an_ascent_direction() -> None:
    with pytest.raises(ValueError, match='descent'):
        sufficient_decrease(1.0, 1.0)


@pytest.mark.parametrize('c2', [0.9, 0.1])
def test_strong_wolfe_on_rosenbrock(c2) -> None:
    x = np.array([-1.2, 1.0])
    g = rosen_der(x)
    p = -g / np.linalg.norm(g)
    phi_and_slope = lambda a: (rosen(x + a * p), float(rosen_der(x + a * p) @ p))  # noqa: E731
    phi0, slope0 = phi_and_slope(0.0)
    found = strong_wolfe(phi_and_slope, phi0, slope0, c2=c2)
    assert found.phi <= phi0 + 1e-4 * found.step * slope0
    assert abs(found.slope) <= c2 * abs(slope0)
    # The oracle (tests only) finds a step meeting the same two conditions.
    from scipy.optimize import line_search  # noqa: PLC0415

    oracle = line_search(rosen, rosen_der, x, p, gfk=g, old_fval=phi0, c2=c2)[0]
    assert oracle is not None
    assert abs(float(rosen_der(x + oracle * p) @ p)) <= c2 * abs(slope0)


def test_strong_wolfe_brackets_past_a_too_short_first_step() -> None:
    phi_and_slope = lambda a: ((a - 10.0) ** 2, 2.0 * (a - 10.0))  # noqa: E731
    found = strong_wolfe(phi_and_slope, 100.0, -20.0, c2=0.1)
    assert abs(found.slope) <= 2.0
    assert found.step > 1.0


def test_strong_wolfe_raises_rather_than_return_a_failing_step() -> None:
    # Unbounded below along the ray: no Wolfe point exists.
    with pytest.raises(WolfeSearchFailed):
        strong_wolfe(lambda a: (-a, -1.0), 0.0, -1.0, max_evaluations=10)


def test_strong_wolfe_reports_a_collapsed_bracket_as_a_failed_search() -> None:
    """A jump at a = 0.5 holds no Wolfe point; zoom bisects until lo == hi and must say so by name."""

    def jump(a: float) -> tuple[float, float]:
        return (-a, -1.0) if a < 0.5 else (1.0, -1.0)

    with pytest.raises(WolfeSearchFailed, match='collapsed'):
        strong_wolfe(jump, 0.0, -1.0, max_evaluations=500)


def test_strong_wolfe_returns_a_capped_step_where_phi_still_falls_at_max_step() -> None:
    """More and Thuente's stpmax exit: sufficient decrease holds at the cap, the curvature condition cannot."""
    found = strong_wolfe(lambda a: (-a, -1.0), 0.0, -1.0, max_step=1.0)
    assert (found.step, found.capped) == (1.0, True)
    assert not strong_wolfe(lambda a: ((a - 0.5) ** 2 - 0.25, 2 * a - 1.0), 0.0, -1.0, max_step=1.0).capped


def test_strong_wolfe_still_raises_at_a_cap_where_sufficient_decrease_fails() -> None:
    """Planted control: a cap is not a licence -- phi rising past the cap's Armijo line still zooms or fails."""
    found = strong_wolfe(lambda a: (-a + 2 * a * a, -1.0 + 4 * a), 0.0, -1.0, max_step=1.0)
    assert not found.capped and abs(found.slope) <= 0.9


@pytest.mark.parametrize('slope0', [-0.0, 0.0, 1.0, float('nan')])
def test_strong_wolfe_refuses_a_non_descent_slope_by_its_declared_type(slope0) -> None:
    # -0.0 is a gradient whose square underflowed: no descent is visible, so no step is found.
    with pytest.raises(WolfeSearchFailed, match='descent'):
        strong_wolfe(lambda a: (a * a, 2.0 * a), 0.0, slope0)


def test_strong_wolfe_still_searches_along_a_descent_slope() -> None:
    # Planted control: the same parabola from the descent side searches and lands at its minimum.
    found = strong_wolfe(lambda a: ((a - 1.0) ** 2, 2.0 * (a - 1.0)), 1.0, -2.0)
    assert found.step == pytest.approx(1.0)
