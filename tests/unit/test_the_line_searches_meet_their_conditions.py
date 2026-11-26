"""The public line searches: backtracking and Armijo -- each step meets its stated condition."""

from __future__ import annotations

import pytest

from numerics_lab.line_search import (
    armijo,
    backtracking,
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
