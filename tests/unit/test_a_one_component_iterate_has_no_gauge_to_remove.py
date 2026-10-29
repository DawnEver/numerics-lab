"""``increment_ratio`` reported EXACTLY ZERO for every one-component step, and six strategies believed it.

THE DEFECT, STATED AS ITS ARITHMETIC. :func:`~numerics_lab.nonlinear.increment_ratio` carries a
gauge relaxation: in 2-D magnetostatics the unknown ``A`` is defined only up to an additive
constant unless a Dirichlet row pins it, so an increment that is PURE CONSTANT MODE is not
movement, and the function reports the mean-removed part instead when the shape part is negligible
against the increment::

    shape = dx - dx.mean()
    if ||shape|| / ||dx|| < _GAUGE_ONLY_SHAPE_FRACTION:   # 1e-3
        return ||shape|| / ||x||

For ``n = 1`` that test is not a test. The shape subspace of a one-component vector is ``{0}``:
``dx - dx.mean()`` is identically zero for EVERY possible increment, ``0.0 / ||dx||`` is ``0.0``,
``0.0 < 1e-3`` always holds, and the function returns ``0.0`` -- "the iterate has stopped moving"
-- on a step of any size whatsoever.

WHAT THAT COST, MEASURED. ``increment_ratio([5.0], [7.5])`` returned ``0.0``, so the second half of
`_has_converged` -- the direct fixed-point test that exists precisely because a flat energy is not
a fixed point -- was satisfied on a step of 50 %. On the plateau fixture of
``test_a_flat_energy_plateau_is_not_convergence`` every Picard- and Newton-family strategy then
returned ``converged=True`` at ``x = 7.5`` against a fixed point of ``10.0``, three iterations in:
the exact failure those files were written to refuse, reintroduced underneath them by a change to
the metric rather than to any strategy. Nineteen tests across four files were red on it.

WHY A SEPARATE FILE RATHER THAN A CASE IN THOSE. Every one of them drives the metric THROUGH a
strategy, which is what let a metric defect read as twelve strategy defects. This file asserts the
metric directly, on the two-line arithmetic above, so the next reader is told which layer lied.

THE FIX IS NOT A TOLERANCE. At ``n = 1`` the constant/shape decomposition carries no information,
so there is nothing to relax towards: the raw ratio is the only thing there is to report. The
relaxation stays exactly as measured for ``n >= 2``, where a constant mode and a shape are
genuinely different directions.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.nonlinear import increment_ratio


def test_a_one_component_step_reports_its_actual_size() -> None:
    """The defect, at the two numbers that produced it: 5 to 7.5 is half the iterate, not nothing."""
    assert increment_ratio(np.array([5.0]), np.array([7.5])) == pytest.approx(0.5)


@pytest.mark.parametrize('step', [1e-6, 1e-3, 0.5, 4.0])
def test_no_one_component_step_is_reported_as_zero(step: float) -> None:
    """Across six decades, because the old return was zero for ALL of them, not for small ones."""
    x = np.array([2.0])
    assert increment_ratio(x, x * (1.0 + step)) == pytest.approx(step)


def test_the_genuinely_stationary_one_component_step_is_still_zero() -> None:
    """The converse, or "never report zero" would pass this file."""
    assert increment_ratio(np.array([3.0]), np.array([3.0])) == 0.0


def test_the_gauge_relaxation_still_fires_where_it_was_MEASURED() -> None:
    """``n >= 2``, pure constant mode: the case_c defect this relaxation was added for.

    Measured 2026-09-06 on ``case_c``: raw increment 3.5e-02 against a criterion
    needing 1e-4, mean-removed 1.0e-06, constant fraction 1.0000 to four decimals on all 400
    instrumented steps, energy flat to six digits. The solve was physically converged and the
    CRITERION was measuring a gauge mode. Fixing the ``n = 1`` degeneracy must not undo that.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    drifted = x + 0.035
    assert increment_ratio(x, drifted) < 1.0e-12
    assert float(np.linalg.norm(drifted - x) / np.linalg.norm(x)) > 1.0e-2


def test_a_REAL_shape_change_on_two_components_is_still_reported_in_full() -> None:
    """The other side of the same ratchet: the relaxation is narrow, not a blanket mean removal."""
    x = np.array([1.0, 1.0])
    moved = np.array([1.0, 1.1])
    assert increment_ratio(x, moved) == pytest.approx(0.1 / np.sqrt(2.0))
