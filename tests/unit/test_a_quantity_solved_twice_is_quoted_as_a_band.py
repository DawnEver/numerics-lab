"""The mesh-convergence band, and the real annulus measured through it.

P1 constant-strain triangles give an element-CONSTANT stress that converges one order slower than
the displacement, and at a concentration it converges FROM BELOW -- so a coarse mesh reads an
OPTIMISTIC safety factor. A single-mesh peak is a statement about that mesh while looking like a
property of the design, which is the shape this repo treats as the dominant defect.

The band is the answer the plan chose over reaching for P2 elements up front: measure the
residual, quote an interval, and let the WIDTH of that interval be the evidence that buys a better
element if it ever does.
"""

from __future__ import annotations

import numpy as np
import pytest

from motronics.core.utils.config import repo_root
from motronics.gauss.convergence import ConvergenceBand, convergence_band
from motronics.hamilton.physics.structural.validation import StructuralCaseError, structural_case_model

pytestmark = pytest.mark.unit

_CASE = 'cases/field/benchmark/rotor_stress'


# ---------------------------------------------------------------------------
# The band itself
# ---------------------------------------------------------------------------


def test_the_residual_is_the_relative_move_of_the_LAST_refinement() -> None:
    band = convergence_band(lambda r: {1: 100.0, 2: 108.0, 4: 110.0}[r], refinements=(1, 2, 4))

    assert isinstance(band, ConvergenceBand)
    assert band.finest == pytest.approx(110.0)
    assert band.residual == pytest.approx(2.0 / 110.0)
    lo, hi = band.band()
    assert (lo, hi) == pytest.approx((108.0, 112.0))


def test_the_band_is_SYMMETRIC_even_though_P1_converges_from_below() -> None:
    """Deliberate. The residual measures how far the last step MOVED; where the remaining error
    lies is an argument, and a one-sided interval would encode it as a measurement.
    """
    band = convergence_band(lambda r: {1: 100.0, 2: 110.0}[r], refinements=(1, 2))
    lo, hi = band.band()

    assert hi - band.finest == pytest.approx(band.finest - lo)


def test_a_sequence_whose_differences_GROW_is_refused_rather_than_banded() -> None:
    """The floor under every band above. A divergent run's finest value is not its best one, and
    an interval around it would be a confidence claim on a run that supports none.
    """
    band = convergence_band(lambda r: {1: 100.0, 2: 101.0, 4: 111.0}[r], refinements=(1, 2, 4))

    assert not band.converging
    with pytest.raises(ValueError, match='not converging'):
        band.band()


def test_two_points_are_vacuously_converging_and_three_is_the_first_real_answer() -> None:
    """Stated so nobody reads a two-point ``converging`` as evidence: there is one difference and
    nothing to compare it against.
    """
    two = convergence_band(lambda r: {1: 100.0, 2: 111.0}[r], refinements=(1, 2))
    assert two.converging

    three = convergence_band(lambda r: {1: 100.0, 2: 101.0, 4: 111.0}[r], refinements=(1, 2, 4))
    assert not three.converging, 'the same last step, now judged against the one before it'


def test_a_single_refinement_is_refused() -> None:
    with pytest.raises(ValueError, match='at least two refinements'):
        convergence_band(lambda _r: 1.0, refinements=(2,))


@pytest.mark.parametrize('bad', [(2, 1), (1, 1), (0, 1)])
def test_refinements_must_be_strictly_ascending_positive_integers(bad: tuple[int, ...]) -> None:
    """The ORDER is load-bearing: the last entry is taken as the finest."""
    with pytest.raises(ValueError, match='strictly ascending'):
        convergence_band(lambda _r: 1.0, refinements=bad)


def test_a_non_finite_solve_is_refused_rather_than_banded() -> None:
    with pytest.raises(ValueError, match='cannot be built on it'):
        convergence_band(lambda r: float('nan') if r == 2 else 1.0, refinements=(1, 2))


# ---------------------------------------------------------------------------
# The refinement knob
# ---------------------------------------------------------------------------


def _model(refine: int):
    return structural_case_model(repo_root() / _CASE, rpm=12000.0, refine=refine)


def test_the_refinement_multiplies_the_declared_element_counts_on_both_axes() -> None:
    """A refinement that only touched one axis would converge on a different limit."""
    base = _model(1)
    fine = _model(2)

    assert len(fine.ele_mat) == 4 * len(base.ele_mat), 'both axes doubled -> 4x the elements'
    assert len(fine.vert_mat) > len(base.vert_mat)
    # ...and the GEOMETRY is unchanged, so the two meshes are the same case.
    assert np.hypot(fine.vert_mat[:, 0], fine.vert_mat[:, 1]).max() == pytest.approx(
        np.hypot(base.vert_mat[:, 0], base.vert_mat[:, 1]).max(), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize('bad', [0, -1])
def test_a_refinement_below_one_is_refused_because_it_COARSENS_the_declared_case(bad: int) -> None:
    with pytest.raises(StructuralCaseError, match='positive integer multiplier'):
        _model(bad)
