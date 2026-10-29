"""ONE QUANTITY SOLVED AT SEVERAL REFINEMENTS -- what turns a number into a number with a bound.

THE REFINEMENT AXIS IS THE CALLER'S, and that is why this module sits in `gauss` rather than beside
the physics that first needed it. `solve_at(r)` takes an integer multiplier and returns a scalar;
what `r` REFINES is the caller's to decide. Two are live: the structural path refines the MESH
(`cli/_structural_report.py`), and the transient field-circuit path refines TIME, marching the same
window at `dt / r`. The band arithmetic is identical because it never looks at the axis -- it looks
at whether successive values stop moving. The module docstring used to OPEN with "the
mesh-convergence band", which was true when there was one caller and became a declaration that lies
the moment there were two.

IT WAS BORN MESH-SHAPED, and the original argument is kept because it is the clearest statement of
why a band beats a number. P1 constant-strain triangles give an element-CONSTANT stress that
converges one order slower than the displacement, and it converges FROM BELOW at a concentration:
a coarse mesh at a bridge fillet reads an OPTIMISTIC safety factor. A single-mesh peak von Mises is
therefore a statement about that mesh, and a safety factor computed from it inherits the same
status while looking like a property of the design.

The plan for this codebase deliberately does NOT answer that with P2 elements up front. It answers
it by MEASURING: solve at two or three refinements, publish the residual, and quote the answer as
an interval. If the interval is too wide to decide with, that is the evidence that buys P2 -- sized
by a number rather than by anticipation. The time axis inherits the same discipline: a march at one
`dt` is a statement about that `dt`, and nothing else.

WHAT "CONVERGING" MEANS HERE, and why it is asserted rather than assumed. Successive differences
must SHRINK. A sequence whose differences grow is not converging on anything, and its last value
is not closer to the truth than its first -- extrapolating a band from it would manufacture a
confidence the run does not support. `ConvergenceBand.converging` is that check, and
`residual` is meaningless without it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ['ConvergenceBand', 'convergence_band']


@dataclass(frozen=True)
class ConvergenceBand:
    """One quantity solved at several refinements, and the interval that implies.

    Attributes:
        refinements: the multipliers the quantity was solved at, ascending.
        values: the quantity at each, in the same order. The LAST is the finest and the best
            estimate; the band is built around it.

    """

    refinements: tuple[int, ...]
    values: tuple[float, ...]

    @property
    def finest(self) -> float:
        """The value at the finest refinement -- the estimate the band is centred on."""
        return self.values[-1]

    @property
    def residual(self) -> float:
        """``|finest - previous| / |finest|`` -- the relative change the last refinement bought.

        A LOWER BOUND on the discretization error, never the error itself: the true answer is
        further out than the last step moved. Quoted as such rather than dressed up as an error
        estimate, because a Richardson extrapolation needs an observed order and this needs three
        points to even check that the order is stable.
        """
        return abs(self.values[-1] - self.values[-2]) / abs(self.values[-1])

    @property
    def converging(self) -> bool:
        """Do successive absolute differences SHRINK?

        With two values there is one difference and nothing to compare it to, so this is ``True``
        vacuously -- which is exactly why `band` refuses to widen a two-point run into a claim it
        cannot support. Three points is the first honest answer.
        """
        diffs = np.abs(np.diff(self.values))
        return bool(np.all(np.diff(diffs) < 0.0)) if diffs.size > 1 else True

    def band(self) -> tuple[float, float]:
        """``(finest - residual*|finest|, finest + residual*|finest|)`` -- the interval to quote.

        SYMMETRIC, deliberately, even though P1 at a concentration converges from below and the
        truth is therefore expected on the HIGH side. Biasing the interval upward would encode an
        expectation about where the answer is; the residual only measures how far the last step
        moved, and the direction of the remaining error is an argument, not a measurement.

        Raises:
            ValueError: the sequence is not converging, so there is no band to quote.

        """
        if not self.converging:
            msg = (
                f'the differences between successive refinements are GROWING '
                f'({np.abs(np.diff(self.values)).tolist()}), so this sequence is not converging '
                f'and its finest value is not its best one. A band around it would be a '
                f'confidence interval on a divergent run.'
            )
            raise ValueError(msg)
        half = self.residual * abs(self.finest)
        return (self.finest - half, self.finest + half)

    def __str__(self) -> str:
        lo, hi = self.band()
        return (
            f'{self.finest:.6g} (band {lo:.6g} - {hi:.6g}, residual {self.residual:.2%} '
            f'over refinements {list(self.refinements)})'
        )


def convergence_band(
    solve_at: Callable[[int], float],
    refinements: Sequence[int] = (1, 2),
) -> ConvergenceBand:
    """Evaluate ``solve_at`` at each refinement and package the result.

    ``solve_at(refine)`` returns the scalar being converged -- a peak von Mises, a hoop stress, a
    radial growth, a marched winding current. Passing a CALLABLE rather than a case keeps this
    module free of the meshing, the march and the manifest: the same band applies to any quantity
    anyone can solve twice.

    Args:
        solve_at: the quantity as a function of the refinement multiplier.
        refinements: ascending, distinct, at least two. Two gives a residual; three is the first
            that can say whether the sequence is converging at all.

    Raises:
        ValueError: fewer than two refinements, or they are not strictly ascending positive
            integers, or a solve returned a non-finite value.

    """
    refinements = tuple(int(r) for r in refinements)
    if len(refinements) < 2:
        msg = (
            f'a band needs at least two refinements; got {list(refinements)}. One refinement gives a '
            f'number with no statement about itself, which is the situation this exists to end.'
        )
        raise ValueError(msg)
    if any(r < 1 for r in refinements) or any(b <= a for a, b in itertools.pairwise(refinements)):
        msg = (
            f'refinements must be strictly ascending positive integers; got {list(refinements)}. '
            f'The order is load-bearing -- the LAST is taken as the finest and the band is built '
            f'around it.'
        )
        raise ValueError(msg)

    values = []
    for r in refinements:
        value = float(solve_at(r))
        if not np.isfinite(value):
            msg = f'the solve at refinement {r} returned {value!r}; a band cannot be built on it.'
            raise ValueError(msg)
        values.append(value)
    return ConvergenceBand(refinements=refinements, values=tuple(values))
