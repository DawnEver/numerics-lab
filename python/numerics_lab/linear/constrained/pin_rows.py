"""LIL-matrix Dirichlet row pinning -- pure algebra, no physics, no mesh.

Consolidated out of FOUR independent copies (MANUAL-20260728-442 follow-up, hand-copied-guard-drift
audit): ``hamilton.discretizations.hcurl.backend``, ``hamilton.physics.electrostatic.backend``,
``hamilton.physics.structural.backend`` and ``hamilton.physics.thermal.backend`` each carried a
private ``_pin_rows(mat, pinned_dofs)`` with the identical docstring -- and had already drifted:
the hcurl copy cast the row index too (``mat.rows[int(dof)] = ...``) where the other three indexed
``mat.rows[dof]`` directly, a difference nothing caught because nothing compared the four. This
function never referenced an element, a material, a chart or the jacobian triplet shape
:func:`~numerics_lab.linear.constrained.fixed_values.apply_dirichlet_triplets` pins (COO, value-keyed, with a
bias) -- it only replaces rows of a LIL matrix by the identity row, which makes it a ``gauss``
citizen by the dependency arrow, same as that sibling module.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ['pin_rows']


def pin_rows(mat, pinned_dofs: NDArray[np.intp]):
    """Replace each pinned row by the identity row (zero row, unit diagonal) on a LIL matrix."""
    for dof in pinned_dofs:
        dof = int(dof)
        mat.rows[dof] = [dof]
        mat.data[dof] = [1.0]
    return mat
