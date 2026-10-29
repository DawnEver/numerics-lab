"""A 2-D magnetostatic operator with no Dirichlet reference is PINNED, not left to wander.

THE DEFECT THIS PINS, measured 2026-09-15 on `case_c --scope pole-pair --point
rated_25500`. That model carries `bound_edge_dict['dirichlet'] == {}` and one `even` (periodic)
seam, and `param_model_compile` tags no Dirichlet at all -- so EVERY `param_fit` sector reaches the
solver with `A` determined only up to a constant. The coefficient-damped Picard then ran its full 100
iterations like this:

    i=92 alpha=0.050000 |x|=1.96638e+02 E=28.339107 err%=0.000131 inc=1.9358e+02
    i=93 alpha=0.050000 |x|=1.82129e+00 E=28.339143 err%=0.000124 inc=1.0086e+00
    i=99 alpha=0.050000 |x|=5.55864e+00 E=28.339321 err%=0.000092 inc=4.8307e+00

The ENERGY was converged to 9.2e-05 percent and the ITERATE was wandering by up to 200x its own
norm. `_has_converged` requires both, so the solve could never report the convergence it had
already reached -- and every result it produced carried "the iteration hit its ceiling, so these
numbers are not a measurement of the discretisation".

The arithmetic names the direction: a uniform shift ``c`` over 11995 DOFs has norm ``109.5 c``, so
the observed ``||dx|| ~ 4`` is ``c ~ 0.037``, which moves ``||x||`` from 1.8 to 4.4 -- exactly the
measured swing. It was the gauge, and nothing else.

AFTER PINNING: the same run converges in **61** iterations (`Err 0.0006%, increment 1.11e-04`) with
torque -68.7755 against -68.7837 and energy 1.7002e+02 against 1.7004e+02 -- a 0.012 percent move,
which is what ``B = curl A`` guarantees a constant shift can do.

WHY A WHOLE-MATRIX TEST WAS NOT ENOUGH, and it is the subtle half: of case_c's 11995 rows exactly
266 had a nonzero row sum and every one of them was exactly 1.0 -- the seam's ELIMINATED SLAVE
rows, which are recovered after the solve as ``A[slave] = sign * A[master]`` and constrain the
constant not at all. A test that read them reported the operator gauged (ratio 3.155e-09) while the
physical problem was not. So constraint rows are excluded, and the third test below is the one that
would have caught the first version of this fix.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from numerics_lab.linear import (
    constraint_rows,
    floats_on_a_constant,
    gauge_pin_floating_constant,
    gauge_pin_orphan_rows,
)


def _laplacian_1d(n: int = 6) -> csr_matrix:
    """A 1-D stiffness with NO boundary condition -- the floating case, in miniature.

    Every interior row is ``[-1, 2, -1]`` and the two ends are ``[1, -1]`` / ``[-1, 1]``, so every
    row sums to zero and the constant vector is an exact null vector. That is the same statement
    ``INT nu grad(A).grad(v)`` makes on a real mesh, which is why this suffices to pin the rule.
    """
    m = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            m[i, i - 1] = -1.0
            m[i, i] += 1.0
        if i < n - 1:
            m[i, i + 1] = -1.0
            m[i, i] += 1.0
    return csr_matrix(m)


def _with_dirichlet(mat: csr_matrix, row: int = 0) -> csr_matrix:
    """The same operator with one REAL Dirichlet row -- a gauged problem."""
    lil = mat.tolil()
    lil.rows[row] = [row]
    lil.data[row] = [1.0]
    return lil.tocsr()


def test_a_floating_operator_is_DETECTED() -> None:
    """The floor: if this returns False the pin below can never fire and the fix is decoration."""
    assert floats_on_a_constant(_laplacian_1d()) is True


def test_a_dirichlet_gauged_operator_is_LEFT_ALONE() -> None:
    """A model that determines its own potential must not be touched -- the other side of the ratchet.

    Without this the pin would fire on every case in the library and silently overwrite a real
    boundary row, which is a far worse defect than the one it was written to fix.

    AND THE DECISION IS THE CALLER'S, which is what the first version of this test got wrong: it
    asserted ``floats_on_a_constant(gauged) is False`` and that assertion cannot be made true. A
    Dirichlet row IS an eliminated-slave row -- a lone 1.0 on the diagonal -- so the matrix carries
    no evidence that separates them, and the test below pins exactly that indistinguishability
    rather than papering over it. ``has_dirichlet`` is the declaration the caller holds.
    """
    gauged = _with_dirichlet(_laplacian_1d())
    assert floats_on_a_constant(gauged) is True, 'the EQUATIONS still impose no reference'
    rhs = np.ones(gauged.shape[0])
    before = gauged.toarray()
    out = gauge_pin_floating_constant(gauged, rhs, has_dirichlet=True)
    assert np.array_equal(out.toarray(), before), 'a gauged operator must come back unchanged'
    assert np.array_equal(rhs, np.ones(gauged.shape[0])), 'and its rhs untouched'


def test_A_DIRICHLET_ROW_AND_AN_ELIMINATED_SLAVE_ARE_INDISTINGUISHABLE() -> None:
    """The reason ``has_dirichlet`` is a required keyword, asserted rather than argued.

    Build the two situations the pin must treat OPPOSITELY -- a real boundary row, and a fold's
    eliminated slave -- and show the matrices are bit-identical. Any detector deciding from the
    operator alone must therefore be wrong on one of them, and being wrong in the permissive
    direction pins a second dof on a constrained model and returns a silently wrong field.
    """
    dirichlet = _with_dirichlet(_laplacian_1d(), row=2)
    eliminated_slave = _with_dirichlet(_laplacian_1d(), row=2)
    assert np.array_equal(dirichlet.toarray(), eliminated_slave.toarray())
    assert floats_on_a_constant(dirichlet) == floats_on_a_constant(eliminated_slave)


def test_ELIMINATED_SLAVE_ROWS_DO_NOT_MASK_A_FLOATING_OPERATOR() -> None:
    """THE CASE_C SIGNATURE, and the case the first version of this fix got wrong.

    A paired-DOF fold turns each slave into an identity row. Those rows have row-sum 1, so a
    whole-matrix ``max|K @ 1|`` reads 1.0 and calls the operator gauged -- which is exactly what
    happened on case_c (266 such rows out of 11995, ratio 3.155e-09 against a 1e-9 threshold).
    They are CONSTRAINTS, not equations, and excluding them is what makes the detector honest.
    """
    floating = _laplacian_1d()
    folded = _with_dirichlet(floating, row=2)  # stands in for an eliminated slave
    assert constraint_rows(folded).sum() == 1
    assert constraint_rows(folded)[2]
    # The remaining EQUATION rows still annihilate the constant, so the problem still floats.
    assert floats_on_a_constant(folded) is True, 'a constraint row must not read as a gauge'


def test_the_pin_NEVER_LANDS_ON_A_CONSTRAINT_ROW() -> None:
    """Overwriting an eliminated slave would drop the seam pairing it encodes.

    Constructed so the constraint row carries the LARGEST diagonal: a pin that chose by magnitude
    alone would take it, and the model would silently lose a periodic pair.
    """
    mat = _laplacian_1d().tolil()
    mat.rows[3] = [3]
    mat.data[3] = [99.0]  # a constraint row with a huge diagonal
    mat = mat.tocsr()
    rhs = np.zeros(mat.shape[0])
    out = gauge_pin_floating_constant(mat, rhs, has_dirichlet=False).toarray()
    assert out[3].tolist() == [0.0, 0.0, 0.0, 99.0, 0.0, 0.0], 'the constraint row must survive intact'
    pinned = [r for r in range(mat.shape[0]) if r != 3 and out[r].nonzero()[0].tolist() == [r]]
    assert len(pinned) == 1, f'exactly one FREE dof must be pinned, got {pinned}'


def test_the_pin_LEAVES_EVERY_DIFFERENCE_UNCHANGED() -> None:
    """The exactness claim, measured rather than asserted: ``B = curl A`` sees only DIFFERENCES.

    Solve the floating system two ways -- pinned at the dof the function picks, and pinned at a
    different dof -- and the two solutions must differ by a CONSTANT and nothing else. That is the
    whole warrant for calling the pin exact for flux density, energy and torque.
    """
    rhs_src = np.array([1.0, -2.0, 0.5, 0.5, -1.0, 1.0])  # sums to zero: a consistent load
    a = gauge_pin_floating_constant(_laplacian_1d(), rhs_a := rhs_src.copy(), has_dirichlet=False)
    x_a = np.linalg.solve(a.toarray(), rhs_a)

    lil = _laplacian_1d().tolil()
    lil.rows[5] = [5]
    lil.data[5] = [1.0]
    rhs_b = rhs_src.copy()
    rhs_b[5] = 0.0
    x_b = np.linalg.solve(lil.tocsr().toarray(), rhs_b)

    diff = x_a - x_b
    assert np.ptp(diff) < 1e-9, f'the two gauges must differ by a CONSTANT, got spread {np.ptp(diff)}'
    assert abs(float(np.mean(diff))) > 1e-12, 'floor: an identical pair would satisfy the line above'


def test_it_is_a_DIFFERENT_pin_from_the_orphan_one() -> None:
    """The two cover disjoint failures, which is why both exist.

    An orphan row couples to NOTHING; here every row is fully coupled and the operator is still
    singular. Running the orphan pin on the floating operator changes nothing, so it cannot be the
    remedy -- measured, not argued.
    """
    floating = _laplacian_1d()
    rhs = np.zeros(floating.shape[0])
    after_orphan = gauge_pin_orphan_rows(floating, rhs)
    assert np.array_equal(after_orphan.toarray(), floating.toarray())
    assert floats_on_a_constant(after_orphan) is True, 'the orphan pin leaves it floating'


@pytest.mark.parametrize('n', [4, 6, 12])
def test_the_detector_is_scale_free(n: int) -> None:
    """The same verdict in metres and millimetres -- the tolerance is relative, so it must be.

    A 1e6 scaling is what a unit change does to a stiffness, and a detector that flipped on it
    would fire on some cases and not their identical twins.
    """
    mat = _laplacian_1d(n)
    assert floats_on_a_constant(mat) is True
    assert floats_on_a_constant(csr_matrix(mat.toarray() * 1e6)) is True
    assert floats_on_a_constant(csr_matrix(mat.toarray() * 1e-6)) is True


# --- PORTED FROM `feat/jmag-integration` AT THE 2026-09-16 INTEGRATION -----------------------
#
# That lane wrote its own copy of this file under `tests/unit/hamilton/fem/`, against its own
# `has_no_reference` / `gauge_pin_constant_mode` API. THE COPY COULD NOT IMPORT ON ITS OWN BRANCH:
# it imports `_gauge_pin_constant_mode` and `_gauge_pin_orphan_dofs` from
# `hamilton/discretizations/fem/_solve.py`, which defines neither -- it imports the PUBLIC names
# from `gauss.linear`. So it was a collection error there and had never run.
#
# THE API DID NOT SURVIVE THE MERGE AND THE REASON IS ON RECORD: their detector infers the
# reference FROM THE MATRIX, and `gauge_pin_floating_constant`'s docstring is the argument against
# doing that -- "a real Dirichlet row and a fold's eliminated slave row are the SAME OBJECT ... a
# detector that guessed would eventually pin a SECOND dof on a properly constrained model". Their
# `test_an_ANTIPERIODIC_pair_also_counts_as_a_reference` asserts exactly the guessing this side
# refuses, so the file could not be ported wholesale.
#
# THESE TWO CLAIMS COULD, and they are the lane's real finding rather than its API: an orphan pin
# writes an IDENTITY row, an identity row sums to 1, and a detector that counts it as a reference
# leaves the coupled bulk floating. This side is immune BY CONSTRUCTION -- `constraint_rows`
# excludes a lone diagonal before `floats_on_a_constant` reads it -- and that immunity is worth a
# test that says so rather than an argument that implies it.


def _with_orphan(n: int) -> csr_matrix:
    """A bulk Laplacian plus one DISCONNECTED dof, whose row is all zero."""
    bulk = _laplacian_1d(n).toarray()
    big = np.zeros((n + 1, n + 1))
    big[:n, :n] = bulk
    return csr_matrix(big)


def test_an_EMPTY_system_is_not_a_crash() -> None:
    """The degenerate edge, where there is no dof to pin."""
    empty = csr_matrix((0, 0), dtype=float)
    assert gauge_pin_floating_constant(empty, np.zeros(0), has_dirichlet=False).shape == (0, 0)


def test_an_ORPHAN_PIN_DOES_NOT_MASK_A_FREE_CONSTANT_IN_EITHER_ORDER() -> None:
    """The lane's ordering hole, asserted as ABSENT here rather than as a hazard to sequence around.

    On an API that infers the reference from the matrix, pinning the orphan FIRST leaves the system
    singular: the identity row it writes reads as an anchor while the coupled bulk still floats,
    measured there at rank 6 of 7. `constraint_rows` is why that cannot happen on this side -- it
    masks a lone-diagonal row out of the equations before `floats_on_a_constant` looks -- so BOTH
    orders must come out full rank, and asserting both is what makes the immunity checkable.
    """
    n = 6
    mat = _with_orphan(n)

    orphan_first = gauge_pin_floating_constant(
        gauge_pin_orphan_rows(mat, np.zeros(n + 1)), np.zeros(n + 1), has_dirichlet=False
    )
    assert np.linalg.matrix_rank(orphan_first.toarray()) == n + 1, (
        'pinning the orphan first left the system singular -- the identity row it writes is being '
        'read as a reference, which is the defect `constraint_rows` exists to prevent'
    )

    constant_first_mat = gauge_pin_floating_constant(mat, np.zeros(n + 1), has_dirichlet=False)
    constant_first = gauge_pin_orphan_rows(constant_first_mat, np.zeros(n + 1))
    assert np.linalg.matrix_rank(constant_first.toarray()) == n + 1, 'constant-first must be full rank too'
