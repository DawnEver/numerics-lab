"""The direct-solve residual guard is REACHED from the production paths, not merely defined.

# What was actually wrong, measured rather than assumed

The guard is NOT missing. ``numerics_lab.linear.solve_linear_system`` has carried both halves
since the seam landed -- the ``MatrixRankWarning`` belt in ``_direct`` and the residual suspenders
in the caller -- and the production CLI already refuses the system the 2026-09-04 migration note
named:

(``_DIRECT_RESIDUAL_RTOL`` has since become the cheap TRIGGER rather than the verdict, because a
fixed bar on the relative residual is a disguised CONDITION test -- see
``test_the_direct_guard_separates_ill_conditioning_from_failure.py``. Nothing in THIS file changes:
every system it plants is refused or accepted for the same reason it always was.)

    downstream run cases/case_i --select fea,tri --set mesh.mesh_size="6mm"
    -> failed: direct solve residual ||A x - b|| / ||b|| = 1.608e+01 exceeds 1e-06

So "SuperLU ships finite garbage and the production Python path ships it" is WITHDRAWN as a claim
about the FEA seam. What was true is narrower and worse: **twelve other modules bound
``scipy.sparse.linalg.spsolve`` into their own namespace and never consulted the seam at all**, so
the same finite-garbage return shipped unchecked from the transient march, the harmonic phasor
solve, the DWR adjoint, the hcurl gauged block and the circuit MNA. The guard was PRESENT and
UNREACHED, which is `taste.md`'s dominant defect wearing the opposite face: not a declaration that
lies, but an enforcement nothing calls.

# The fixture is the REAL system, and it refutes the belt

``goldens/case_i_m6_singular_157.npz`` is the order-157 system the production CLI assembles
for ``case_i`` at ``--set mesh.mesh_size="6mm"``, captured through the seam itself. Measured on it:

    order 157, nnz 837, condition estimate 1.49e17
    spsolve returns a FINITE vector, relative residual 17.89, and emits NO WARNING AT ALL

That last clause is the sharp one, and it is why a synthetic plant was not good enough. An
EXACTLY singular matrix (a duplicated row) makes SuperLU hit a zero pivot, emit
``MatrixRankWarning`` and return NaNs -- caught by the belt and by ``_check_finite``, so a test
built on one is green without the residual guard existing at all. This system is NUMERICALLY
singular instead: SuperLU finds a pivot everywhere, reports nothing, and returns garbage that is
finite. **The residual is the only instrument that sees it.** The first draft of this file used
the duplicated-row plant and its own floor test refused it; the floor is kept below for that
reason.

# Why every test here has two sides

`taste.md`: "A scope claim needs its own test -- one that PLANTS the thing in the region and calls
the REAL guard", and "a ratchet has two sides". A guard that refused EVERY system would satisfy
the raises-half of each pair identically, so each planted refusal is paired with a well-conditioned
twin that must still return the right answer through the same call.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import LinearSolveError, direct_relative_residual, solve_linear_system
from numerics_lab.time_integrators.march import _default_solve

GOLDEN = Path(__file__).parent / 'goldens' / 'case_i_m6_singular_157.npz'


def _inductor_m6_singular():
    """The production order-157 system, as the CLI assembled it. See the module docstring."""
    data = np.load(GOLDEN)
    shape = tuple(int(v) for v in data['shape'])
    return sp.csr_matrix((data['data'], data['indices'], data['indptr']), shape=shape), data['rhs']


def _well_conditioned(n: int = 5):
    """A 1-D Laplacian with a diagonal boost -- SPD, sparse, residual ~ machine epsilon."""
    dense = np.diag(np.full(n, 4.0)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)
    return sp.csr_matrix(dense), np.arange(1.0, n + 1.0)


def test_the_plant_is_finite_garbage_and_silent_so_only_the_residual_can_catch_it():
    """THE FLOOR under every refusal below, and it refuted this file's first draft.

    Three properties are asserted, because losing any one of them would leave the rest of the file
    green for a reason that has nothing to do with the residual guard:

    * FINITE -- otherwise ``_check_finite`` raises first;
    * SILENT -- otherwise the ``MatrixRankWarning`` belt raises first;
    * a residual far above the bar -- otherwise there is nothing to catch.
    """
    matrix, rhs = _inductor_m6_singular()
    assert matrix.shape == (157, 157)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        x = sp.linalg.spsolve(sp.csc_matrix(matrix), rhs)
    assert np.all(np.isfinite(x)), 'the plant returned NaNs; _check_finite would catch it, not the residual'
    assert not [w for w in caught if issubclass(w.category, sp.linalg.MatrixRankWarning)], (
        'SuperLU warned on this system; the BELT would catch it and the suspenders would be untested'
    )
    assert direct_relative_residual(matrix, rhs, x) > 1.0


def test_the_production_order_157_system_is_refused_naming_its_residual_and_order():
    """The refusal is INFORMATIVE: the measured residual, the system order, and a remedy.

    The residual VALUE is not pinned, only its scale. The garbage a singular factorisation returns
    depends on the pivot ordering, so the CLI reads 1.608e+01 through CSR and 1.789e+01 through
    CSC -- pinning either would be pinning an artefact of the storage format. What is stable, and
    what the guard is judged on, is that it is seven orders of magnitude over the bar.
    """
    matrix, rhs = _inductor_m6_singular()
    with pytest.raises(LinearSolveError) as excinfo:
        solve_linear_system(matrix, rhs)
    message = str(excinfo.value)
    assert 'residual' in message
    assert 'n=157' in message, message
    assert 'gauge' in message, 'the refusal must say what the caller can DO about it'


def test_a_well_conditioned_system_still_passes_through_the_same_call():
    """The other side of the ratchet: a guard that refused everything would pass the test above."""
    matrix, rhs = _well_conditioned()
    x = solve_linear_system(matrix, rhs)
    assert direct_relative_residual(matrix, rhs, x) < 1e-12
    np.testing.assert_allclose(matrix @ x, rhs, rtol=1e-10)


def test_a_complex_system_reaches_the_guard_at_all():
    """The time-harmonic operator is COMPLEX, and the guard used to cast its residual to float.

    That cast is why ``studies/harmonic`` could not consult the seam: routing a complex phasor
    system through it raised a dtype error instead of a verdict. Both halves are planted -- a
    complex well-posed system must still be solved, and a complex singular one must be refused.
    """
    well = sp.csr_matrix(np.array([[2.0 + 1.0j, -1.0], [-1.0, 2.0 - 1.0j]]))
    rhs = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    x = solve_linear_system(well, rhs)
    np.testing.assert_allclose(well @ x, rhs, rtol=1e-10)
    assert direct_relative_residual(well, rhs, x) < 1e-12

    singular = sp.csr_matrix(np.array([[1.0 + 1.0j, 2.0 + 2.0j], [2.0 + 2.0j, 4.0 + 4.0j]]))
    with pytest.raises(LinearSolveError):
        solve_linear_system(singular, np.array([1.0 + 0.0j, 5.0 + 0.0j]))


def test_the_transient_integrator_seam_reaches_the_guard():
    """PLANTED IN THE REGION: ``_default_solve`` is the per-step solve of the time march.

    Measured on the baseline tree, ``case_i --study transient`` drives it **312 times** in
    one run and never consulted the guard once. It is called here directly -- the REAL production
    function, not a re-implementation -- with both a singular and a well-conditioned system.
    """
    matrix, rhs = _inductor_m6_singular()
    with pytest.raises(LinearSolveError, match='residual'):
        _default_solve(matrix, rhs)

    matrix, rhs = _well_conditioned()
    np.testing.assert_allclose(matrix @ _default_solve(matrix, rhs), rhs, rtol=1e-10)


def test_a_zero_right_hand_side_is_unjudged_rather_than_silently_green():
    """``b == 0`` makes ``||A x - b|| / ||b||`` a 0/0, so the statistic must decline to answer.

    Every null-space vector solves ``A x = 0`` exactly, so reporting 0.0 there would be a pass the
    arithmetic never earned -- ``taste.md``'s vacuous-green shape. ``None`` is UNJUDGED, and the
    caller must not read it as a verdict.
    """
    matrix, _ = _well_conditioned()
    assert direct_relative_residual(matrix, np.zeros(5), np.zeros(5)) is None
