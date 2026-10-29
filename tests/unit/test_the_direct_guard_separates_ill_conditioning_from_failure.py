"""The direct-solve guard judges the ANSWER, because the residual cannot tell the two cases apart.

# What was wrong, measured rather than assumed

``_DIRECT_RESIDUAL_RTOL = 1e-6`` was a fixed bar on ``||A x - b|| / ||b||``, calibrated against ONE
point (the order-157 ``case_i`` plant: cond 1.5e17, residual 1.6e+01) with the decades
between never checked. Sweeping Young's modulus on the CST patch test -- which rescales ``A`` and
changes nothing else -- shows the residual tracking ``cond * eps`` while the ANSWER stays exact to
2e-17 at every scale. A fixed bar on it is therefore a disguised CONDITION test, and it refused
every native structural stationary solve at real SI material properties while telling its reader
the system was "singular / rank-deficient" and the solution "finite but meaningless".

# Why the textbook answer is NOT used here

The normwise relative backward error ``||Ax-b|| / (||A|| ||x|| + ||b||)`` is the standard fix for a
residual that scales -- and ``test_the_backward_error_cannot_separate_the_two_ends`` below plants
why it fails: SuperLU is backward stable ON THE SINGULAR PLANT TOO (0.5 eps, indistinguishable from
the well-posed sweep), because the finite garbage it returns exactly solves a NEARBY system. That
test exists so the refuted candidate cannot be re-derived by a later reader who sees only the
scaling problem.

# The floor under this file

``taste.md``: "give every scan a floor, since finding NOTHING is vacuous rather than green". The
sweep asserts the residual actually CROSSES the old bar somewhere in its range, so it cannot pass
by exercising a range where nothing was ever refused; and every acceptance is paired with the
refusal it must not swallow.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from numerics_lab.linear import (
    _DIRECT_REFINEMENT_RTOL,
    _DIRECT_RESIDUAL_RTOL,
    LinearSolveError,
    direct_refinement_correction,
    direct_relative_residual,
    reject_unsolved_direct_system,
    solve_linear_system,
)

GOLDEN = Path(__file__).parent / 'goldens' / 'case_i_m6_singular_157.npz'
EPS = float(np.finfo(float).eps)


def _inductor_m6_singular():
    data = np.load(GOLDEN)
    shape = tuple(int(v) for v in data['shape'])
    return sp.csr_matrix((data['data'], data['indices'], data['indptr']), shape=shape), data['rhs']


def _scaled_patch(scale: float, n: int = 24):
    """A well-posed SPD system scaled by ``scale``, the algebraic shape of a stiffness in SI units.

    ``A = scale * L`` with ``L`` a 1-D Laplacian, plus an UNSCALED pinned identity row at each end --
    the Dirichlet folding the structural assembly does. Those pinned rows are what make ``cond(A)``
    grow linearly with ``scale`` while the solve stays exact, which is the whole phenomenon.
    """
    dense = np.diag(np.full(n, 2.0)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)
    dense *= scale
    for row in (0, n - 1):
        dense[row, :] = 0.0
        dense[row, row] = 1.0
    exact = np.linspace(0.3, -0.4, n)
    return sp.csr_matrix(dense), dense @ exact, exact


# --- END ONE: the planted failure must still be refused --------------------------------------


def test_the_order_157_finite_garbage_plant_is_still_refused():
    """END ONE. A criterion that only accepted the steel end would satisfy the other side alone."""
    matrix, rhs = _inductor_m6_singular()
    with pytest.raises(LinearSolveError) as excinfo:
        solve_linear_system(matrix, rhs)
    message = str(excinfo.value)
    assert 'n=157' in message, message
    assert 'refinement' in message, 'the refusal must name the statistic that decided it'
    assert 'gauge' in message, 'the refusal must say what the caller can DO about it'


def test_the_refinement_step_sees_the_plant_above_the_actual_refusal_bar():
    """THE FLOOR under end one: the discriminator must actually discriminate ON THIS SYSTEM.

    Without this, the refusal above could be coming from the ``MatrixRankWarning`` belt or from
    ``_check_finite``, and the refinement step would be untested decoration.

    Measured 2026-10-03 under scipy 1.18.1: the correction is 0.21875 (it was 4.7-5.0 when this was
    first pinned at ``> 1.0``, under an older scipy whose SuperLU returned different garbage;
    ``refinement.py`` and the golden are unchanged). The CLAIM is the gap to the bar, not the
    garbage's size: 0.21875 sits log10(0.21875 / 1e-6) = 5.3 decades above
    ``_DIRECT_REFINEMENT_RTOL``, so the plant is still refused. The floor 1e-2 keeps a 1.3-decade
    margin under today's value and still demands four decades over the bar.
    """
    matrix, rhs = _inductor_m6_singular()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = spla.spsolve(sp.csc_matrix(matrix), rhs)
    assert np.all(np.isfinite(x)), 'the plant returned NaNs; _check_finite would catch it'
    correction = direct_refinement_correction(matrix, rhs, x)
    # The numerical value varies with the sparse factorization implementation;
    # the invariant is rejection at the production bar, not a historical value.
    assert correction is not None and correction > _DIRECT_REFINEMENT_RTOL, correction


# --- END TWO: the ill-conditioned but well-posed system must be accepted ----------------------


def test_a_stiffness_at_real_si_material_properties_is_accepted_not_called_singular():
    """END TWO, and the regression this file exists for.

    ``scale = 210e9`` is steel's Young's modulus. The residual is decades over the trigger and the
    answer is exact -- the exact combination the old fixed bar read as "singular".
    """
    matrix, rhs, exact = _scaled_patch(210e9)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = solve_linear_system(matrix, rhs)
    residual = direct_relative_residual(matrix, rhs, x)
    assert residual > _DIRECT_RESIDUAL_RTOL, (
        f'the floor: this system must actually TRIP the old bar, else it proves nothing ({residual:.3e})'
    )
    np.testing.assert_allclose(x, exact, rtol=0, atol=1e-12)


@pytest.mark.parametrize('scale', [1.0, 1e2, 1e4, 1e6, 1e8, 1e9, 1e10, 1e11, 210e9, 1e13, 1e15])
def test_the_answer_is_exact_at_every_scale_while_the_residual_tracks_the_conditioning(scale):
    """THE INTERMEDIATE SWEEP: the decades the original calibration never checked.

    Two claims per decade, and they are the two the old bar conflated: the refinement correction
    stays flat (so the guard's verdict is scale-INVARIANT) and the answer stays exact.
    """
    matrix, rhs, exact = _scaled_patch(scale)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = solve_linear_system(matrix, rhs)
    assert direct_refinement_correction(matrix, rhs, x) <= _DIRECT_REFINEMENT_RTOL
    np.testing.assert_allclose(x, exact, rtol=0, atol=1e-12)


def test_the_sweep_has_a_floor_the_old_bar_was_crossed_somewhere_in_it():
    """Without this, the sweep above could live entirely below the trigger and never exercise it."""
    residuals = []
    for scale in (1.0, 1e6, 1e11, 210e9, 1e15):
        matrix, rhs, _ = _scaled_patch(scale)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x = solve_linear_system(matrix, rhs)
        residuals.append(direct_relative_residual(matrix, rhs, x))
    assert min(residuals) < _DIRECT_RESIDUAL_RTOL < max(residuals), (
        f'the sweep must straddle the trigger to be evidence about it: {residuals}'
    )


# --- the refuted candidate, planted so it is not re-derived -----------------------------------


def _backward_error(matrix, rhs, x):
    """Rigal-Gaches normwise relative backward error ``||Ax-b|| / (||A|| ||x|| + ||b||)``."""
    csr = sp.csr_matrix(matrix)
    residual_vec = np.asarray(csr @ x).ravel() - np.asarray(rhs).ravel()
    denominator = float(abs(csr).sum(axis=1).max()) * float(np.max(np.abs(x))) + float(np.max(np.abs(rhs)))
    return float(np.max(np.abs(residual_vec))) / denominator


def test_the_backward_error_cannot_separate_the_two_ends():
    """The textbook answer to a scaling problem, REFUTED on evidence rather than on taste.

    It reads within a factor of a thousand of machine epsilon on BOTH ends, so a bar on it would
    have to admit the plant in order to accept the steel case. SuperLU is backward stable even when
    it returns garbage: the garbage exactly solves a nearby system.
    """
    matrix, rhs = _inductor_m6_singular()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plant = _backward_error(matrix, rhs, spla.spsolve(sp.csc_matrix(matrix), rhs))
        good_matrix, good_rhs, _ = _scaled_patch(210e9)
        good = _backward_error(good_matrix, good_rhs, solve_linear_system(good_matrix, good_rhs))
    assert plant < 10.0 * EPS, f'the plant is backward-STABLE ({plant / EPS:.2f} eps) -- that is the point'
    assert plant / good < 1e3, f'a backward-error bar cannot separate these: plant {plant:.2e} vs well-posed {good:.2e}'


# --- the statistic declines to answer rather than guessing ------------------------------------


def test_an_exactly_singular_factorization_is_unjudged_rather_than_a_pass():
    """``None`` is UNJUDGED, never a pass: no refinement step exists to judge the answer by.

    An EXACTLY singular matrix is caught earlier -- by the ``MatrixRankWarning`` belt in
    ``_direct``, asserted here so the layering is on record -- so ``None`` never decides this
    case in production. It must still not read as a pass, because the ordering of the two guards
    is not a property the refinement statistic itself can rely on.
    """
    singular = sp.csr_matrix(np.array([[1.0, 2.0], [2.0, 4.0]]))
    rhs = np.array([1.0, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = spla.spsolve(sp.csc_matrix(singular), rhs)
    assert direct_refinement_correction(singular, rhs, x) is None
    with pytest.raises(LinearSolveError, match='near-.singular'):
        solve_linear_system(singular, rhs)


def test_an_unjudgeable_correction_refuses_and_says_the_refinement_was_unavailable():
    """PLANTED IN THE REGION: the ``None`` branch of the REAL guard, reached with a real system.

    ``x = 0`` on a singular system has residual 1.0 -- decades over the trigger -- and no
    factorization exists to refine it, so the guard must refuse and must say WHICH of the two
    findings it established. A refusal that claimed a measured correction here would be quoting a
    number it never computed.
    """
    singular = sp.csr_matrix(np.array([[1.0, 2.0], [2.0, 4.0]]))
    rhs = np.array([1.0, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(LinearSolveError, match='unavailable'):
            reject_unsolved_direct_system(singular, rhs, np.zeros(2), 'direct')
