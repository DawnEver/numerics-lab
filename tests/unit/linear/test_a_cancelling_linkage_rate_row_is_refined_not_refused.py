"""A joint border answer carries the Schur formation error; the actual rows refine it away.

THE MEASURED CASE (motronics, 2026-11-07): ``transformer_3ph_distributed``, a magnetostatic field in an
isolated star. A linkage-rate row ``lambda' - b^T a' = 0`` sums 236 field-rate terms of 4.5e-9 that
cancel to the linkage rate. The border answer (``lambda'``) and the field rates reconstructed from it
met that row 3.2e-17 apart -- 1.6x its bar -- and the initial pair was refused: "joint stored state
contradicts actual original rows after exact homogeneous anchors". One refinement pass of the same
elimination on the actual residual took it to 8e-25.

The system below is that shape: a field block ``K a = B i`` eliminated (selected), the linkage rate
and current retained (remaining), the border answer perturbed by a formation-size error.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from numerics_lab.linear.initial_storage_border import _refined_joint_correction, _row_bars

FIELD = 240
TOLERANCE = 1e-10


def _system():
    stiffness = sp.diags([-np.ones(FIELD - 1), 2.5 * np.ones(FIELD), -np.ones(FIELD - 1)], [-1, 0, 1], format='csr')
    coupling_vector = np.linspace(0.5, 1.5, FIELD) * 1e-2
    # Columns: a [FIELD], lambda, i. Rows: field (selected), linkage, current drive (remaining).
    field = sp.hstack((stiffness, sp.csr_matrix((FIELD, 1)), sp.csr_matrix(-coupling_vector[:, None])))
    linkage = sp.hstack((sp.csr_matrix(-coupling_vector[None, :]), sp.csr_matrix([[1.0, 0.0]])))
    drive = sp.csr_matrix(([1.0], ([0], [FIELD + 1])), shape=(1, FIELD + 2))
    matrix = sp.vstack((field, linkage, drive), format='csr')
    rhs = np.zeros(FIELD + 2)
    rhs[-1] = 4.5e-9
    exact = splu(matrix.tocsc()).solve(rhs)
    return matrix, rhs, exact


def _refine(matrix, rhs, correction, fixed):
    selected_rows, selected_columns = np.arange(FIELD), np.arange(FIELD)
    remaining_rows, remaining_columns = np.array([FIELD, FIELD + 1]), np.array([FIELD, FIELD + 1])
    factor = splu(matrix[selected_rows][:, selected_columns].tocsc())
    coupling = matrix[selected_rows][:, remaining_columns]
    left = matrix[remaining_rows][:, selected_columns]
    response = factor.solve(coupling.toarray())
    schur = np.asarray(matrix[remaining_rows][:, remaining_columns].toarray() - left @ response)
    rows = np.arange(matrix.shape[0])
    return _refined_joint_correction(
        matrix,
        rhs,
        correction,
        original_rows=matrix.shape[0],
        selected=(selected_rows, selected_columns, np.zeros(FIELD), factor.solve),
        remaining=(remaining_rows, remaining_columns, left, schur),
        coupling=coupling,
        fixed=fixed,
        bars=lambda trial: _row_bars(matrix, rows, trial, rhs, np.zeros_like(rhs), TOLERANCE),
    )


def _worst(matrix, rhs, trial):
    rows = np.arange(matrix.shape[0])
    bars = _row_bars(matrix, rows, trial, rhs, np.zeros_like(rhs), TOLERANCE)
    return np.max(abs(matrix @ trial - rhs) / bars)


def test_a_formation_error_in_the_border_answer_is_refined_onto_the_actual_rows() -> None:
    matrix, rhs, exact = _system()
    perturbed = exact.copy()
    perturbed[FIELD] *= 1 + 1e-8  # the linkage rate as the formed border returned it
    assert _worst(matrix, rhs, perturbed) > 1  # the planted control: the row refuses this answer
    refined = _refine(matrix, rhs, perturbed, np.zeros(FIELD + 2, dtype=bool))
    assert _worst(matrix, rhs, refined) <= 1
    np.testing.assert_allclose(refined, exact, rtol=1e-12, atol=1e-30)  # volt-seconds per second


def test_refinement_holds_proven_coordinates_and_keeps_an_admitted_answer() -> None:
    matrix, rhs, exact = _system()
    assert _refine(matrix, rhs, exact, np.zeros(FIELD + 2, dtype=bool)) is exact
    perturbed = exact.copy()
    perturbed[FIELD] *= 1 + 1e-8
    fixed = np.zeros(FIELD + 2, dtype=bool)
    fixed[:3] = True
    refined = _refine(matrix, rhs, perturbed, fixed)
    assert np.array_equal(refined[:3], perturbed[:3])
