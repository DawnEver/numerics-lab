"""A solid body's hidden net-current residual is shared with its SIBLING bodies, at the trial it holds.

case_a 90 deg admission at t=0.0011921
(.claude/memory/2026/10/15/a-solid-bodys-hidden-row-is-shared-with-its-siblings.md): a body's
net-current row ``G u + s . sigma a_dot`` loses its body voltage once the field rows are eliminated
(each field rate answers ``-C u``), so its Schur row holds only the phase current -- a hidden state
constraint the border cannot move. Its residual belongs on the eliminated field rows, whose bars hold
it. Sharing only the row currently over its bar moved the residual through the field row the two
bodies share onto the sibling body and back; the share now keeps every row it has shared and judges
each row at the trial it holds.

PLANTED: five field rows, two bodies sharing the middle (stiffer, 100x) one, one phase current. The
data are consistent except for ``delta`` on body 1's forcing, which is 46x its own bar but well within
the bars of the field rows it couples to -- solvable by construction. Measured on the parent of
c4c85020f1: body 1 left at 46.1x its bar; here every row is inside its own bar (worst 0.006x).
"""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.initial_storage_border import _row_bars, _sparse_joint_border

pytestmark = pytest.mark.unit

TOLERANCE = 1e-10


def _two_bodies_sharing_a_field_row(delta):
    m = 5
    n = m + 3
    bodies, phase = [m, m + 1], m + 2
    sigma = np.full(m, 1e8)
    sigma[m // 2] *= 100.0  # the shared field row
    coupling = np.zeros((m, 2))  # field row k's body-voltage coefficients
    coupling[: m // 2, 0] = 1.0
    coupling[m // 2 + 1 :, 1] = 1.0
    coupling[m // 2] = 0.5
    share = 1e-6 * (coupling.T > 0)  # body j's field-rate coefficients
    current = np.full(m, -1.0)
    current[m // 2] = 4.0 / 100.0 + 1e-3  # the body rows' phase-current coefficients nearly cancel
    stiffness = 1e5
    jacobian = np.zeros((n, n))
    mass = np.zeros((n, n))
    for k in range(m):
        mass[k, k] = sigma[k]
        jacobian[k, bodies] = coupling[k] * sigma[k]
        jacobian[k, phase] = current[k] * sigma[k]
        jacobian[k, k] = stiffness * sigma[k]
    for j, row in enumerate(bodies):
        mass[row, :m] = share[j] * sigma
        jacobian[row, bodies] = (share * sigma)[j] @ coupling  # the body voltage cancels on elimination
    jacobian[phase, phase] = 1.0
    jacobian, mass = sp.csr_matrix(jacobian), sp.csr_matrix(mass)
    stored_rows = np.arange(m)
    voltage_rows = [sp.csr_matrix(([1.0, 1.0], ([0, 0], [u, phase])), shape=(1, 2 * n)) for u in bodies]
    matrix = sp.vstack((sp.bmat([[jacobian, mass], [mass[stored_rows], None]]), *voltage_rows), format='csr')
    rhs = np.zeros(matrix.shape[0])
    rhs[:m] = stiffness * sigma  # the stored field (a = 1) answers its own forcing: a_dot is small
    rhs[n : n + m] = sigma
    rhs[bodies] = -(share * sigma) @ current * 2.0
    rhs[phase] = 2.0
    rhs[-2:] = 3.0
    rhs[bodies[0]] += delta  # PLANTED: the hidden constraint's inconsistency, on body 1 alone
    return matrix, mass, sp.csr_matrix((0, n)), stored_rows, rhs, m


def test_a_body_rows_residual_is_shared_with_its_siblings_until_every_row_holds_its_bar():
    matrix, mass, algebraic, stored_rows, rhs, m = _two_bodies_sharing_a_field_row(1e-4)
    row_scale = np.zeros(len(rhs))
    correction, _directions, _admitted = _sparse_joint_border(
        matrix, rhs, mass, algebraic, stored_rows, np.arange(m), row_scale=row_scale, tolerance=TOLERANCE
    )
    defect = abs(matrix @ correction - rhs)
    bars = _row_bars(matrix, np.arange(len(rhs)), correction, rhs, row_scale, TOLERANCE)
    assert np.all(defect[bars == 0] == 0), 'a row with no bar is met exactly'
    soft = bars > 0
    assert np.all(defect[soft] <= bars[soft]), f'rows over their bar: {(defect[soft] / bars[soft]).max():.3g}x'
