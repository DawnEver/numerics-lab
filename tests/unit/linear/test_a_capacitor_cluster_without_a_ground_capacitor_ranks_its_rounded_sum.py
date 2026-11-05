"""A capacitor cluster with no capacitor to ground stores one charge fewer than it has nodes.

THE MEASURED CASE (motronics, 2026-11-02): a level-1 JFET with RD and RS whose drain and source pins
are both on ground. Its gate node ``a`` reaches the internal drain/source nodes ``d'``/``s'`` only
through Cgd and Cgs, and those nodes reach ground only through RD/RS. The stamped mass is

    [[Cgd + Cgs, -Cgd, -Cgs], [-Cgd, Cgd, 0], [-Cgs, 0, Cgs]]

whose row sum ``(1, 1, 1)`` is a physical left null vector -- but the diagonal is the ROUNDED sum
``fl(Cgd + Cgs)``, one rounding off ``Cgd + Cgs``. Ranked in exact rationals the three rows read
independent (residual 8.1e-28 against entries of 2.4e-11), so three "stored" rows were declared on a
rank-two mass and the initial pair refused the circuit: "initial state/rate equations lack a
nonsingular sparse physical constraint subsystem".

The values below are the stamped bits of that circuit at t = 0 (gate swing at -10.5 V).
"""

from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear.initial_sparse import sparse_initial_pair
from numerics_lab.linear.storage_border import storage_border_rows

_CGS, _CGD = float.fromhex('0x1.2894229d902dbp-36'), float.fromhex('0x1.da869dc8e6bbbp-38')
MASS = sp.csr_matrix(
    np.array([
        [_CGD + _CGS, -_CGD, -_CGS, 0.0],
        [-_CGD, _CGD, 0.0, 0.0],
        [-_CGS, 0.0, _CGS, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
)
_G = float.fromhex('0x1.197998145b833p-40')
JACOBIAN = sp.csr_matrix(
    np.array([
        [2 * _G, -_G, -_G, 1.0],
        [-_G, float.fromhex('0x1.0000000000466p+2'), 0.0, 0.0],
        [-_G, 0.0, float.fromhex('0x1.4000000000466p+2'), 0.0],
        [-1.0, 0.0, 0.0, 0.0],
    ])
)
RESIDUAL = np.array([2.0**-88, 0.0, 0.0, 0.0])
INITIAL = np.array([
    -10.5,
    float.fromhex('-0x1.7223bc7452341p-39'),
    float.fromhex('-0x1.281c96c37506cp-39'),
    float.fromhex('0x1.7223bc74523e5p-36'),
])
PARTIAL_TIME = np.array([0.0, 0.0, 0.0, float.fromhex('0x1.22d375c3432d6p-31')])


def test_the_stamped_diagonal_is_the_rounded_sum_of_its_two_capacitors() -> None:
    row_sum = sum(Fraction(value) for value in MASS.toarray()[0])
    assert row_sum != 0  # the planted control: exact rationals read this row as independent
    assert abs(float(row_sum)) <= np.finfo(float).eps * MASS[0, 0]


def test_the_cluster_ranks_as_two_stored_charges_and_its_sum_is_algebraic() -> None:
    rows = storage_border_rows(MASS, bulk_indices=np.array([], dtype=int), border_rows=np.arange(4))
    assert len(rows.stored_rows) == 2
    assert rows.algebraic_rows.shape == (2, 4)
    defect = abs(rows.algebraic_rows @ MASS).max()
    assert defect <= 64 * np.finfo(float).eps * abs(MASS).max()


def test_the_grounded_jfet_reaches_its_initial_pair() -> None:
    rows = storage_border_rows(MASS, bulk_indices=np.array([], dtype=int), border_rows=np.arange(4))
    pair = sparse_initial_pair(
        JACOBIAN,
        RESIDUAL,
        MASS,
        INITIAL,
        algebraic_rows=rows.algebraic_rows,
        stored_rows=rows.stored_rows,
        partial_time=PARTIAL_TIME,
        stored_increment=np.zeros(len(rows.stored_rows)),
    )
    defect = RESIDUAL + JACOBIAN @ (pair.state - INITIAL) + MASS @ pair.rate
    scale = abs(JACOBIAN) @ abs(pair.state) + abs(MASS) @ abs(pair.rate) + abs(RESIDUAL)
    assert np.all(abs(defect) <= 1e-10 * scale.max())
    # The source row (-v_a + V(t) = 0) fixes the gate rate at the source's own dV/dt.
    assert pair.rate_is_unique[0]
    assert np.isclose(pair.rate[0], PARTIAL_TIME[3], rtol=1e-12)
