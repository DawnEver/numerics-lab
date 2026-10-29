"""An actual disconnected storage row cannot invent a rate in another block."""

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear.storage_border import storage_border_rows


def test_charge_dependency_does_not_mix_disconnected_inductor_or_field_linkage_rows():
    mass = sp.csr_matrix([
        [1e-9, 0, -1e-9, 0],
        [-1e-9, 0, 1e-9, 0],
        [0, 0, 0, 0.25],
        [0, 0.0005, 0, 0],
    ])
    rows = storage_border_rows(mass, bulk_indices=np.array([], dtype=int), border_rows=np.arange(4))
    assert len(rows.stored_rows) == 3
    assert rows.algebraic_rows.shape == (1, 4)
    assert rows.algebraic_rows[:, 2:].nnz == 0
    defect = rows.algebraic_rows @ mass
    defect.eliminate_zeros()
    assert defect.nnz == 0
    # A voltage appearing only in the disconnected field row remains a truly
    # free first-order rate, to be resolved by actual higher constraints later.
    projected = rows.algebraic_rows @ sp.eye(4, format='csr')
    assert projected[:, 3].nnz == 0
