"""Construction provenance never replaces charge or differentiated equations."""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.contracts.runtime.ports import ResidualAssemblyEvaluation
from motronics.gauss.formulation import ResidualAssemblyBound
from motronics.gauss.linear.storage_border import StorageRows
from motronics.gauss.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair


@pytest.mark.parametrize('witness', ['charge', 'derivative'])
def test_certified_top_construction_bound_cannot_excuse_an_independent_witness(witness):
    mass = sp.diags([1.0, 0.0], format='csr')
    jacobian = sp.csr_matrix([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match='contradict|compatible|constraint|charge'):
        finite_storage_actual_rate_initial_pair(
            np.zeros(2),
            residual_at=lambda state, rate: (
                mass @ rate + jacobian @ state - np.array([0.0, 1e-30 if witness == 'charge' else 0.0])
            ),
            jacobian_at=lambda _state, _rate: jacobian,
            mass_at=lambda _state, _rate: mass,
            storage_at=lambda state: mass @ state,
            partial_time_at=lambda _state, _rate: np.array([0.0, 1e-30 if witness == 'derivative' else 0.0]),
            storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]])),
            stored_columns_at=lambda _state, _rate, _storage: np.array([0]),
            rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
            residual_evaluation_at=lambda state, rate: ResidualAssemblyEvaluation(
                mass @ rate + jacobian @ state - np.array([0.0, 1e-30 if witness == 'charge' else 0.0]),
                ResidualAssemblyBound(np.ones(2), np.ones(2, dtype=bool)),
            ),
        )


def test_non_cancelling_tiny_dynamic_contradiction_is_not_a_roundoff_floor():
    mass = sp.csr_matrix([[1.0, 0.0], [1.0, 0.0]])
    jacobian = sp.csr_matrix([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match='contradict|compatible|constraint|charge'):
        finite_storage_actual_rate_initial_pair(
            np.zeros(2),
            residual_at=lambda state, rate: mass @ rate + jacobian @ state - np.array([0.0, 1e-30]),
            jacobian_at=lambda _state, _rate: jacobian,
            mass_at=lambda _state, _rate: mass,
            storage_at=lambda state: mass @ state,
            partial_time_at=lambda _state, _rate: np.zeros(2),
            storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.csr_matrix([[-1.0, 1.0]])),
            stored_columns_at=lambda _state, _rate, _storage: np.array([0]),
            rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
            residual_evaluation_at=lambda state, rate: ResidualAssemblyEvaluation(
                mass @ rate + jacobian @ state - np.array([0.0, 1e-30]),
                ResidualAssemblyBound(np.array([0.0, 1e-45]), np.ones(2, dtype=bool)),
            ),
        )
