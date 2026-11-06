"""Two conducting bodies that SHARE boundary nodes leave their voltages algebraic, as one body does.

Golden: the joint border handed to ``_sparse_joint_border`` by a turning voltage-driven ring whose magnet is
split into two adjacent conducting segments (motronics
``test_a_turning_rotor_with_touching_conducting_bodies_admits_its_pair``). State coordinates 96, 97 are the two
body voltages; their rates enter the hidden lift rows only through the difference ``rate(V1) - rate(V2)`` with
coefficients that cancel to within their computed entry error. Rank discovery therefore leaves
``rate(V1) + rate(V2)`` free, and its solve picks up voltage STATE components of rounding size (6e-20). A body
voltage is fixed by its net-current row and the rate: the free direction is a RATE direction, never a state one.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear.initial_storage_border import _sparse_joint_border

_GOLDEN = Path(__file__).parents[1] / 'goldens' / 'touching_bodies_joint_border.npz'
_VOLTAGES = [96, 97]


def _border():
    data = np.load(_GOLDEN)

    def sparse(name):
        return sp.csr_matrix(
            (data[f'{name}_data'], data[f'{name}_indices'], data[f'{name}_indptr']), shape=tuple(data[f'{name}_shape'])
        )

    return {
        'matrix': sparse('matrix'),
        'rhs': data['rhs'],
        'mass': sparse('mass'),
        'algebraic': sparse('algebraic'),
        'stored_rows': data['stored_rows'],
        'indices': data['indices'],
        'residual_allowance': data['residual_allowance'],
        'state_directions': sparse('state_directions'),
        'row_scale': data['row_scale'],
        'tolerance': float(data['tolerance']),
    }


def test_touching_body_voltages_are_fixed_states_with_free_rates():
    border = _border()
    matrix, rhs = border.pop('matrix'), border.pop('rhs')
    mass, algebraic, stored, indices = (border.pop(key) for key in ('mass', 'algebraic', 'stored_rows', 'indices'))
    correction, rate_directions, _ = _sparse_joint_border(matrix, rhs, mass, algebraic, stored, indices, **border)
    assert np.isfinite(correction).all()
    rate_directions = sp.csr_matrix(rate_directions).toarray()
    free_rates = np.flatnonzero(np.any(rate_directions != 0, axis=1))
    assert set(_VOLTAGES) <= set(free_rates.tolist()), 'each body voltage rate stays non-unique'
    # The voltage-rate freedom lifts into the other rates only through the same cancellation; the field rates are
    # left unproven (conservative), never moved by more than rounding of the voltage-rate direction.
    others = free_rates[free_rates < min(_VOLTAGES)]  # the field rates; the circuit rows follow the voltages
    assert np.max(abs(rate_directions[others])) < 1e-15 * np.max(abs(rate_directions[_VOLTAGES]))
