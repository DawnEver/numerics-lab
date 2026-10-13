"""Index-two algebraic state authority, re-solved jointly with the actual rates.

A coordinate whose rate appears in no equation (a zero mass column) and whose
jacobian column every left-null storage row annihilates (the column lies in
the range of the mass) is an INDEX-TWO algebraic variable: no undifferentiated
equation fixes it, only the differentiated hidden constraint does. A solid
body's voltage under a zero-net-current row is the physical instance. After a
discrete step its value is the integrator's, not the one the hidden
constraint demands, so a frozen-state rate solve over every original row is
inconsistent. Its value is therefore STATE authority, solved with the rates:

    [ 0           Y F_u' ] [du_S]   [ -Y F_t          ]
    [ F_u[:, S]   M      ] [ v  ] = [ -(F + F_u dx)   ]

Its rate is free and never claimed. One dependent net-current mass row per
body leaves the square pivot set (``sparse_initial_rate``'s matching), and
every original row and every differentiated hidden row remains a witness at
the same bar.
"""

import numpy as np
import scipy.sparse as sp

from motronics.gauss.linear.arithmetic import certified_sparse_product
from motronics.gauss.linear.initial_rate import sparse_initial_rate

__all__ = ['index_two_columns', 'index_two_frozen_rate']


def index_two_columns(mass, jacobian, algebraic_rows):
    """Zero-mass columns whose actual jacobian column lies in the range of the mass."""
    mass, jacobian = sp.csr_matrix(mass), sp.csc_matrix(jacobian)
    mass.eliminate_zeros()
    jacobian.eliminate_zeros()
    absent = np.asarray(mass.getnnz(axis=0)).ravel() == 0
    candidates = np.flatnonzero(absent & (np.diff(jacobian.indptr) > 0))
    if not len(candidates) or algebraic_rows.shape[0] == 0:
        return np.array([], dtype=int)
    seen = sp.csc_matrix(certified_sparse_product(algebraic_rows, jacobian[:, candidates]))
    seen.eliminate_zeros()
    return candidates[np.diff(seen.indptr) == 0]


def index_two_frozen_rate(algebraic, hidden, mass, jacobian, hidden_forcing, forcing, *, tolerance, allowance):
    """Solve ``[hidden; mass] v = [hidden_forcing; forcing]``, freeing index-two states if refused.

    ``algebraic`` is the left-null storage basis ``Y``; ``hidden`` is the
    certified ``Y F_u'``. Returns ``(columns, state_step, rate, rate_unique)``.
    The frozen state is tried first: where it already satisfies the hidden
    constraint (a consistent start, or a body whose uniform mode the
    stiffness does not see, so its voltage is a gauge of its uniform rate)
    the result is the plain frozen-state solve, bit for bit. Only a refused
    frozen state takes its index-two coordinates as unknowns.
    """
    mass = sp.csr_matrix(mass)
    rhs = np.concatenate((hidden_forcing, forcing))
    bound = np.concatenate((np.zeros(hidden.shape[0]), allowance))
    try:
        rate, unique = sparse_initial_rate(
            sp.vstack((hidden, mass), format='csr'), rhs, tolerance=tolerance, residual_allowance=bound
        )
    except ValueError as frozen:
        columns = index_two_columns(mass, jacobian, algebraic)
        if not len(columns):
            raise
        matrix = sp.bmat(
            [
                [sp.csr_matrix((hidden.shape[0], len(columns))), hidden],
                [sp.csr_matrix(jacobian)[:, columns], mass],
            ],
            format='csr',
        )
        try:
            solution, unique = sparse_initial_rate(matrix, rhs, tolerance=tolerance, residual_allowance=bound)
        except ValueError as error:
            raise error from frozen
        if not unique[: len(columns)].all():
            msg = 'index-two algebraic state lacks unique hidden-constraint authority'
            raise ValueError(msg) from frozen
        return columns, solution[: len(columns)], solution[len(columns) :], unique[len(columns) :]
    return np.array([], dtype=int), np.zeros(0), rate, unique
