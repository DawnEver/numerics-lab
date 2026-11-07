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
the same bar. ``S`` is a set of DIRECTIONS, not columns: two bodies that share
a field node have only their common voltage in the range of the mass.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import null_space, qr
from scipy.sparse.csgraph import connected_components

from numerics_lab.linear.arithmetic import certified_sparse_product, sparse_product_bound
from numerics_lab.linear.border_rank import border_affine_space
from numerics_lab.linear.initial_rate import sparse_initial_rate

__all__ = ['certified_null_directions', 'index_two_directions', 'index_two_frozen_rate']


def index_two_directions(mass, jacobian, algebraic_rows, *, sparse_output=False):
    """Index-two directions over the zero-mass coordinates: ``(columns, basis)``.

    A candidate is a zero-mass column whose jacobian column lies wholly in
    mass-bearing rows (a column on an undifferentiated row is fixed by that
    row). A combination ``d`` of candidates is index two when ``J[:, c] d``
    lies in the range of the mass, i.e. the certified ``Y J[:, c] d`` is zero.
    An isolated body's column is seen by no hidden row and is its own
    direction; two bodies sharing a field node are each seen, with equal and
    opposite weight, so only their common voltage (a uniform field over both)
    is index two. Candidates couple only through the hidden rows that see
    them, so the null space is taken per connected group. ``basis`` has one
    column per direction, supported on ``columns``.
    """
    mass, jacobian = sp.csr_matrix(mass), sp.csc_matrix(jacobian)
    mass.eliminate_zeros()
    jacobian.eliminate_zeros()
    absent = np.asarray(mass.getnnz(axis=0)).ravel() == 0
    stateless = np.asarray(mass.getnnz(axis=1)).ravel() == 0
    on_state_rows = np.asarray(abs(jacobian[stateless]).sum(axis=0)).ravel() > 0
    candidates = np.flatnonzero(absent & (np.diff(jacobian.indptr) > 0) & ~on_state_rows)
    if not len(candidates) or algebraic_rows.shape[0] == 0:
        return np.array([], dtype=int), np.zeros((0, 0))
    basis = certified_null_directions(algebraic_rows, jacobian[:, candidates], sparse_output=sparse_output)
    used = np.asarray((basis != 0).getnnz(axis=1)).ravel() != 0 if sparse_output else np.any(basis != 0, axis=1)
    return candidates[used], basis[used]


def _column_components(pattern):
    """Actual bipartite support components, without overflowing shared-row counts."""
    support = sp.csc_matrix(pattern, dtype=bool)
    support.eliminate_zeros()
    if support.nnz:
        graph = sp.bmat([[None, support], [support.T, None]], format='csr', dtype=bool)
        _, labels = connected_components(graph, directed=False)
        labels = labels[support.shape[0] :]
    else:
        labels = np.arange(support.shape[1])
    groups = {}
    for column, label in enumerate(labels):
        groups.setdefault(int(label), []).append(column)
    for columns in groups.values():
        members = np.asarray(columns, dtype=int)
        yield members, np.unique(support[:, members].indices)


def _sparse_block_columns(size, blocks):
    """Embed bounded local bases with no dense global state-by-range allocation."""
    values, rows, columns = [], [], []
    count = 0
    for members, block in blocks:
        local_rows, local_columns = np.nonzero(block)
        values.append(block[local_rows, local_columns])
        rows.append(members[local_rows])
        columns.append(local_columns + count)
        count += block.shape[1]
    if not values:
        return sp.csr_matrix((size, count))
    return sp.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(size, count))


def certified_null_directions(left, right, *, sparse_output=False):
    """Directions ``d`` with ``left @ right @ d`` certified zero, dense by default or CSR.

    A combination is null when its image is no larger than the summation bound of the product it
    is read from -- the certificate ``certified_sparse_product`` applies to a single column, so a
    pair whose entries cancel only to rounding (two bodies sharing a field node, each seen by the
    same hidden rows with equal and opposite weight) is null exactly when each column alone would
    be. Columns couple only through the certified rows that see them, so the null space is taken
    per connected group; a column no row sees is its own direction.
    """
    size = right.shape[1]
    if left.shape[0] == 0:
        return sp.eye(size, format='csr') if sparse_output else np.eye(size)
    seen = sp.csc_matrix(certified_sparse_product(left, right))
    seen.eliminate_zeros()
    product, bound = (sp.csc_matrix(matrix) for matrix in sparse_product_bound(left, right))

    def blocks():
        for members, rows in _column_components(seen):
            if len(members) > 512 or (sparse_output and len(rows) * len(members) > 512**2):
                msg = 'state range requires bounded actual connected component rank work'
                raise NotImplementedError(msg)
            yield members, local_basis(members, rows)

    def local_basis(members, rows):
        if not len(rows):
            null = np.eye(len(members))
        else:
            dense, limit = product[:, members][rows].toarray(), bound[:, members][rows].toarray()
            _, candidates = border_affine_space(dense, np.zeros(len(rows)), entry_errors=limit)
            # Rank discovery's directions are candidates, not physical witnesses.
            # Anchor their independent coordinates and re-solve each as a particular
            # of the ORIGINAL rows, whose solver reconstructs and refines those rows.
            # QR selects coordinates of an already issued basis; it decides no physical rank.
            if candidates.shape[1]:
                anchors = qr(candidates.T, pivoting=True)[2][: candidates.shape[1]]
                equations = np.vstack((dense, np.eye(len(members))[anchors]))
                errors = np.vstack((limit, np.zeros((len(anchors), len(members)))))
                candidates = np.column_stack([
                    border_affine_space(
                        equations,
                        np.concatenate((np.zeros(len(rows)), direction[anchors])),
                        mandatory_rows=np.arange(len(rows), len(equations)),
                        entry_errors=errors,
                    )[0]
                    for direction in candidates.T
                ])
            null = np.array([
                direction for direction in candidates.T if np.all(abs(dense @ direction) <= limit @ abs(direction))
            ]).reshape(-1, len(members))
        return null.T

    basis = _sparse_block_columns(size, blocks())
    return basis if sparse_output else basis.toarray()


def index_two_frozen_rate(
    algebraic, hidden, mass, jacobian, hidden_forcing, forcing, *, tolerance, allowance, differentiated_allowance=None
):
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
    hidden_bound = (
        np.zeros(hidden.shape[0]) if differentiated_allowance is None else np.asarray(differentiated_allowance)
    )
    if (
        hidden_bound.shape != (hidden.shape[0],)
        or hidden_bound.dtype.kind not in 'fiu'
        or not np.isfinite(hidden_bound).all()
        or np.any(hidden_bound < 0)
    ):
        msg = 'differentiated allowance requires finite nonnegative original hidden-row bounds'
        raise ValueError(msg)
    bound = np.concatenate((hidden_bound, allowance))
    try:
        rate, unique = sparse_initial_rate(
            sp.vstack((hidden, mass), format='csr'), rhs, tolerance=tolerance, residual_allowance=bound
        )
    except ValueError as frozen:
        columns, basis = index_two_directions(mass, jacobian, algebraic, sparse_output=mass.shape[0] > 512)
        if not len(columns):
            raise
        count, size = basis.shape[1], mass.shape[1]
        # The rate along an index-two direction is free (it multiplies nothing), so the rate
        # unknowns on these columns are only its complement: two touching bodies' voltages
        # keep the rate of their difference, which the hidden rows fix, and drop the free one.
        kept = np.setdiff1d(np.arange(size), columns)
        if sp.issparse(basis):

            def complements():
                for members, rows in _column_components(basis.T):
                    if len(members) > 512 or len(rows) * len(members) > 512**2:
                        msg = 'index-two rate complement requires bounded actual component work'
                        raise NotImplementedError(msg)
                    yield members, null_space(basis.T[rows][:, members].toarray())

            complement = _sparse_block_columns(len(columns), complements())
            selectors = sp.eye(size, format='csr')
            rate_map = sp.hstack((selectors[:, kept], selectors[:, columns] @ complement), format='csr')
        else:
            complement = null_space(basis.T)
            rate_map = sp.hstack(
                (
                    sp.csr_matrix((np.ones(len(kept)), (kept, np.arange(len(kept)))), shape=(size, len(kept))),
                    sp.csr_matrix((size, complement.shape[1])),
                ),
                format='lil',
            )
            rate_map[columns, len(kept) :] = complement
            rate_map = rate_map.tocsr()
        matrix = sp.bmat(
            [
                [sp.csr_matrix((hidden.shape[0], count)), sp.csr_matrix(hidden) @ rate_map],
                [sp.csr_matrix(sp.csc_matrix(jacobian)[:, columns] @ basis), mass @ rate_map],
            ],
            format='csr',
        )
        try:
            solution, unique = sparse_initial_rate(matrix, rhs, tolerance=tolerance, residual_allowance=bound)
        except ValueError as error:
            raise error from frozen
        if not unique[:count].all():
            msg = 'index-two algebraic state lacks unique hidden-constraint authority'
            raise ValueError(msg) from frozen
        rate_unique = np.zeros(size, dtype=bool)
        rate_unique[kept] = unique[count : count + len(kept)]
        return columns, basis @ solution[:count], rate_map @ solution[count:], rate_unique
    return np.array([], dtype=int), np.zeros(0), rate, unique
