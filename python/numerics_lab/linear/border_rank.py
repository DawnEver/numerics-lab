"""Small-border elimination with local roundoff bounds, not a global mixed-index SVD cutoff."""

from fractions import Fraction
from itertools import zip_longest
from math import gcd, isqrt

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching

from numerics_lab.linear.arithmetic import rounded_defect

__all__ = ['border_affine_space', 'homogeneous_zero_coordinates']


#: Primes below 2**31: a product of two residues stays inside int64.
_PRIMES = (2_147_483_647, 2_147_483_629)
#: Primes one kernel certificate may combine (~31 bits each). An exact dependence among float entries with
#: small rational weights -- the common deficiency -- reconstructs within a few; a larger one is left to the
#: fraction-free elimination, so the cap bounds the attempt's cost, never the verdict.
_CERTIFICATE_PRIMES = 8


def _integer_rows(block):
    """``block`` with each row scaled by a power of two to EXACT integers (Python ints): the same column rank."""
    out = []
    for row in block:
        values = [Fraction(float(value)) for value in row]
        shift = max((v.denominator.bit_length() - 1 for v in values if v), default=0)
        out.append([v.numerator << (shift - (v.denominator.bit_length() - 1)) for v in values])
    return out


def _is_prime(n):
    """Deterministic Miller-Rabin below 3.2e9 (bases 2, 3, 5, 7)."""
    d, s = n - 1, 0
    while not d & 1:
        d, s = d >> 1, s + 1
    for base in (2, 3, 5, 7):
        x = pow(base, d, n)
        if x in {1, n - 1}:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _primes():
    """Primes below 2**31, descending from the largest."""
    candidate = _PRIMES[0]
    while True:
        if _is_prime(candidate):
            yield candidate
        candidate -= 2


def _full_column_rank_mod(rows, prime):
    """Whether integer ``rows`` have full column rank over GF(``prime``), by elimination in int64."""
    m = np.array([[value % prime for value in row] for row in rows], dtype=np.int64)
    n_rows, n_cols = m.shape
    for column in range(n_cols):
        live = np.flatnonzero(m[column:, column]) + column
        if not live.size:
            return False
        pivot = int(live[0])
        m[[column, pivot]] = m[[pivot, column]]
        m[column] = m[column] * pow(int(m[column, column]), -1, prime) % prime
        below = np.flatnonzero(m[column + 1 :, column]) + column + 1
        if below.size:
            m[below] = (m[below] - m[below, column, None] * m[column]) % prime
    return n_rows >= n_cols


def _echelon_mod(rows, prime):
    """Reduced row echelon form of integer ``rows`` over GF(``prime``) in int64: (matrix, pivot columns)."""
    m = np.array([[value % prime for value in row] for row in rows], dtype=np.int64).reshape(len(rows), -1)
    pivots = []
    for column in range(m.shape[1]):
        rank = len(pivots)
        if rank == m.shape[0]:
            break
        live = np.flatnonzero(m[rank:, column]) + rank
        if not live.size:
            continue
        pivot = int(live[0])
        m[[rank, pivot]] = m[[pivot, rank]]
        m[rank] = m[rank] * pow(int(m[rank, column]), -1, prime) % prime
        others = np.flatnonzero(m[:, column])
        others = others[others != rank]
        if others.size:
            m[others] = (m[others] - m[others, column, None] * m[rank]) % prime
        pivots.append(column)
    return m, tuple(pivots)


def _rational(residue, modulus):
    """The fraction ``a / b`` with ``|a|, |b| <= sqrt(modulus / 2)`` congruent to ``residue``, else None."""
    bound = isqrt(modulus // 2)
    r0, r1, t0, t1 = modulus, residue % modulus, 0, 1
    while r1 > bound:
        quotient = r0 // r1
        r0, r1, t0, t1 = r1, r0 - quotient * r1, t1, t0 - quotient * t1
    if t1 == 0 or abs(t1) > bound or gcd(r1, abs(t1)) != 1:
        return None
    return Fraction(r1, t1)


def _annihilates(rows, vector):
    """Whether integer ``rows`` times the rational ``vector`` is EXACTLY zero, in integer arithmetic."""
    denominator = 1
    for value in vector:
        denominator = denominator * value.denominator // gcd(denominator, value.denominator)
    integers = [value.numerator * (denominator // value.denominator) for value in vector]
    support = [index for index, value in enumerate(integers) if value]
    return all(sum(row[index] * integers[index] for index in support) == 0 for row in rows)


def _kernel_certificate(rows, needed):
    """Per prime tried, whether integer ``rows`` EXACTLY annihilate ``needed`` independent vectors yet.

    The canonical kernel basis (one free coordinate one, the other free ones zero) is formed mod successive
    primes, combined by the Chinese remainder theorem and reconstructed as rationals. A reconstruction is only
    a CANDIDATE: it counts once ``rows @ x == 0`` holds exactly. Canonical vectors on distinct free coordinates
    are independent. Yields None per prime short of a certificate, then stops; True once certified.
    """
    n_cols = len(rows[0])
    pivots, residues, modulus = None, None, 1
    for _attempt, prime in zip(range(_CERTIFICATE_PRIMES), _primes(), strict=False):
        echelon, these = _echelon_mod(rows, prime)
        if pivots is not None and these != pivots:
            if len(these) <= len(pivots):
                continue  # a prime dividing a pivot minor reads a different echelon form: its residues differ
            residues, modulus = None, 1
        pivots = these
        pivot_set = set(pivots)
        free = [column for column in range(n_cols) if column not in pivot_set][:needed]
        if len(free) < needed:
            return
        vectors = np.zeros((needed, n_cols), dtype=object)
        for index, column in enumerate(free):
            vectors[index, column] = 1
            for position, pivot in enumerate(pivots):
                vectors[index, pivot] = int(-echelon[position, column]) % prime
        if residues is None:
            residues = vectors
        else:
            residues = residues + modulus * ((vectors - residues) * pow(modulus, -1, prime) % prime)
        modulus *= prime
        candidates = [[_rational(int(value), modulus) for value in vector] for vector in residues]
        if any(value is None for vector in candidates for value in vector):
            yield None
            continue
        if all(_annihilates(rows, vector) for vector in candidates):
            yield True
            return
        yield None


def _bareiss_full_column_rank(rows):
    """Exact full column rank by fraction-free (Bareiss) elimination: every division is exact in integers."""
    rows = [list(row) for row in rows]
    previous = 1
    for column in range(len(rows[0])):
        pivot = next((index for index in range(column, len(rows)) if rows[index][column]), None)
        if pivot is None:
            return False
        rows[column], rows[pivot] = rows[pivot], rows[column]
        head = rows[column]
        for index in range(column + 1, len(rows)):
            row, factor = rows[index], rows[index][column]
            rows[index] = [(head[column] * row[j] - factor * head[j]) // previous for j in range(len(row))]
        previous = head[column]
    return True


def _exact_full_column_rank(block):
    """Whether the actual entries of ``block`` have full column rank, by exact arithmetic.

    Full rank over a prime field CERTIFIES full rank over the rationals (a minor nonzero mod p is nonzero), so a
    full-rank block is decided in machine integers. A structural deficiency (no full bipartite matching) needs
    no arithmetic. Deficient under every prime tried, an exact dependence is the common case, and it is
    CERTIFIED by exact null vectors -- one right null vector, or ``m - n + 1`` independent left ones (the row
    rank is then below ``n``) -- each verified in integers (:func:`_kernel_certificate`). Only a block neither
    certifies (a prime dividing every maximal minor, or a dependence whose weights outgrow the certificate's
    primes) falls to fraction-free elimination: exact, as the rational elimination it replaces, without its
    gcds (that one ran over 33 min on a downstream bulk-storage border, 2026-10-22).
    """
    integers = _integer_rows(block)
    if any(_full_column_rank_mod(integers, prime) for prime in _PRIMES):
        return True
    n_rows, n_cols = block.shape
    if n_rows < n_cols:
        return False
    matching = maximum_bipartite_matching(sp.csr_matrix(block), perm_type='row')
    if np.count_nonzero(matching >= 0) < n_cols:
        return False
    transposed = [list(column) for column in zip(*integers, strict=True)]
    # Both sides in lockstep, one prime each: whichever dependence is small certifies first.
    for verdicts in zip_longest(_kernel_certificate(integers, 1), _kernel_certificate(transposed, n_rows - n_cols + 1)):
        if True in verdicts:
            return False
    return _bareiss_full_column_rank(integers)


def _homogeneous_blocks(original, rhs, remaining, consumed):
    """Components of unconsumed homogeneous rows over their remaining support, as (rows, columns)."""
    owner = {}

    def find(column):
        while owner.setdefault(column, column) != column:
            owner[column] = owner[owner[column]]
            column = owner[column]
        return column

    supports = {}
    for row in range(original.shape[0]):
        if row in consumed or rhs[row] != 0:
            continue
        columns = original.indices[original.indptr[row] : original.indptr[row + 1]]
        support = [int(column) for column in columns[remaining[columns]]]
        if support:
            supports[row] = support
            for column in support[1:]:
                owner[find(column)] = find(support[0])
    blocks = {}
    for row, support in supports.items():
        rows, columns = blocks.setdefault(find(support[0]), ([], set()))
        rows.append(row)
        columns.update(support)
    return [(rows, sorted(columns)) for rows, columns in blocks.values()]


def homogeneous_zero_coordinates(matrix, rhs, *, row_order=None, entry_errors=None):
    """Prove exact zero coordinates from original homogeneous support alone.

    A homogeneous row whose remaining support is ONE entry proves that coordinate zero -- if the entry is
    one. ``entry_errors`` bounds a COMPUTED matrix's entries (a Schur border's formation error), and an
    entry inside ``16 max(shape)`` of its bound (the eliminator's own reliability test) is a possible zero:
    the row may be ``0 = 0``, so it proves nothing and its column stays free.

    So does, jointly, any set of homogeneous rows whose remaining support they hold at full EXACT column
    rank -- the singleton is its one-by-one case. Left to the floating eliminator, such a block inherits the
    cancellation of the driven rows that share its columns: an eps-relative image of a driven coordinate
    (1.5e-33 against 2e-17) lands on coordinates the hidden constraints pin to zero. Exact rank is a proof
    only of exact entries: a COMPUTED block anchors where its full rank survives its entry errors -- its
    least singular value above the singleton's reliability factor times their norm.
    """
    original = sp.csr_matrix(matrix, copy=True)
    original.sum_duplicates()
    original.eliminate_zeros()
    rhs = np.asarray(rhs)
    if rhs.shape != (original.shape[0],) or not np.isfinite(rhs).all() or not np.isfinite(original.data).all():
        msg = 'homogeneous anchors require finite original equations'
        raise ValueError(msg)
    if entry_errors is None:
        reliable = None
    else:
        errors = np.asarray(entry_errors, dtype=float)
        if errors.shape != original.shape:
            msg = 'homogeneous anchors require entry error bounds aligned with the equations'
            raise ValueError(msg)
        rows_of = np.repeat(np.arange(original.shape[0]), np.diff(original.indptr))
        bound = 16 * max(original.shape)
        reliable = np.abs(original.data) > bound * errors[rows_of, original.indices]
    remaining = np.ones(original.shape[1], dtype=bool)
    consumed = set()
    order = range(original.shape[0]) if row_order is None else row_order
    while True:
        changed = False
        for row in order:
            if row in consumed or rhs[row] != 0:
                continue
            start, end = original.indptr[row], original.indptr[row + 1]
            columns = original.indices[start:end]
            live = remaining[columns]
            support = columns[live]
            if len(support) == 1 and (reliable is None or reliable[start:end][live][0]):
                remaining[support[0]] = False
                consumed.add(int(row))
                changed = True
        if changed:
            continue
        for rows, columns in _homogeneous_blocks(original, rhs, remaining, consumed):
            if not 1 < len(columns) <= min(len(rows), 512):
                continue
            block = original[rows][:, columns].toarray()
            block_errors = None if entry_errors is None else errors[np.ix_(rows, columns)]
            if block_errors is None or not block_errors.any():
                full_rank = _exact_full_column_rank(block)
            else:
                # Rank survives any perturbation smaller than the least singular value (Weyl), and
                # ||E||_2 <= ||E||_F; ``bound`` is the singleton's own reliability factor.
                full_rank = np.linalg.svd(block, compute_uv=False)[-1] > bound * np.linalg.norm(block_errors)
            if full_rank:
                remaining[columns] = False
                consumed.update(rows)
                changed = True
        if not changed:
            return remaining, consumed


def _direction_support(subsystem, forcing):
    """Conservative structural support of solves of an already independent subsystem.

    Match each equation to one unknown. Every other nonzero coefficient gives
    an edge from its unknown to that matched unknown. Propagate each forcing's
    support over these edges. Unreached variables form a closed homogeneous
    diagonal block; the selected subsystem's nonsingularity proves that block
    nonsingular too, so its solution is EXACT zero. Reachable cancellations
    prove nothing and remain numerical solves. This neither discovers rank nor
    asserts that a structurally matched but dependent matrix is invertible.
    """
    pattern = subsystem != 0
    matched = maximum_bipartite_matching(sp.csr_matrix(pattern), perm_type='column')
    if np.any(matched < 0):
        return np.ones((subsystem.shape[1], forcing.shape[1]), dtype=bool)
    outgoing = [matched[np.flatnonzero(pattern[:, column])] for column in range(subsystem.shape[1])]
    support = np.zeros((subsystem.shape[1], forcing.shape[1]), dtype=bool)
    for column in range(forcing.shape[1]):
        pending = matched[np.flatnonzero(forcing[:, column])].tolist()
        while pending:
            variable = pending.pop()
            if support[variable, column]:
                continue
            support[variable, column] = True
            pending.extend(outgoing[variable].tolist())
    return support


def border_affine_space(matrix, rhs, *, mandatory_rows=(), inherited_allowance=None, entry_errors=None):
    """Return a representative and explicit free directions of actual border equations.

    Re-equilibrate remaining rows after each elimination. A weak independent
    index constraint is not discarded just because another row has different
    units. Cancellation-only rows are removed using propagated arithmetic
    error bounds before any normalization can turn their noise into authority.
    ``entry_errors`` bounds a COMPUTED matrix's own entries (a Schur border's
    formation error); it seeds that propagation, so an entry inside its bound
    is never a pivot. Omitted, the entries are taken as exact.
    """
    matrix, rhs = np.asarray(matrix, dtype=float).copy(), np.asarray(rhs, dtype=float).copy()
    if (
        matrix.ndim != 2
        or not all(matrix.shape)
        or matrix.shape[1] > 512
        or rhs.shape != (matrix.shape[0],)
        or not np.isfinite(matrix).all()
        or not np.isfinite(rhs).all()
    ):
        msg = 'border affine equations require a finite bounded matrix and aligned forcing'
        raise ValueError(msg)
    rows, columns = matrix.shape
    allowance = np.zeros(rows) if inherited_allowance is None else np.asarray(inherited_allowance)
    if (
        allowance.shape != (rows,)
        or allowance.dtype.kind not in 'fiu'
        or any(isinstance(value, (bool, np.bool_)) for value in np.asarray(allowance, dtype=object).flat)
        or (
            inherited_allowance is not None
            and any(isinstance(value, (bool, np.bool_)) for value in np.asarray(inherited_allowance, dtype=object).flat)
        )
        or not np.isfinite(allowance).all()
        or np.any(allowance < 0)
    ):
        msg = 'inherited border allowance requires finite nonnegative actual row bounds'
        raise ValueError(msg)
    mandatory_rows = np.asarray(mandatory_rows)
    if (
        mandatory_rows.ndim != 1
        or (mandatory_rows.size and mandatory_rows.dtype.kind not in 'iu')
        or len(set(mandatory_rows.tolist())) != len(mandatory_rows)
        or np.any(mandatory_rows < 0)
        or np.any(mandatory_rows >= rows)
    ):
        msg = 'mandatory border constraints require unique actual equation rows'
        raise ValueError(msg)
    mandatory_rows = mandatory_rows.astype(int)
    seeded = np.zeros_like(matrix) if entry_errors is None else np.asarray(entry_errors, dtype=float)
    if seeded.shape != matrix.shape or not np.isfinite(seeded).all() or np.any(seeded < 0):
        msg = 'border entry error bounds require a finite nonnegative aligned matrix'
        raise ValueError(msg)
    # Original homogeneous singleton equations prove exact zeros. Propagate
    # only those proven columns, before driven-row arithmetic can contaminate
    # the representative or its free directions.
    ordered_rows = np.concatenate((mandatory_rows, np.setdiff1d(np.arange(rows), mandatory_rows)))
    remaining, consumed = homogeneous_zero_coordinates(matrix, rhs, row_order=ordered_rows, entry_errors=seeded)
    if consumed:
        support_rows = np.any(matrix[:, remaining] != 0, axis=1)
        eliminated_rows = (~support_rows) & np.any(matrix != 0, axis=1)
        if np.any(eliminated_rows & (abs(rhs) > allowance)):
            msg = 'joint stored state contradicts actual state/rate equations at exact homogeneous anchors'
            raise ValueError(msg)
        pending = [int(row) for row in mandatory_rows if row not in consumed]
        if any(not support_rows[row] for row in pending):
            msg = 'mandatory border constraints must retain independent actual equation authority'
            raise ValueError(msg)
        answer = np.zeros(columns)
        if not np.any(remaining):
            return answer, np.zeros((columns, 0))
        retained = np.flatnonzero(support_rows)
        if not retained.size:
            free = np.flatnonzero(remaining)
            directions = np.zeros((columns, len(free)))
            directions[free, np.arange(len(free))] = 1
            return answer, directions
        positions = {int(row): index for index, row in enumerate(retained)}
        reduced_answer, reduced_directions = border_affine_space(
            matrix[retained][:, remaining],
            rhs[retained],
            mandatory_rows=tuple(positions[row] for row in pending),
            inherited_allowance=allowance[retained],
            entry_errors=seeded[retained][:, remaining],
        )
        answer[remaining] = reduced_answer
        directions = np.zeros((columns, reduced_directions.shape[1]))
        directions[remaining] = reduced_directions
        return answer, directions
    order = np.concatenate((mandatory_rows, np.setdiff1d(np.arange(rows), mandatory_rows)))
    matrix, rhs = matrix[order], rhs[order]
    original_matrix, original_rhs = matrix.copy(), rhs.copy()
    row_ids = np.arange(rows)
    errors = seeded[order].copy()
    eps = np.finfo(float).eps
    bound = 16 * max(matrix.shape)
    pivots = []
    cursor = 0
    available = list(range(columns))
    while available and cursor < rows:
        active = matrix[cursor:, available]
        uncertainty = errors[cursor:, available]
        reliable = np.abs(active) > bound * uncertainty
        magnitude = np.max(np.where(reliable, np.abs(active), 0), axis=1)
        magnitude[magnitude == 0] = 1
        ratios_all = np.where(reliable, np.abs(active) / magnitude[:, None], 0)
        eligible = np.flatnonzero(np.max(ratios_all, axis=0) > eps * bound)
        if cursor < len(mandatory_rows) and np.max(ratios_all[0]) <= eps * bound:
            msg = 'mandatory border constraints must retain independent actual equation authority'
            raise ValueError(msg)
        if not eligible.size:
            break
        index = int(np.argmax(ratios_all[0])) if cursor < len(mandatory_rows) else int(eligible[0])
        column = available.pop(index)
        ratios = ratios_all[:, index]
        chosen = cursor if cursor < len(mandatory_rows) else cursor + int(np.argmax(ratios))
        # Original homogeneous anchors were already proved above. Rank discovery
        # now belongs to the matrix and its entry bounds, never the evolving RHS.
        matrix[[cursor, chosen]] = matrix[[chosen, cursor]]
        errors[[cursor, chosen]] = errors[[chosen, cursor]]
        rhs[[cursor, chosen]] = rhs[[chosen, cursor]]
        row_ids[[cursor, chosen]] = row_ids[[chosen, cursor]]
        pivot = matrix[cursor, column]
        matrix[cursor] /= pivot
        errors[cursor] = errors[cursor] / abs(pivot) + eps * np.abs(matrix[cursor])
        rhs[cursor] /= pivot
        for row in range(rows):
            if row == cursor:
                continue
            factor = matrix[row, column]
            factor_error = errors[row, column]
            original = matrix[row].copy()
            matrix[row] -= factor * matrix[cursor]
            errors[row] += (
                abs(factor) * errors[cursor]
                + factor_error * np.abs(matrix[cursor])
                + eps * (np.abs(original) + abs(factor) * np.abs(matrix[cursor]))
            )
            rhs[row] -= factor * rhs[cursor]
            matrix[row, column] = 0
        pivots.append(column)
        cursor += 1
        if cursor == rows:
            break
    free_columns = np.setdiff1d(np.arange(columns), pivots)
    answer = np.zeros(columns)
    answer[pivots] = rhs[:cursor]
    if pivots:
        # Rank discovery is not the physical solve: reconstruct from the
        # original selected equations before elimination noise becomes a state.
        selected = row_ids[:cursor]
        subsystem = original_matrix[selected][:, pivots]
        answer[pivots] = np.linalg.solve(subsystem, original_rhs[selected])
        for _ in range(3):
            # The EXACT defect: a working-precision one rounds an answer's last ulp to zero.
            defect = rounded_defect(original_matrix[selected], answer, original_rhs[selected])
            if not np.any(defect) or not np.isfinite(defect).all():
                break
            answer[pivots] += np.linalg.solve(subsystem, defect)
    directions = np.zeros((columns, len(free_columns)))
    directions[free_columns, np.arange(len(free_columns))] = 1
    if pivots and len(free_columns):
        # A homogeneous column is a solve of those SAME original equations,
        # not the noisy Gauss-Jordan coefficients used to discover their rank.
        forcing = -original_matrix[selected][:, free_columns]
        support = _direction_support(subsystem, forcing)
        directions[pivots] = np.linalg.solve(subsystem, forcing)
        for column in range(len(free_columns)):
            for _ in range(3):
                defect = rounded_defect(original_matrix[selected], directions[:, column], np.zeros(cursor))
                if not np.any(defect) or not np.isfinite(defect).all():
                    break
                directions[pivots, column] += np.linalg.solve(subsystem, defect)
        # Dense solves/refinement can pollute upstream homogeneous blocks.
        # Only the original selected equations' structural proof restores zeros.
        directions[pivots] = np.where(support, directions[pivots], 0.0)
    return answer, directions
