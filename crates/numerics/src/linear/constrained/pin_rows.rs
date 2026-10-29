//! Pinned rows — the Rust mirror of `python/numerics_lab/linear/constrained/pin_rows.py`.
//!
//! Pins an ALREADY BUILT matrix; [`super::fixed_values`] pins an assembly's COO stream.

use crate::sparse::{Coo, Csr, SparseError};

/// Replace each pinned row by the identity row (zero row, unit diagonal).
///
/// The Python operates on a `scipy.sparse.lil_matrix` in place, because LIL is the one format
/// where replacing a row is O(row). There is no LIL in `contracts::sparse` — and adding one to
/// serve a single operation would be a format that exists for a caller rather than for a
/// question — so this REBUILDS through [`Coo`], which is the write format and the one place
/// duplicates are summed.
///
/// It was consolidated out of FOUR hand-copied Python copies that had already drifted (one cast
/// the row index, three did not, and nothing compared them). One implementation is the point.
///
/// # Errors
/// [`SparseError::NotSquare`], or [`SparseError::IndexOutOfBounds`] when a pinned index is
/// outside the matrix.
pub fn pin_rows(matrix: &Csr, pinned: &[u32]) -> Result<Csr, SparseError> {
    let (n_rows, n_cols) = matrix.shape();
    if n_rows != n_cols {
        return Err(SparseError::NotSquare { n_rows, n_cols });
    }
    let mut is_pinned = vec![false; n_rows];
    for &dof in pinned {
        let d = dof as usize;
        if d >= n_rows {
            return Err(SparseError::IndexOutOfBounds {
                row: d,
                col: d,
                n_rows,
                n_cols,
            });
        }
        is_pinned[d] = true;
    }
    let mut out = Coo::with_capacity(n_rows, n_cols, matrix.nnz() + pinned.len());
    let (indptr, indices, values) = (matrix.indptr(), matrix.indices(), matrix.values());
    for r in 0..n_rows {
        if is_pinned[r] {
            out.push(r as u32, r as u32, 1.0)?;
            continue;
        }
        for i in indptr[r] as usize..indptr[r + 1] as usize {
            out.push(r as u32, indices[i], values[i])?;
        }
    }
    out.to_csr()
}
