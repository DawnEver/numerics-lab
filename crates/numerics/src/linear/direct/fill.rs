//! The FILL of a sparse direct factor, read from its symbolic analysis alone -- no numeric storage.
//!
//! A direct solve allocates its factor before it can report that the factor does not fit; on a
//! shared box that allocation is the failure (MEASURED 2026-10-20, ipm_uv_tesla_3d5 sdm: an
//! 806,777-unknown seed took RSS from 10.5 to 15.96 GiB inside the factor and died). The symbolic
//! analysis is O(nnz(A)) in memory and names `nnz(L)` exactly for the ordering it chose, so a caller
//! can refuse BEFORE allocating.
//!
//! Moved here from motronics-studio's `core_num::fill` on 2026-11-19.

use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering,
};
use faer::sparse::SymbolicSparseColMat;
use faer::Side;

/// `nnz(L)` of the Cholesky factor of a square pattern under the AMD ordering.
///
/// `col_ptr` / `row_idx` describe the FULL pattern (both triangles, CSC; a symmetric pattern's CSR
/// is the same arrays). Only one triangle is read, so an unsymmetric pattern must be symmetrized
/// (`A + A^T`) by the caller -- the structure the LU's own fill is bounded against.
///
/// # Errors
/// A string naming the defect when the arrays do not form a valid `n x n` pattern or faer refuses
/// the analysis.
pub fn cholesky_factor_len(
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
) -> Result<usize, String> {
    if col_ptr.len() != n + 1
        || col_ptr.last().copied() != Some(row_idx.len())
        || col_ptr.windows(2).any(|w| w[0] > w[1])
        || row_idx.iter().any(|&i| i >= n)
        || col_ptr
            .windows(2)
            .any(|w| row_idx[w[0]..w[1]].windows(2).any(|r| r[0] >= r[1]))
    {
        return Err(format!(
            "the arrays do not form an n={n} square CSC pattern with sorted unique rows"
        ));
    }
    let pattern = SymbolicSparseColMat::<usize>::new_checked(n, n, col_ptr, None, row_idx);
    let symbolic = factorize_symbolic_cholesky(
        pattern.as_ref(),
        Side::Lower,
        SymmetricOrdering::Amd,
        CholeskySymbolicParams::default(),
    )
    .map_err(|e| format!("symbolic cholesky refused the n={n} pattern: {e:?}"))?;
    Ok(symbolic.len_val())
}

#[cfg(test)]
mod tests {
    use super::cholesky_factor_len;

    fn pattern(n: usize, col: impl Fn(usize) -> Vec<usize>) -> (Vec<usize>, Vec<usize>) {
        let (mut ptr, mut idx) = (vec![0usize], Vec::new());
        for j in 0..n {
            let mut c = col(j);
            c.sort_unstable();
            c.dedup();
            idx.extend(c);
            ptr.push(idx.len());
        }
        (ptr, idx)
    }

    /// A tridiagonal pattern factors with no fill: nnz(L) = n + (n - 1).
    #[test]
    fn a_tridiagonal_pattern_has_no_fill() {
        let n = 50;
        let (ptr, idx) = pattern(n, |j| (j.saturating_sub(1)..(j + 2).min(n)).collect());
        assert_eq!(cholesky_factor_len(n, ptr, idx).unwrap(), 2 * n - 1);
    }

    /// An arrow pattern (dense first row and column) is ordered with its hub last: still no fill,
    /// where the natural order fills completely -- the planted control that the ordering is read.
    #[test]
    fn the_ordering_moves_an_arrow_hub_last() {
        let n = 40;
        let (ptr, idx) = pattern(n, |j| if j == 0 { (0..n).collect() } else { vec![0, j] });
        assert_eq!(cholesky_factor_len(n, ptr, idx).unwrap(), 2 * n - 1);
    }

    /// A malformed pattern is refused with its size named, never a panic.
    #[test]
    fn a_malformed_pattern_is_refused() {
        let err = cholesky_factor_len(3, vec![0, 1], vec![0]).unwrap_err();
        assert!(err.contains("n=3"), "{err}");
    }
}
