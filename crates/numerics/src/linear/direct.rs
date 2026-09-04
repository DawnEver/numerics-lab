//! The DIRECT sparse LU backend — `faer`'s sparse LU, guarded.
//!
//! # This is NOT `spsolve`, and the port cannot inherit that promise
//!
//! `src/motronics/gauss/linear/__init__.py` documents its `direct` backend as
//! `scipy.sparse.linalg.spsolve` VERBATIM — "bit-for-bit default, so every golden is unchanged".
//! That claim is about SuperLU's exact pivot sequence, its column ordering, and the order in
//! which it accumulates each update. **No other LU can inherit it** — not `faer`, not a
//! hand-written one — because bit-identity is a property of an implementation, not of the
//! mathematics.
//!
//! So this backend is a DIFFERENT backend and is CHARACTERIZED rather than equated: the golden
//! pins the Python's solution and this side is judged against it at a RESIDUAL-based tolerance,
//! not by bit equality. See `rust/crates/gauss/tests/golden_linear.rs` for the tolerance and its
//! reasoning. Nothing here claims to match `spsolve`.
//!
//! # The singular guard, both halves
//!
//! SuperLU emits a `MatrixRankWarning` on a zero pivot and then returns EITHER NaNs OR a
//! FINITE-but-meaningless vector. The finite case slipped past a caller's finiteness check and
//! shipped as `status=ok` garbage — which is why the Python has two guards, not one:
//!
//! * the BELT — promote the warning to a hard error. Here the equivalent is `faer`'s typed
//!   `LuError`, which for a structurally singular matrix refuses to factorise at all.
//! * the SUSPENDERS — reject a relative residual above `DIRECT_RESIDUAL_RTOL`. This is the half
//!   that catches finite garbage, it lives in [`super::solve_linear_system`], and it applies to
//!   this LU exactly as it applied to SuperLU's. A numerically (rather than structurally)
//!   singular system factorises here and produces non-finite or wildly wrong values, so the
//!   suspenders are not decoration.

use contracts::sparse::{Csr, SparseError};
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};
use faer::Mat;

/// Why the direct factorisation refused.
#[derive(Debug, Clone, PartialEq)]
pub enum DirectFailure {
    /// The matrix is (near-)singular — `faer` refused to factorise it.
    ///
    /// The Python's counterpart is `spsolve` raising on the promoted `MatrixRankWarning`, and the
    /// question it should prompt is the same: is a gauge / reference set, and is the (loop) basis
    /// full-rank at this mesh resolution?
    Singular {
        /// `faer`'s own description of the failure.
        detail: String,
    },
    /// The operator or the vectors do not agree on a shape.
    Shape(SparseError),
}

impl From<SparseError> for DirectFailure {
    fn from(e: SparseError) -> Self {
        Self::Shape(e)
    }
}

/// Solve `A x = b` by sparse LU.
///
/// # Errors
/// [`DirectFailure::Singular`] when the factorisation refuses, [`DirectFailure::Shape`] when the
/// matrix is not square or `b` is the wrong length. The residual half of the guard is applied by
/// the caller, [`super::solve_linear_system`].
pub fn solve(a: &Csr, b: &[f64]) -> Result<Vec<f64>, DirectFailure> {
    let (n_rows, n_cols) = a.shape();
    if n_rows != n_cols {
        return Err(DirectFailure::Shape(SparseError::NotSquare {
            n_rows,
            n_cols,
        }));
    }
    if b.len() != n_rows {
        return Err(DirectFailure::Shape(SparseError::ShapeMismatch {
            expected: n_rows,
            got: b.len(),
        }));
    }

    let mut triplets = Vec::with_capacity(a.nnz());
    let (indptr, indices, values) = (a.indptr(), a.indices(), a.values());
    for r in 0..n_rows {
        for i in indptr[r] as usize..indptr[r + 1] as usize {
            triplets.push(Triplet::new(r, indices[i] as usize, values[i]));
        }
    }
    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n_rows, n_cols, &triplets)
        .map_err(|e| DirectFailure::Singular {
            detail: format!("{e:?}"),
        })?;
    let symbolic = SymbolicLu::try_new(mat.symbolic()).map_err(|e| DirectFailure::Singular {
        detail: format!("symbolic: {e:?}"),
    })?;
    let lu =
        Lu::try_new_with_symbolic(symbolic, mat.as_ref()).map_err(|e| DirectFailure::Singular {
            detail: format!("numeric: {e:?}"),
        })?;
    let rhs = Mat::from_fn(n_rows, 1, |i, _| b[i]);
    let x = lu.solve(&rhs);
    Ok((0..n_rows).map(|i| x[(i, 0)]).collect())
}
