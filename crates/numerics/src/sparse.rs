//! The sparse matrix, as a type BOTH sides own — the keystone of ruling D3.
//!
//! D3 (2026-09-03) is that Rust owns the data structures rather than marshalling numpy across
//! the FFI on every call. For the solve path that decision reduces almost entirely to this
//! file: an assembly produces a sparse operator, a nonlinear iteration re-produces it every
//! step, and a linear solve consumes it. If that operator is a numpy object, every one of
//! those steps pays a conversion; if it is owned here, the whole loop runs without crossing
//! the boundary at all.
//!
//! It lives in the numerics crate, the leaf every layer above may depend on: the solvers consume
//! it and the assemblers produce it, so the engine that solves it owns it.
//!
//! # Why COO and CSR, and not one of them
//!
//! They answer different questions and the Python already uses both. Assembly is a stream of
//! `(row, col, value)` contributions with duplicates — that is COO, and it is the only format
//! in which "add another element's contribution" is O(1). Solving wants each row's entries
//! contiguous and its columns sorted — that is CSR. So [`Coo`] is the write format, [`Csr`] the
//! read format, and [`Coo::to_csr`] is the one place duplicates are summed.
//!
//! Summing duplicates rather than overwriting them is not a detail: an FE assembly relies on
//! it (a node shared by k elements contributes k times and the entries must ADD), while a
//! boundary-condition pass relies on the opposite. `scipy.sparse.coo_matrix.tocsr()` sums, so
//! the port sums, and the Dirichlet path gets its own explicit row-clearing operation instead
//! of leaning on a format's implicit behaviour.

use std::fmt;

/// Every way a sparse matrix can be refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseError {
    /// The three COO arrays do not have the same length.
    RaggedTriplets {
        rows: usize,
        cols: usize,
        values: usize,
    },
    /// An index outside the declared shape.
    IndexOutOfBounds {
        row: usize,
        col: usize,
        n_rows: usize,
        n_cols: usize,
    },
    /// An operation that needs a square matrix got a rectangular one.
    NotSquare { n_rows: usize, n_cols: usize },
    /// A vector whose length contradicts the matrix shape.
    ShapeMismatch { expected: usize, got: usize },
}

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RaggedTriplets { rows, cols, values } => write!(
                f,
                "COO triplets must be the same length; got rows={rows}, cols={cols}, values={values}"
            ),
            Self::IndexOutOfBounds {
                row,
                col,
                n_rows,
                n_cols,
            } => write!(
                f,
                "triplet ({row}, {col}) is outside the declared {n_rows}x{n_cols} shape. The \
                 shape is DECLARED rather than inferred from the maximum index, so an assembly \
                 that writes past the end is a refusal instead of a silently grown matrix."
            ),
            Self::NotSquare { n_rows, n_cols } => {
                write!(f, "expected a square matrix; got {n_rows}x{n_cols}")
            }
            Self::ShapeMismatch { expected, got } => {
                write!(f, "expected a vector of length {expected}; got {got}")
            }
        }
    }
}

impl std::error::Error for SparseError {}

/// Coordinate-format sparse matrix — the ASSEMBLY format.
///
/// Duplicate `(row, col)` entries are allowed and are SUMMED when converted to [`Csr`]. That
/// is what makes it the assembly format: an element kernel pushes its contributions without
/// knowing which other elements share the node.
///
/// The shape is DECLARED at construction, never inferred from the maximum index. An inferred
/// shape silently shrinks when the last row happens to be empty — which for an FE operator is
/// exactly the case where a boundary node was constrained away.
#[derive(Debug, Clone, PartialEq)]
pub struct Coo {
    n_rows: usize,
    n_cols: usize,
    rows: Vec<u32>,
    cols: Vec<u32>,
    values: Vec<f64>,
}

impl Coo {
    /// An empty matrix of the declared shape, ready to accumulate.
    #[must_use]
    pub const fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            rows: Vec::new(),
            cols: Vec::new(),
            values: Vec::new(),
        }
    }

    /// An empty matrix with room for `capacity` triplets — the assembly's own estimate.
    #[must_use]
    pub fn with_capacity(n_rows: usize, n_cols: usize, capacity: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            rows: Vec::with_capacity(capacity),
            cols: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    /// Build from existing triplet arrays, as they arrive from a numpy `coo_matrix`.
    ///
    /// # Errors
    /// [`SparseError::RaggedTriplets`] or [`SparseError::IndexOutOfBounds`].
    pub fn from_triplets(
        n_rows: usize,
        n_cols: usize,
        rows: Vec<u32>,
        cols: Vec<u32>,
        values: Vec<f64>,
    ) -> Result<Self, SparseError> {
        if rows.len() != cols.len() || rows.len() != values.len() {
            return Err(SparseError::RaggedTriplets {
                rows: rows.len(),
                cols: cols.len(),
                values: values.len(),
            });
        }
        for (&r, &c) in rows.iter().zip(&cols) {
            if r as usize >= n_rows || c as usize >= n_cols {
                return Err(SparseError::IndexOutOfBounds {
                    row: r as usize,
                    col: c as usize,
                    n_rows,
                    n_cols,
                });
            }
        }
        Ok(Self {
            n_rows,
            n_cols,
            rows,
            cols,
            values,
        })
    }

    /// Accumulate one contribution. Unchecked on purpose — see [`Coo::push`] for the checked
    /// form. This is the innermost line of an element-assembly loop and a bounds check per
    /// triplet is the cost the whole D3 ruling exists to remove; the shape is verified once,
    /// in [`Coo::to_csr`], where an out-of-range index still cannot escape.
    pub fn push_unchecked(&mut self, row: u32, col: u32, value: f64) {
        self.rows.push(row);
        self.cols.push(col);
        self.values.push(value);
    }

    /// Accumulate one contribution, checking it against the declared shape.
    ///
    /// # Errors
    /// [`SparseError::IndexOutOfBounds`].
    pub fn push(&mut self, row: u32, col: u32, value: f64) -> Result<(), SparseError> {
        if row as usize >= self.n_rows || col as usize >= self.n_cols {
            return Err(SparseError::IndexOutOfBounds {
                row: row as usize,
                col: col as usize,
                n_rows: self.n_rows,
                n_cols: self.n_cols,
            });
        }
        self.push_unchecked(row, col, value);
        Ok(())
    }

    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    /// Number of stored triplets, duplicates included.
    #[must_use]
    pub fn nnz_stored(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Compress to [`Csr`], SUMMING duplicate `(row, col)` entries.
    ///
    /// Summing is what `scipy.sparse.coo_matrix.tocsr()` does and what an FE assembly needs: a
    /// node shared by `k` elements contributes `k` times and the entries must add. A format
    /// that overwrote instead would silently keep whichever element happened to be last.
    ///
    /// Exact zeros produced by cancellation are KEPT rather than dropped. Dropping them would
    /// make the sparsity pattern depend on the numbers in it, so two assemblies of the same
    /// mesh could produce structurally different matrices — and a factorisation cached on the
    /// pattern would then be silently invalid.
    ///
    /// # Errors
    /// [`SparseError::IndexOutOfBounds`] for a triplet pushed through
    /// [`Coo::push_unchecked`] that lies outside the declared shape.
    pub fn to_csr(&self) -> Result<Csr, SparseError> {
        for (&r, &c) in self.rows.iter().zip(&self.cols) {
            if r as usize >= self.n_rows || c as usize >= self.n_cols {
                return Err(SparseError::IndexOutOfBounds {
                    row: r as usize,
                    col: c as usize,
                    n_rows: self.n_rows,
                    n_cols: self.n_cols,
                });
            }
        }
        // Counting sort by row: O(nnz), and it is what makes this cheap enough to run every
        // nonlinear iteration rather than once.
        let mut counts = vec![0usize; self.n_rows + 1];
        for &r in &self.rows {
            counts[r as usize + 1] += 1;
        }
        for i in 0..self.n_rows {
            counts[i + 1] += counts[i];
        }
        let mut cursor = counts.clone();
        let nnz = self.values.len();
        let mut cols = vec![0u32; nnz];
        let mut vals = vec![0.0; nnz];
        for i in 0..nnz {
            let slot = &mut cursor[self.rows[i] as usize];
            cols[*slot] = self.cols[i];
            vals[*slot] = self.values[i];
            *slot += 1;
        }
        // Sort each row by column and coalesce duplicates in place.
        let mut out_indptr = Vec::with_capacity(self.n_rows + 1);
        let mut out_cols = Vec::with_capacity(nnz);
        let mut out_vals = Vec::with_capacity(nnz);
        out_indptr.push(0u32);
        let mut scratch: Vec<(u32, f64)> = Vec::new();
        for r in 0..self.n_rows {
            let (lo, hi) = (counts[r], counts[r + 1]);
            scratch.clear();
            scratch.extend((lo..hi).map(|i| (cols[i], vals[i])));
            scratch.sort_by_key(|&(c, _)| c);
            let mut i = 0;
            while i < scratch.len() {
                let col = scratch[i].0;
                let mut acc = 0.0;
                while i < scratch.len() && scratch[i].0 == col {
                    acc += scratch[i].1;
                    i += 1;
                }
                out_cols.push(col);
                out_vals.push(acc);
            }
            out_indptr.push(out_cols.len() as u32);
        }
        Ok(Csr {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            indptr: out_indptr,
            indices: out_cols,
            values: out_vals,
        })
    }
}

/// Compressed-sparse-row matrix — the SOLVE format.
///
/// Row `i` occupies `indices[indptr[i]..indptr[i+1]]`, columns ASCENDING and unique. Those two
/// invariants are established by [`Coo::to_csr`] and are what let a solve walk a row without
/// searching; [`Csr::check_invariants`] states them as something checkable rather than as a
/// comment, because every operation below assumes them.
#[derive(Debug, Clone, PartialEq)]
pub struct Csr {
    n_rows: usize,
    n_cols: usize,
    indptr: Vec<u32>,
    indices: Vec<u32>,
    values: Vec<f64>,
}

impl Csr {
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn indptr(&self) -> &[u32] {
        &self.indptr
    }

    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutable access to the values, with the pattern held fixed.
    ///
    /// This is the whole point of separating pattern from values: a nonlinear iteration
    /// re-assembles the SAME sparsity with new numbers every step, and a solver that cached a
    /// symbolic factorisation on the pattern stays valid across that update. Handing out the
    /// values alone makes "the pattern did not change" a property of the API rather than a
    /// promise in a docstring.
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// `y = A x`.
    ///
    /// # Errors
    /// [`SparseError::ShapeMismatch`] when `x` is not `n_cols` long.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, SparseError> {
        if x.len() != self.n_cols {
            return Err(SparseError::ShapeMismatch {
                expected: self.n_cols,
                got: x.len(),
            });
        }
        let mut y = vec![0.0; self.n_rows];
        for (r, y_r) in y.iter_mut().enumerate() {
            let (lo, hi) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
            let mut acc = 0.0;
            for i in lo..hi {
                acc += self.values[i] * x[self.indices[i] as usize];
            }
            *y_r = acc;
        }
        Ok(y)
    }

    /// The main diagonal, with a structural absence read as `0.0`.
    ///
    /// A Jacobi preconditioner wants this, and it wants the absence to be a zero rather than a
    /// panic — a row with no diagonal entry is a real (singular) matrix, and diagnosing it is
    /// the solver's job, not this accessor's.
    ///
    /// # Errors
    /// [`SparseError::NotSquare`].
    pub fn diagonal(&self) -> Result<Vec<f64>, SparseError> {
        if self.n_rows != self.n_cols {
            return Err(SparseError::NotSquare {
                n_rows: self.n_rows,
                n_cols: self.n_cols,
            });
        }
        let mut d = vec![0.0; self.n_rows];
        for (r, d_r) in d.iter_mut().enumerate() {
            let (lo, hi) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
            // Columns ascend, so a binary search is honest here.
            if let Ok(k) = self.indices[lo..hi].binary_search(&(r as u32)) {
                *d_r = self.values[lo + k];
            }
        }
        Ok(d)
    }

    /// Is the matrix structurally AND numerically symmetric to `tol`?
    ///
    /// The linear registry tags `cg`/`minres` as assuming a symmetric operator, and the Python
    /// benchmark harness reads that tag to skip them on a non-symmetric Jacobian. The tag says
    /// what a SOLVER assumes; this says what a MATRIX is, and the two are worth being able to
    /// compare — a symmetric-only solver silently run on an asymmetric operator converges to
    /// the wrong answer rather than failing.
    #[must_use]
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if self.n_rows != self.n_cols {
            return false;
        }
        for r in 0..self.n_rows {
            let (lo, hi) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
            for i in lo..hi {
                let c = self.indices[i] as usize;
                let (clo, chi) = (self.indptr[c] as usize, self.indptr[c + 1] as usize);
                let mirror = self.indices[clo..chi]
                    .binary_search(&(r as u32))
                    .map_or(0.0, |k| self.values[clo + k]);
                if (self.values[i] - mirror).abs() > tol * self.values[i].abs().max(1.0) {
                    return false;
                }
            }
        }
        true
    }

    /// Check the two invariants every operation here assumes: `indptr` is non-decreasing and
    /// ends at `nnz`, and each row's columns ascend strictly and stay in range.
    ///
    /// Not called on the hot path. It exists so a test — or a future FFI constructor taking
    /// borrowed numpy CSR arrays, which is where these invariants can genuinely be violated —
    /// can ASK, instead of the assumption living only in this doc comment.
    #[must_use]
    pub fn check_invariants(&self) -> bool {
        if self.indptr.len() != self.n_rows + 1 || self.indptr[0] != 0 {
            return false;
        }
        if *self.indptr.last().unwrap_or(&0) as usize != self.values.len() {
            return false;
        }
        if self.indices.len() != self.values.len() {
            return false;
        }
        for r in 0..self.n_rows {
            let (lo, hi) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
            if lo > hi || hi > self.indices.len() {
                return false;
            }
            for i in lo..hi {
                if self.indices[i] as usize >= self.n_cols {
                    return false;
                }
                if i > lo && self.indices[i] <= self.indices[i - 1] {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 3x3 the doc examples use: a symmetric tridiagonal with one duplicate contribution.
    fn tri() -> Coo {
        let mut a = Coo::new(3, 3);
        for (r, c, v) in [
            (0, 0, 2.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 1.0),
            (1, 1, 1.0), // the duplicate: two elements contributing to one node
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 2.0),
        ] {
            a.push(r, c, v).unwrap();
        }
        a
    }

    #[test]
    fn duplicates_are_summed_not_overwritten() {
        // The FE assembly's load-bearing property: a node shared by k elements adds k times.
        let csr = tri().to_csr().unwrap();
        let d = csr.diagonal().unwrap();
        assert_eq!(d, vec![2.0, 2.0, 2.0], "the two 1.0 contributions must ADD");
    }

    #[test]
    fn the_csr_invariants_hold_after_compression() {
        let csr = tri().to_csr().unwrap();
        assert!(csr.check_invariants());
        assert_eq!(csr.nnz(), 7); // 8 triplets, one duplicate pair coalesced
    }

    #[test]
    fn matvec_agrees_with_the_dense_product() {
        let csr = tri().to_csr().unwrap();
        let x = [1.0, 2.0, 3.0];
        // dense: [2*1 - 1*2, -1*1 + 2*2 - 1*3, -1*2 + 2*3] = [0, 0, 4]
        assert_eq!(csr.matvec(&x).unwrap(), vec![0.0, 0.0, 4.0]);
    }

    #[test]
    fn matvec_refuses_a_vector_of_the_wrong_length() {
        let csr = tri().to_csr().unwrap();
        assert!(matches!(
            csr.matvec(&[1.0, 2.0]).unwrap_err(),
            SparseError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn an_index_outside_the_declared_shape_is_refused() {
        let mut a = Coo::new(2, 2);
        assert!(matches!(
            a.push(2, 0, 1.0).unwrap_err(),
            SparseError::IndexOutOfBounds { .. }
        ));
        // ...and the unchecked path still cannot escape: to_csr re-checks.
        a.push_unchecked(9, 9, 1.0);
        assert!(matches!(
            a.to_csr().unwrap_err(),
            SparseError::IndexOutOfBounds { .. }
        ));
    }

    #[test]
    fn the_shape_is_declared_not_inferred_from_the_maximum_index() {
        // A 4x4 whose last row and column are structurally empty -- exactly the shape a
        // constrained-away boundary node produces. An inferred shape would silently return 2x2.
        let mut a = Coo::new(4, 4);
        a.push(0, 0, 1.0).unwrap();
        a.push(1, 1, 1.0).unwrap();
        let csr = a.to_csr().unwrap();
        assert_eq!(csr.shape(), (4, 4));
        assert_eq!(csr.diagonal().unwrap(), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_cancellation_zero_is_kept_so_the_pattern_is_a_function_of_the_mesh_alone() {
        let mut a = Coo::new(2, 2);
        a.push(0, 1, 1.0).unwrap();
        a.push(0, 1, -1.0).unwrap();
        let csr = a.to_csr().unwrap();
        assert_eq!(csr.nnz(), 1, "the cancelled entry stays in the pattern");
        assert_eq!(csr.values(), &[0.0]);
    }

    #[test]
    fn columns_come_out_ascending_whatever_order_they_went_in() {
        let mut a = Coo::new(1, 5);
        for c in [4u32, 0, 3, 1] {
            a.push(0, c, f64::from(c)).unwrap();
        }
        let csr = a.to_csr().unwrap();
        assert_eq!(csr.indices(), &[0, 1, 3, 4]);
        assert_eq!(csr.values(), &[0.0, 1.0, 3.0, 4.0]);
        assert!(csr.check_invariants());
    }

    #[test]
    fn symmetry_is_measured_not_assumed() {
        assert!(tri().to_csr().unwrap().is_symmetric(1e-12));
        let mut a = Coo::new(2, 2);
        a.push(0, 1, 1.0).unwrap();
        a.push(1, 0, 2.0).unwrap();
        assert!(!a.to_csr().unwrap().is_symmetric(1e-12));
    }

    #[test]
    fn a_diagonal_is_refused_on_a_rectangular_matrix() {
        let a = Coo::new(2, 3);
        assert!(matches!(
            a.to_csr().unwrap().diagonal().unwrap_err(),
            SparseError::NotSquare { .. }
        ));
    }

    #[test]
    fn ragged_triplet_arrays_are_refused() {
        assert!(matches!(
            Coo::from_triplets(2, 2, vec![0], vec![0, 1], vec![1.0]).unwrap_err(),
            SparseError::RaggedTriplets { .. }
        ));
    }

    #[test]
    fn values_mut_updates_the_numbers_and_leaves_the_pattern_alone() {
        // The property a cached symbolic factorisation depends on.
        let mut csr = tri().to_csr().unwrap();
        let pattern = (csr.indptr().to_vec(), csr.indices().to_vec());
        for v in csr.values_mut() {
            *v *= 3.0;
        }
        assert_eq!((csr.indptr().to_vec(), csr.indices().to_vec()), pattern);
        assert_eq!(csr.diagonal().unwrap(), vec![6.0, 6.0, 6.0]);
    }

    #[test]
    fn an_empty_matrix_compresses_to_empty_rows_rather_than_failing() {
        let csr = Coo::new(3, 3).to_csr().unwrap();
        assert!(csr.check_invariants());
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.matvec(&[1.0, 1.0, 1.0]).unwrap(), vec![0.0; 3]);
    }
}
