//! The dense BORDER of a sparse direct factorisation: the dense rows/columns found, and their
//! elimination through the `k x k` Schur complement so the supernodal kernel sees only the sparse
//! block. Split out of `direct.rs` on 2026-09-29; see [`super::DirectFactorization`] for why.

use super::{DirectFailure, FactorKind, Kernel};
use crate::sparse::{Coo, Csr};
use faer::linalg::solvers::{PartialPivLu, Solve};
use faer::Mat;

// -- The dense border ---------------------------------------------------------------------------

/// A row/column is DENSE when it stores more than this many entries AND more than
/// [`DENSE_SQRT_FACTOR`]` * sqrt(n)`.
pub(super) const DENSE_FLOOR: usize = 64;
/// See [`DENSE_FLOOR`].
const DENSE_SQRT_FACTOR: f64 = 10.0;
/// More dense indices than this and the Schur block itself stops being small; the plain kernel
/// takes the whole operator instead.
const MAX_BORDER: usize = 64;

/// Indices whose row or column is dense, or empty when there are none (or too many).
pub(super) fn dense_indices(n: usize, indptr: &[u32], indices: &[u32]) -> Vec<usize> {
    let mut count = vec![0usize; n];
    for r in 0..n {
        count[r] += (indptr[r + 1] - indptr[r]) as usize;
    }
    let mut col = vec![0usize; n];
    for &c in indices {
        col[c as usize] += 1;
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let bar = DENSE_FLOOR.max((DENSE_SQRT_FACTOR * (n as f64).sqrt()) as usize);
    let dense: Vec<usize> = (0..n).filter(|&i| count[i].max(col[i]) > bar).collect();
    if dense.len() > MAX_BORDER || dense.len() == n {
        return Vec::new();
    }
    dense
}

/// `A = [[A11, A12], [A21, A22]]` with the dense indices last: `A11` is factorised sparsely and
/// the border is eliminated through the `k x k` Schur complement `S = A22 - A21 A11^{-1} A12`.
pub(super) struct Border {
    pub(super) dense: Vec<usize>,
    interior: Vec<usize>,
    /// `(original CSR slot, slot in a11)`.
    to_a11: Vec<(usize, usize)>,
    /// `(original slot, interior row, border col)`.
    to_a12: Vec<(usize, usize, usize)>,
    /// `(original slot, border row, interior col)`.
    to_a21: Vec<(usize, usize, usize)>,
    /// `(original slot, border row, border col)`.
    to_a22: Vec<(usize, usize, usize)>,
    a11: Csr,
    pub(super) kernel: Kernel,
    /// `A11^{-1} A12`, `n1 x k` column-major.
    y: Vec<f64>,
    /// `A21`, `k x n1` row-major.
    a21: Vec<f64>,
    s_lu: Option<PartialPivLu<f64>>,
}

impl Border {
    pub(super) fn new(a: &Csr, dense: Vec<usize>) -> Result<Self, DirectFailure> {
        let n = a.shape().0;
        let mut where_ = vec![(false, 0usize); n]; // (is_dense, position within its block)
        for (p, &d) in dense.iter().enumerate() {
            where_[d] = (true, p);
        }
        let interior: Vec<usize> = (0..n).filter(|&i| !where_[i].0).collect();
        for (p, &i) in interior.iter().enumerate() {
            where_[i] = (false, p);
        }
        let (indptr, indices) = (a.indptr(), a.indices());
        let n1 = interior.len();
        let mut coo = Coo::new(n1, n1);
        let (mut to_a11, mut to_a12, mut to_a21, mut to_a22) = (vec![], vec![], vec![], vec![]);
        for r in 0..n {
            let (rd, rp) = where_[r];
            for slot in indptr[r] as usize..indptr[r + 1] as usize {
                let (cd, cp) = where_[indices[slot] as usize];
                match (rd, cd) {
                    (false, false) => {
                        to_a11.push((slot, to_a11.len()));
                        #[allow(clippy::cast_possible_truncation)]
                        coo.push(rp as u32, cp as u32, 0.0)?;
                    }
                    (false, true) => to_a12.push((slot, rp, cp)),
                    (true, false) => to_a21.push((slot, rp, cp)),
                    (true, true) => to_a22.push((slot, rp, cp)),
                }
            }
        }
        // Rows are visited in interior order and columns ascend within each row, so the COO is
        // already canonical and `to_a11`'s running count IS the CSR slot.
        let a11 = coo.to_csr()?;
        if a11.nnz() != to_a11.len() {
            return Err(DirectFailure::PatternChanged);
        }
        let kernel = Kernel::new(&a11)?;
        let k = dense.len();
        Ok(Self {
            y: vec![0.0; n1 * k],
            a21: vec![0.0; k * n1],
            dense,
            interior,
            to_a11,
            to_a12,
            to_a21,
            to_a22,
            a11,
            kernel,
            s_lu: None,
        })
    }

    pub(super) fn refactor(&mut self, values: &[f64]) -> Result<FactorKind, DirectFailure> {
        let (n1, k) = (self.interior.len(), self.dense.len());
        self.s_lu = None;
        {
            let slot = self.a11.values_mut();
            for &(o, t) in &self.to_a11 {
                slot[t] = values[o];
            }
        }
        let kind = self.kernel.refactor(&self.a11)?;
        self.y.iter_mut().for_each(|v| *v = 0.0);
        for &(o, r, c) in &self.to_a12 {
            self.y[c * n1 + r] = values[o];
        }
        self.kernel.solve_in_place(&mut self.y, k)?;
        self.a21.iter_mut().for_each(|v| *v = 0.0);
        for &(o, r, c) in &self.to_a21 {
            self.a21[r * n1 + c] = values[o];
        }
        let mut s = Mat::<f64>::zeros(k, k);
        for &(o, r, c) in &self.to_a22 {
            s[(r, c)] += values[o];
        }
        for i in 0..k {
            let row = &self.a21[i * n1..(i + 1) * n1];
            for j in 0..k {
                let col = &self.y[j * n1..(j + 1) * n1];
                s[(i, j)] -= row.iter().zip(col).map(|(p, q)| p * q).sum::<f64>();
            }
        }
        self.s_lu = Some(s.partial_piv_lu());
        Ok(kind)
    }

    /// One bordered solve of `nrhs` column-major columns of length `n`, in place.
    pub(super) fn solve(&self, b: &mut [f64], nrhs: usize) -> Result<(), DirectFailure> {
        let Some(s_lu) = self.s_lu.as_ref() else {
            return Err(DirectFailure::NotFactored);
        };
        let (n1, k) = (self.interior.len(), self.dense.len());
        let n = n1 + k;
        let mut b1 = vec![0.0; n1 * nrhs];
        let mut t = Mat::<f64>::zeros(k, nrhs);
        for c in 0..nrhs {
            let col = &b[c * n..(c + 1) * n];
            for (p, &i) in self.interior.iter().enumerate() {
                b1[c * n1 + p] = col[i];
            }
            for (p, &d) in self.dense.iter().enumerate() {
                t[(p, c)] = col[d];
            }
        }
        self.kernel.solve_in_place(&mut b1, nrhs)?;
        for c in 0..nrhs {
            let y1 = &b1[c * n1..(c + 1) * n1];
            for i in 0..k {
                let row = &self.a21[i * n1..(i + 1) * n1];
                t[(i, c)] -= row.iter().zip(y1).map(|(p, q)| p * q).sum::<f64>();
            }
        }
        let x2 = s_lu.solve(&t);
        for c in 0..nrhs {
            let col = &mut b[c * n..(c + 1) * n];
            for (p, &i) in self.interior.iter().enumerate() {
                let mut v = b1[c * n1 + p];
                for j in 0..k {
                    v -= self.y[j * n1 + p] * x2[(j, c)];
                }
                col[i] = v;
            }
            for (p, &d) in self.dense.iter().enumerate() {
                col[d] = x2[(p, c)];
            }
        }
        Ok(())
    }
}
