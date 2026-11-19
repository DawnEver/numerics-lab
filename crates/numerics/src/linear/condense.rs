//! Block STATIC CONDENSATION: per block, eliminate the interior (bubble) unknowns through their own LU
//! and hand the caller the boundary Schur complement, then recover the interior step from the boundary
//! one. Blocks are independent, so every block is one rayon task; the dense algebra is faer's.
//!
//! A block is row-major `(k + nb)^2` over `(boundary, interior)`:
//! `S = J_nn - J_nb J_bb^-1 J_bn`, `y = J_bb^-1 rhs_b`, interior step `y - Z dv` with `Z = J_bb^-1 J_bn`.
//!
//! Moved here from motronics-studio's `hamilton_sdm` crate (a74923ef38) on 2026-11-19: nothing in it is
//! a subdomain method's own; it is the generic kernel the SDM Newton step calls.

// The kernels index as the dense formulas read (`j[r * size + c]`), so a reviewer reads one against the other.
#![allow(clippy::needless_range_loop)]

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::matmul::matmul;
use faer::linalg::triangular_solve::{
    solve_unit_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rayon::prelude::*;

/// One cell's condensed tangent: `J_bb`'s LU, `J_nb` and `Z = J_bb^-1 J_bn` (`condensed_tangent`).
pub struct Condensed {
    k: usize,
    nb: usize,
    lu: Mat<f64>,
    perm: Vec<usize>,
    j_nb: Mat<f64>,
    z: Mat<f64>,
}

impl Condensed {
    /// `J_bb^-1 rhs` in place on a `(nb, cols)` matrix.
    fn solve_in_place(&self, mut rhs: MatMut<'_, f64>) {
        let permuted = Mat::from_fn(self.nb, rhs.ncols(), |i, j| rhs[(self.perm[i], j)]);
        rhs.copy_from(&permuted);
        solve_unit_lower_triangular_in_place(self.lu.as_ref(), rhs.as_mut(), Par::Seq);
        solve_upper_triangular_in_place(self.lu.as_ref(), rhs.as_mut(), Par::Seq);
    }

    /// Factors one row-major `(k + nb)^2` Jacobian block and returns it with its row-major `(k, k)` Schur
    /// complement `J_nn - J_nb J_bb^-1 J_bn`.
    pub fn new(j: &[f64], k: usize, nb: usize) -> (Self, Vec<f64>) {
        let size = k + nb;
        let at = |r: usize, c: usize| j[r * size + c];
        let mut lu = Mat::from_fn(nb, nb, |r, c| at(k + r, k + c));
        let mut fwd = vec![0usize; nb];
        let mut bwd = vec![0usize; nb];
        {
            use faer::linalg::lu::partial_pivoting::factor::{lu_in_place, lu_in_place_scratch};
            let mut buf = MemBuffer::new(lu_in_place_scratch::<usize, f64>(
                nb,
                nb,
                Par::Seq,
                Default::default(),
            ));
            lu_in_place(
                lu.as_mut(),
                &mut fwd,
                &mut bwd,
                Par::Seq,
                MemStack::new(&mut buf),
                Default::default(),
            );
        }
        let j_nb = Mat::from_fn(k, nb, |r, c| at(r, k + c));
        let mut z = Mat::from_fn(nb, k, |r, c| at(k + r, c));
        let out = Self {
            k,
            nb,
            lu,
            perm: fwd,
            j_nb,
            z: Mat::zeros(0, 0),
        };
        out.solve_in_place(z.as_mut());
        let mut s = Mat::from_fn(k, k, at);
        matmul(
            s.as_mut(),
            Accum::Add,
            out.j_nb.as_ref(),
            z.as_ref(),
            -1.0,
            Par::Seq,
        );
        let schur = (0..k * k).map(|x| s[(x / k, x % k)]).collect();
        (Self { z, ..out }, schur)
    }
}

/// Every cell's condensed tangent, in the order its blocks were handed in.
pub struct CondensedCells {
    pub cells: Vec<Condensed>,
    /// Prefix sums over `k` and over `nb`.
    pub off_k: Vec<usize>,
    pub off_nb: Vec<usize>,
}

impl CondensedCells {
    /// Condenses every row-major block of `j` (`sizes[e] = (k, nb)`, blocks concatenated) in parallel;
    /// returns the cells and the concatenated `(k, k)` Schur complements.
    pub fn new(j: &[f64], sizes: &[(usize, usize)]) -> Result<(Self, Vec<f64>), String> {
        let mut off = vec![0usize];
        for (k, nb) in sizes {
            off.push(off.last().unwrap() + (k + nb) * (k + nb));
        }
        if *off.last().unwrap() != j.len() {
            return Err(format!(
                "{} block entries for sizes asking {}",
                j.len(),
                off.last().unwrap()
            ));
        }
        let parts: Vec<(Condensed, Vec<f64>)> = sizes
            .par_iter()
            .enumerate()
            .map(|(e, &(k, nb))| Condensed::new(&j[off[e]..off[e + 1]], k, nb))
            .collect();
        let mut schur = Vec::new();
        let mut cells = Vec::with_capacity(parts.len());
        for (c, s) in parts {
            schur.extend(s);
            cells.push(c);
        }
        let off_k = offsets_of(cells.iter().map(|c| c.k));
        let off_nb = offsets_of(cells.iter().map(|c| c.nb));
        Ok((
            Self {
                cells,
                off_k,
                off_nb,
            },
            schur,
        ))
    }

    /// `(y, J_nb y)`: every cell's `y = J_bb^-1 rhs_b` and its nodal correction.
    pub fn eliminate(&self, rhs_b: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let parts: Vec<(Vec<f64>, Vec<f64>)> = self
            .cells
            .par_iter()
            .enumerate()
            .map(|(e, c)| {
                let mut y = Mat::from_fn(c.nb, 1, |r, _| rhs_b[self.off_nb[e] + r]);
                c.solve_in_place(y.as_mut());
                let mut corr = Mat::<f64>::zeros(c.k, 1);
                matmul(
                    corr.as_mut(),
                    Accum::Replace,
                    c.j_nb.as_ref(),
                    y.as_ref(),
                    1.0,
                    Par::Seq,
                );
                (
                    (0..c.nb).map(|r| y[(r, 0)]).collect(),
                    (0..c.k).map(|r| corr[(r, 0)]).collect(),
                )
            })
            .collect();
        let (mut y, mut corr) = (Vec::new(), Vec::new());
        for (a, b) in parts {
            y.extend(a);
            corr.extend(b);
        }
        (y, corr)
    }

    /// Every cell's bubble step `y - Z dv` from its nodal step `dv` (`(k,)` per cell).
    pub fn recover(&self, y: &[f64], dv: &[f64]) -> Vec<f64> {
        let parts: Vec<Vec<f64>> = self
            .cells
            .par_iter()
            .enumerate()
            .map(|(e, c)| {
                let mut out = Mat::from_fn(c.nb, 1, |r, _| y[self.off_nb[e] + r]);
                let v = Mat::from_fn(c.k, 1, |r, _| dv[self.off_k[e] + r]);
                matmul(
                    out.as_mut(),
                    Accum::Add,
                    c.z.as_ref(),
                    v.as_ref(),
                    -1.0,
                    Par::Seq,
                );
                (0..c.nb).map(|r| out[(r, 0)]).collect()
            })
            .collect();
        parts.concat()
    }
}

/// `K_e u_e` for every row-major square block of `k` (sizes `m_e`, blocks and vectors concatenated).
pub fn block_matvec(k: &[f64], sizes: &[usize], u: &[f64]) -> Result<Vec<f64>, String> {
    let off_b = offsets_of(sizes.iter().map(|m| m * m));
    let off_v = offsets_of(sizes.iter().copied());
    if *off_b.last().unwrap() != k.len() || *off_v.last().unwrap() != u.len() {
        return Err(format!(
            "{} block entries and {} vector entries for sizes asking {} and {}",
            k.len(),
            u.len(),
            off_b.last().unwrap(),
            off_v.last().unwrap()
        ));
    }
    let parts: Vec<Vec<f64>> = sizes
        .par_iter()
        .enumerate()
        .map(|(e, &m)| {
            let mut out = vec![0.0; m];
            // the motronics `gemm(.., add = false)` this replaces, without its transpose options
            matmul(
                MatMut::from_row_major_slice_mut(&mut out, m, 1),
                Accum::Replace,
                MatRef::from_row_major_slice(&k[off_b[e]..off_b[e + 1]], m, m),
                MatRef::from_row_major_slice(&u[off_v[e]..off_v[e + 1]], m, 1),
                1.0,
                Par::Seq,
            );
            out
        })
        .collect();
    Ok(parts.concat())
}

fn offsets_of(len: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut out = vec![0usize];
    for l in len {
        out.push(out.last().unwrap() + l);
    }
    out
}
