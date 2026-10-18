//! The linear seam's own tests, mirroring `tests/unit/test_linear_solvers.py`,
//! `test_dirichlet_triplets.py`, and the claim `ssor.py` makes for itself in its docstring.

mod direct;
mod fixed_values;
mod iccg;
mod krylov;
mod master_slave;
mod pin_rows;
mod precond;
mod seam;

use crate::sparse::{Coo, Csr};

/// A well-conditioned SPD tridiagonal (a 1-D Laplacian plus a varying diagonal), and a rhs.
///
/// Mirrors the Python fixture: SPD so `cg`/`minres` apply, sparse and diagonally dominant so
/// every Krylov backend converges. The "random" parts are a fixed deterministic recurrence
/// rather than an RNG — a fixture that differs between runs cannot anchor an iteration count.
pub fn spd_system(n: usize) -> (Csr, Vec<f64>) {
    let mut coo = Coo::new(n, n);
    for i in 0..n {
        let jitter = 0.5 + (((i * 37) % 11) as f64) / 10.0;
        coo.push(i as u32, i as u32, 2.0 + jitter).unwrap();
        if i + 1 < n {
            coo.push(i as u32, (i + 1) as u32, -1.0).unwrap();
            coo.push((i + 1) as u32, i as u32, -1.0).unwrap();
        }
    }
    let b = (0..n).map(|i| ((i % 7) as f64 - 3.0) / 2.0).collect();
    (coo.to_csr().unwrap(), b)
}

/// A well-conditioned NON-symmetric tridiagonal (asymmetric off-diagonals), and a rhs.
pub fn nonsymmetric_system(n: usize) -> (Csr, Vec<f64>) {
    let mut coo = Coo::new(n, n);
    for i in 0..n {
        coo.push(i as u32, i as u32, 3.0 + ((i % 5) as f64) / 4.0)
            .unwrap();
        if i + 1 < n {
            coo.push((i + 1) as u32, i as u32, -1.0).unwrap();
            coo.push(i as u32, (i + 1) as u32, -0.5).unwrap();
        }
    }
    let b = (0..n).map(|i| 1.0 + ((i % 3) as f64)).collect();
    (coo.to_csr().unwrap(), b)
}

/// The 2D 5-point Laplacian on a `side x side` grid — the operator `ssor.py` measured on.
///
/// Its diagonal is CONSTANT, which is the whole point: Jacobi is a no-op on it, so any change in
/// the CG iteration count is attributable to the preconditioner and not to a scaling.
pub fn laplacian_2d(side: usize) -> (Csr, Vec<f64>) {
    let n = side * side;
    let mut coo = Coo::new(n, n);
    let idx = |r: usize, c: usize| (r * side + c) as u32;
    for r in 0..side {
        for c in 0..side {
            coo.push(idx(r, c), idx(r, c), 4.0).unwrap();
            if r > 0 {
                coo.push(idx(r, c), idx(r - 1, c), -1.0).unwrap();
            }
            if r + 1 < side {
                coo.push(idx(r, c), idx(r + 1, c), -1.0).unwrap();
            }
            if c > 0 {
                coo.push(idx(r, c), idx(r, c - 1), -1.0).unwrap();
            }
            if c + 1 < side {
                coo.push(idx(r, c), idx(r, c + 1), -1.0).unwrap();
            }
        }
    }
    // A ROUGH rhs, not a constant one. A constant rhs is close to a single low-frequency mode and
    // CG eats it in far fewer iterations than the operator's condition number implies, which
    // compresses exactly the count this fixture exists to measure. The recurrence is
    // deterministic, so the count stays a property of the operator and not of a machine.
    let b = (0..n)
        .map(|i| (((i * 7919) % 1009) as f64) / 1009.0 - 0.5)
        .collect();
    (coo.to_csr().unwrap(), b)
}

/// A rank-deficient system: two identical rows. The direct guard must refuse it.
pub fn duplicate_row_system() -> (Csr, Vec<f64>) {
    let mut coo = Coo::new(3, 3);
    for (r, row) in [[2.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 3.0]]
        .iter()
        .enumerate()
    {
        for (c, v) in row.iter().enumerate() {
            coo.push(r as u32, c as u32, *v).unwrap();
        }
    }
    (coo.to_csr().unwrap(), vec![1.0, 2.0, 1.0])
}

/// A structurally singular system: an entirely zero row.
pub fn zero_row_system() -> (Csr, Vec<f64>) {
    let mut coo = Coo::new(2, 2);
    coo.push(0, 0, 1.0).unwrap();
    (coo.to_csr().unwrap(), vec![1.0, 1.0])
}

pub fn assert_close(got: &[f64], want: &[f64], rtol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = rtol * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: {g} vs {w} (|d| = {:e}, tol {tol:e})",
            (g - w).abs()
        );
    }
}
