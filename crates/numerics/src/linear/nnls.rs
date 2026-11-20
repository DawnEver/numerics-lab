//! Non-negative least squares: `argmin ||A x - b||_2` subject to `x >= 0`, with `scipy.optimize.nnls` semantics.
//!
//! Lawson & Hanson's active-set method (*Solving Least Squares Problems*, SIAM 1995, ch. 23): grow the passive set
//! by the column of largest dual `w = A^T (b - A x)`, solve the unconstrained problem on it, and step back along the
//! segment to the first bound a negative component crosses. Same algorithm as SciPy, not the same floating-point
//! program: `tests/golden_nnls.rs` holds it to round-off of SciPy's answer, degenerate columns included.
//!
//! Each sub-problem is a Householder QR of the passive columns (dense, `m x |P|`), so this is for the small dense
//! problems NNLS is used on (fitting non-negative weights), not a large sparse solver.

use std::fmt;

/// The solution: `x >= 0` and `rnorm = ||A x - b||_2` (scipy's pair), with the sub-problem solves it took.
#[derive(Clone, Debug, PartialEq)]
pub struct Nnls {
    pub x: Vec<f64>,
    pub rnorm: f64,
    pub iterations: usize,
}

/// Why no solution was returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NnlsError {
    /// `a` is not `m * n` entries or `b` is not `m`.
    Shape {
        m: usize,
        n: usize,
        a: usize,
        b: usize,
    },
    /// A NaN or infinity in `a` or `b` (scipy's `asarray_chkfinite`).
    NotFinite,
    /// The iteration cap (default `3 n`, scipy's) was reached: scipy's `RuntimeError`.
    MaxIterations { maxiter: usize },
}

impl fmt::Display for NnlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { m, n, a, b } => write!(
                f,
                "nnls needs a with {m} x {n} entries and b with {m}, got {a} and {b}"
            ),
            Self::NotFinite => write!(f, "nnls input holds a NaN or infinity"),
            Self::MaxIterations { maxiter } => {
                write!(f, "nnls reached its maximum of {maxiter} iterations")
            }
        }
    }
}

impl std::error::Error for NnlsError {}

/// Solve `min ||A x - b||` over `x >= 0`; `a` row-major `m x n`; `maxiter` defaults to `3 n` (scipy's).
///
/// # Errors
/// [`NnlsError`]: a shape mismatch, a non-finite input, or the iteration cap.
pub fn nnls(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    maxiter: Option<usize>,
) -> Result<Nnls, NnlsError> {
    if a.len() != m * n || b.len() != m {
        return Err(NnlsError::Shape {
            m,
            n,
            a: a.len(),
            b: b.len(),
        });
    }
    if !a.iter().chain(b).all(|v| v.is_finite()) {
        return Err(NnlsError::NotFinite);
    }
    let maxiter = maxiter.unwrap_or(3 * n);
    let col_norm = (0..n)
        .map(|j| (0..m).map(|i| a[i * n + j].powi(2)).sum::<f64>().sqrt())
        .fold(0.0, f64::max);
    // A dual below this is round-off of A^T r, not a descent direction.
    let tol = 10.0 * f64::EPSILON * (m.max(n) as f64) * col_norm * norm(b).max(col_norm);

    let mut x = vec![0.0; n];
    let mut passive = vec![false; n];
    let mut iterations = 0;
    loop {
        let r = residual(a, m, n, &x, b);
        let w: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|i| a[i * n + j] * r[i]).sum())
            .collect();
        // The entering column and the passive solve it gives. A column whose own component comes out
        // non-positive cannot enter (Lawson-Hanson step 6); the next-largest dual is tried instead.
        let mut blocked = vec![false; n];
        let mut z = loop {
            let mut entering: Option<usize> = None;
            for j in (0..n).filter(|&j| !passive[j] && !blocked[j] && w[j] > tol) {
                // `>=`: an exact tie goes to the LATER column, as SciPy does (the duplicate-column golden).
                if entering.is_none_or(|k| w[j] >= w[k]) {
                    entering = Some(j);
                }
            }
            let Some(t) = entering else {
                let rnorm = norm(&residual(a, m, n, &x, b));
                return Ok(Nnls {
                    x,
                    rnorm,
                    iterations,
                });
            };
            passive[t] = true;
            let z = passive_solve(a, m, n, &passive, b);
            if z[t] > 0.0 {
                break z;
            }
            passive[t] = false;
            blocked[t] = true;
        };
        loop {
            iterations += 1;
            if iterations > maxiter {
                return Err(NnlsError::MaxIterations { maxiter });
            }
            if (0..n).all(|j| !passive[j] || z[j] > 0.0) {
                x = z;
                break;
            }
            // Step from x toward z until the first passive component reaches its bound.
            let (mut alpha, mut blocking) = (f64::INFINITY, 0);
            for j in (0..n).filter(|&j| passive[j] && z[j] <= 0.0) {
                let s = x[j] / (x[j] - z[j]);
                if s < alpha {
                    (alpha, blocking) = (s, j);
                }
            }
            let stepped: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
            for j in stepped {
                x[j] += alpha * (z[j] - x[j]);
                // The blocking component lands on its bound exactly, not one rounding short of it.
                if j == blocking || x[j] <= 0.0 {
                    passive[j] = false;
                    x[j] = 0.0;
                }
            }
            z = passive_solve(a, m, n, &passive, b);
        }
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn residual(a: &[f64], m: usize, n: usize, x: &[f64], b: &[f64]) -> Vec<f64> {
    (0..m)
        .map(|i| b[i] - (0..n).map(|j| a[i * n + j] * x[j]).sum::<f64>())
        .collect()
}

/// The unconstrained least-squares solution on the passive columns (zero elsewhere), by Householder QR.
fn passive_solve(a: &[f64], m: usize, n: usize, passive: &[bool], b: &[f64]) -> Vec<f64> {
    let cols: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
    // The passive columns, each its own vector, reduced in place to R; Q^T b alongside.
    let mut r: Vec<Vec<f64>> = cols
        .iter()
        .map(|&j| (0..m).map(|i| a[i * n + j]).collect())
        .collect();
    let mut qtb = b.to_vec();
    let rank = cols.len().min(m);
    for p in 0..rank {
        let len = norm(&r[p][p..]);
        if len == 0.0 {
            continue;
        }
        let mut v = r[p][p..].to_vec();
        v[0] += if v[0] > 0.0 { len } else { -len };
        let vv: f64 = v.iter().map(|x| x * x).sum();
        let reflect = |c: &mut [f64]| {
            let s = 2.0 * v.iter().zip(&c[p..]).map(|(vi, ci)| vi * ci).sum::<f64>() / vv;
            c[p..].iter_mut().zip(&v).for_each(|(ci, vi)| *ci -= s * vi);
        };
        r[p..].iter_mut().for_each(|c| reflect(c));
        reflect(&mut qtb);
    }
    // Back substitution on R; a zero pivot (a zero column) leaves its component at zero.
    let mut zp = vec![0.0; cols.len()];
    for p in (0..rank).rev() {
        let s: f64 = (p + 1..rank).map(|c| r[c][p] * zp[c]).sum();
        zp[p] = if r[p][p] == 0.0 {
            0.0
        } else {
            (qtb[p] - s) / r[p][p]
        };
    }
    let mut z = vec![0.0; n];
    for (&j, v) in cols.iter().zip(zp) {
        z[j] = v;
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shape_infinities_and_the_cap_are_refused() {
        assert!(matches!(
            nnls(&[1.0; 5], 2, 3, &[0.0; 2], None),
            Err(NnlsError::Shape { .. })
        ));
        assert_eq!(
            nnls(&[f64::NAN], 1, 1, &[1.0], None),
            Err(NnlsError::NotFinite)
        );
        let a = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(
            nnls(&a, 2, 2, &[1.0, 1.0], Some(1)),
            Err(NnlsError::MaxIterations { maxiter: 1 })
        );
    }

    #[test]
    fn no_columns_is_the_norm_of_b() {
        let s = nnls(&[], 2, 0, &[3.0, 4.0], None).expect("solves");
        assert!(s.x.is_empty());
        assert_eq!(s.rnorm, 5.0);
    }
}
