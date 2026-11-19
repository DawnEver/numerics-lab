//! numpy's dense `np.linalg.solve` and `np.linalg.cond` of a small square matrix, bit for bit as the OpenBLAS numpy
//! links calls them (MEASURED against numpy by motronics-studio's
//! `tests/unit/laplace/control/test_a_wide_algebraic_loop_solves_natively_as_python.py`).
//!
//! * [`solve`]: `dgesv` = OpenBLAS `getrf_single` (the [`super::expm`] transcription, `getf2` at these sizes) then
//!   `getrs_N_single` for one right-hand side: the row interchanges, `trsv_NLU` and `trsv_NUN` (each update one
//!   `daxpy` FMA, each diagonal a division).
//! * [`cond`]: `s[0] / s[-1]` of `dgesdd(JOBZ='N')`'s singular values, transcribed for `n <= 2` only -- `dgebd2`'s
//!   Householder (`dlarfg`, `dlarf` through `dgemv_t` / `dger`), then `dbdsdc -> dlasdq -> dbdsqr -> dlasq1 -> dlas2`;
//!   a larger `n` meets the iterative dqds, untranscribed, and is refused by the caller.

use super::expm::{getrf, ExpmError};

/// The largest `n` [`cond`] is transcribed for.
pub const COND_MAX_N: usize = 2;

/// Why [`cond`] gives no number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondError {
    /// An entry is NaN: `dgesdd` returns `info = -4` and numpy raises `LinAlgError('SVD did not converge')`.
    NotConverged,
    /// `n > COND_MAX_N`: past the sizes transcribed.
    Untranscribed { n: usize },
}

/// `np.linalg.solve(a, b)` of the row-major `n x n` `a`; `None` where `getrf` meets an exactly zero pivot (numpy
/// raises `LinAlgError('Singular matrix')`).
#[must_use]
pub fn solve(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    // numpy hands LAPACK the column-major copy
    let mut lu: Vec<f64> = (0..n * n).map(|q| a[(q % n) * n + q / n]).collect();
    let mut ipiv = vec![0usize; n];
    match getrf(&mut lu, n, 0, n, n, &mut ipiv) {
        Ok(()) => {}
        Err(ExpmError::Singular) => return None,
        Err(_) => unreachable!("getrf refuses only a zero pivot"),
    }
    let mut x = b.to_vec();
    for (i, &p) in ipiv.iter().enumerate() {
        x.swap(i, p);
    }
    // trsv_NLU: column by column, `daxpy(-x[i], L[i+1.., i])`
    for i in 0..n {
        let xi = -x[i];
        for r in i + 1..n {
            x[r] = xi.mul_add(lu[r + i * n], x[r]);
        }
    }
    // trsv_NUN: from the bottom, `x[i] /= U[i, i]` then `daxpy(-x[i], U[..i, i])`
    for i in (0..n).rev() {
        x[i] /= lu[i + i * n];
        let xi = -x[i];
        for r in 0..i {
            x[r] = xi.mul_add(lu[r + i * n], x[r]);
        }
    }
    Some(x)
}

/// LAPACK `dlamch('S')`: the safe minimum.
const SFMIN: f64 = f64::MIN_POSITIVE;

/// LAPACK `dlascl('G')` of `values` from `cfrom` to `cto`: multiplied by `cto / cfrom` in steps that never overflow.
fn dlascl(cfrom: f64, cto: f64, values: &mut [f64]) {
    let smlnum = SFMIN;
    let bignum = 1.0 / smlnum;
    let (mut cfromc, mut ctoc) = (cfrom, cto);
    loop {
        let cfrom1 = cfromc * smlnum;
        let (mul, done);
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == 1.0 {
                    return;
                }
            }
        }
        for v in values.iter_mut() {
            *v *= mul;
        }
        if done {
            return;
        }
    }
}

/// LAPACK `dlapy2(x, y)` for finite operands: `sqrt(x^2 + y^2)` without overflow.
fn dlapy2(x: f64, y: f64) -> f64 {
    let (xa, ya) = (x.abs(), y.abs());
    let w = xa.max(ya);
    let z = xa.min(ya);
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        let q = z / w;
        w * (1.0 + q * q).sqrt()
    }
}

/// LAPACK `dlas2(f, g, h)`: the singular values `(ssmin, ssmax)` of `[[f, g], [0, h]]`, unfused as compiled.
fn dlas2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let (fa, ga, ha) = (f.abs(), g.abs(), h.abs());
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    if fhmn == 0.0 {
        let ssmax = if fhmx == 0.0 {
            ga
        } else {
            let (big, small) = (fhmx.max(ga), fhmx.min(ga));
            let q = small / big;
            big * (1.0 + q * q).sqrt()
        };
        return (0.0, ssmax);
    }
    if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let au = {
            let q = ga / fhmx;
            q * q
        };
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let p = as_ * au;
            let q = at * au;
            let c = 1.0 / ((1.0 + p * p).sqrt() + (1.0 + q * q).sqrt());
            let ssmin = (fhmn * c) * au;
            (ssmin + ssmin, ga / (c + c))
        }
    }
}

/// LAPACK `dlarfg(2, alpha, x)`: `(beta, tau, v)` of the reflector taking `[alpha, x]` to `[beta, 0]`.
fn dlarfg2(alpha: f64, x: f64) -> (f64, f64, f64) {
    // `dnrm2` of one element is its magnitude
    let xnorm = x.abs();
    if xnorm == 0.0 {
        return (alpha, 0.0, x);
    }
    let safmin = SFMIN / (f64::EPSILON / 2.0);
    let rsafmn = 1.0 / safmin;
    let (mut alpha, mut x) = (alpha, x);
    let mut beta = -dlapy2(alpha, xnorm).copysign(alpha);
    let mut knt = 0;
    if beta.abs() < safmin {
        loop {
            knt += 1;
            x *= rsafmn;
            beta *= rsafmn;
            alpha *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        beta = -dlapy2(alpha, x.abs()).copysign(alpha);
    }
    let tau = (beta - alpha) / beta;
    let v = (1.0 / (alpha - beta)) * x;
    for _ in 0..knt {
        beta *= safmin;
    }
    (beta, tau, v)
}

/// The singular values of the column-major 2x2 `a`, largest first: `dgebd2` then `dlas2`.
fn singular_values_2(a: [f64; 4]) -> [f64; 2] {
    let [a11, a21, a12, a22] = a;
    let (d1, tau, v2) = dlarfg2(a11, a21);
    // dlarf('Left', [1, v2], tau) on the second column: `lastv` drops a trailing zero of `v`; `iladlc` skips an all
    // zero column; `w = dgemv_t` (its two-row tail); `dger` one `daxpy` FMA per row
    let (mut e1, mut d2) = (a12, a22);
    if tau != 0.0 {
        let lastv = if v2 == 0.0 { 1 } else { 2 };
        let column_nonzero = a12 != 0.0 || (lastv == 2 && a22 != 0.0);
        if column_nonzero {
            // the two-row tail `y += a0 x0 + a1 x1` as GCC contracts it: `fma(a0, x0, a1 x1)`, then the add
            let w = if lastv == 2 { 0.0 + a12.mul_add(1.0, a22 * v2) } else { 1.0f64.mul_add(a12, 0.0) };
            let da = -tau * w;
            e1 = da.mul_add(1.0, a12);
            if lastv == 2 {
                d2 = da.mul_add(v2, a22);
            }
        }
    }
    let (ssmin, ssmax) = dlas2(d1, e1, d2);
    [ssmax, ssmin]
}

/// `np.linalg.cond(a)` of the row-major `n x n` `a` (`n <= COND_MAX_N`): `s[0] / s[-1]`, a NaN quotient read as inf
/// (numpy's own rule for an input without NaN).
///
/// # Errors
/// [`CondError::NotConverged`] for a NaN entry, [`CondError::Untranscribed`] past `COND_MAX_N`.
pub fn cond(a: &[f64], n: usize) -> Result<f64, CondError> {
    if n > COND_MAX_N {
        return Err(CondError::Untranscribed { n });
    }
    if a.iter().any(|v| v.is_nan()) {
        return Err(CondError::NotConverged);
    }
    if n == 1 {
        return Ok(if a[0] == 0.0 || a[0].is_infinite() { f64::INFINITY } else { 1.0 });
    }
    // an infinite entry: dgesdd scales by `bignum / inf = 0`, so the scaled matrix holds a NaN and its quotient is NaN
    if a.iter().any(|v| v.is_infinite()) {
        return Ok(f64::INFINITY);
    }
    let mut col = [a[0], a[2], a[1], a[3]];
    // dgesdd's scaling into [smlnum, bignum]
    let anrm = col.iter().fold(0.0f64, |m, v| if m < v.abs() { v.abs() } else { m });
    let smlnum = SFMIN.sqrt() / f64::EPSILON;
    let bignum = 1.0 / smlnum;
    let scale = if anrm > 0.0 && anrm < smlnum {
        Some(smlnum)
    } else if anrm > bignum {
        Some(bignum)
    } else {
        None
    };
    if let Some(to) = scale {
        dlascl(anrm, to, &mut col);
    }
    let mut s = singular_values_2(col);
    if let Some(from) = scale {
        dlascl(from, anrm, &mut s);
    }
    let r = s[0] / s[1];
    Ok(if r.is_nan() { f64::INFINITY } else { r })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_diagonal_matrix_conditions_by_its_ratio() {
        assert_eq!(cond(&[4.0, 0.0, 0.0, 2.0], 2), Ok(2.0));
        assert_eq!(cond(&[0.0, 0.0, 0.0, 2.0], 2), Ok(f64::INFINITY));
        assert_eq!(cond(&[f64::NAN, 0.0, 0.0, 2.0], 2), Err(CondError::NotConverged));
        assert_eq!(cond(&[1.0; 9], 3), Err(CondError::Untranscribed { n: 3 }));
    }

    #[test]
    fn a_two_by_two_system_solves() {
        let x = solve(&[2.0, 1.0, 1.0, 3.0], 2, &[3.0, 5.0]).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-15 && (x[1] - 1.4).abs() < 1e-15, "{x:?}");
        assert!(solve(&[1.0, 2.0, 2.0, 4.0], 2, &[1.0, 1.0]).is_none());
    }
}
