//! Scalar root finding: `scipy.optimize.brentq` transcribed (scipy's C `brentq`), so a Python and a Rust march
//! that bracket the same function place the same root. Moved out of `lti::guard_scan` on 2026-11-19.

use std::fmt;

/// Why Brent's method gave no root: the bracket, or the function's own error.
#[derive(Debug, Clone, PartialEq)]
pub enum BrentError<E> {
    /// `f(a)` and `f(b)` do not differ in sign.
    Bracket { left: f64, right: f64 },
    /// The function refused a point.
    Function(E),
}

impl<E: fmt::Display> fmt::Display for BrentError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bracket { left, right } => {
                write!(
                    f,
                    "f(a) and f(b) must have different signs (a = {left:e}, b = {right:e})"
                )
            }
            Self::Function(e) => write!(f, "{e}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for BrentError<E> {}

/// `scipy.optimize.brentq` (scipy's C `brentq`) with its `maxiter`: `(root, converged)`.
pub fn brentq<F, E>(
    mut f: F,
    xa: f64,
    xb: f64,
    xtol: f64,
    rtol: f64,
    maxiter: usize,
) -> Result<(f64, bool), BrentError<E>>
where
    F: FnMut(f64) -> Result<f64, E>,
{
    let (mut xpre, mut xcur) = (xa, xb);
    let (mut xblk, mut fblk, mut spre, mut scur) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut fpre = f(xpre).map_err(BrentError::Function)?;
    let mut fcur = f(xcur).map_err(BrentError::Function)?;
    if fpre == 0.0 {
        return Ok((xpre, true));
    }
    if fcur == 0.0 {
        return Ok((xcur, true));
    }
    if fpre.is_sign_negative() == fcur.is_sign_negative() {
        return Err(BrentError::Bracket {
            left: xa,
            right: xb,
        });
    }
    for _ in 0..maxiter {
        if fpre != 0.0 && fcur != 0.0 && fpre.is_sign_negative() != fcur.is_sign_negative() {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return Ok((xcur, true));
        }
        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur).map_err(BrentError::Function)?;
    }
    Ok((xcur, false))
}
