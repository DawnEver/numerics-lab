//! The PCHIP slope rule (scipy `PchipInterpolator._find_derivatives`).

/// Non-centered 3-point boundary derivative for PCHIP, monotonicity-clamped.
///
/// This is the rule SciPy's `PchipInterpolator._edge_case` implements -- Cleve Moler's one-sided
/// three-point estimate from *Numerical Computing with MATLAB* ch. 3.6 (`pchiptx.m`), which
/// scipy cites by name in that function. It matches the last three points exactly when they
/// happen to lie on a parabola, then applies TWO clamps: sign disagreement with the adjacent
/// secant collapses it to `0`, and an edge steeper than `3|d0|` where the two secants disagree
/// in sign is capped at `3*d0`.
///
/// The `d0 == 0` early return reproduces scipy's `sign(d) != sign(m0)` mask for `m0 == 0`
/// (`sign(0) == 0`, so any non-zero `d` is masked to `0`, and `d == 0` is already `0`).
fn pchip_edge_slope(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    if d0 == 0.0 {
        return 0.0;
    }
    let m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (m > 0.0) != (d0 > 0.0) {
        return 0.0;
    }
    if (d0 > 0.0) != (d1 > 0.0) && m.abs() > 3.0 * d0.abs() {
        return 3.0 * d0;
    }
    m
}

/// Fritsch-Carlson monotone cubic Hermite slopes `m` at each knot of `(x, y)`.
///
/// Interior: the weighted harmonic mean `(w1 + w2) / (w1/d0 + w2/d1)` with `w1 = 2h_k + h_{k-1}`
/// and `w2 = h_k + 2h_{k-1}`, zeroed wherever the two adjacent secants disagree in sign or
/// either is zero. Ends: [`pchip_edge_slope`]. `n == 2` degrades to the single secant, as scipy
/// does.
///
/// The association is the PYTHON's -- `(w1 + w2) / (w1/d0 + w2/d1)` -- not scipy's
/// `1.0 / ((w1/d0 + w2/d1) / (w1 + w2))`. Same value, one fewer rounding: within the `1e-12` bar of
/// `tests/golden_interp.rs`, and the exact form motronics' B-H golden (its consumer) is pinned to.
pub fn pchip_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut m = vec![0.0; n];
    if n < 2 {
        return m;
    }
    let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let delta: Vec<f64> = y
        .windows(2)
        .zip(&h)
        .map(|(w, hk)| (w[1] - w[0]) / hk)
        .collect();
    if n == 2 {
        m[0] = delta[0];
        m[1] = delta[0];
        return m;
    }
    for k in 1..n - 1 {
        let (d0, d1) = (delta[k - 1], delta[k]);
        if (d0 > 0.0) == (d1 > 0.0) && d0 != 0.0 && d1 != 0.0 {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            m[k] = (w1 + w2) / (w1 / d0 + w2 / d1);
        }
    }
    m[0] = pchip_edge_slope(h[0], h[1], delta[0], delta[1]);
    m[n - 1] = pchip_edge_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
    m
}

#[cfg(test)]
mod tests;
