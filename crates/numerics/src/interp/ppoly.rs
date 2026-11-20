//! A piecewise polynomial in SciPy `PPoly`'s local power basis -- evaluation, derivatives and the
//! exact antiderivative.
//!
//! Interval `i` holds `p_i(x) = sum_k c[i][k] (x - x_i)^(order-1-k)` (highest power first, scipy's
//! layout). Built from the ONE PCHIP slope rule ([`super::pchip_slopes`]) with scipy
//! `CubicHermiteSpline`'s coefficient formula, or from straight segments. Pure numerics: no unit,
//! no material.

use super::pchip_slopes;

/// Piecewise polynomial over strictly ascending breakpoints `x`.
#[derive(Clone, Debug, PartialEq)]
pub struct PiecewisePoly {
    x: Vec<f64>,
    c: Vec<Vec<f64>>,
}

impl PiecewisePoly {
    /// The monotone PCHIP through `(x, y)`, as scipy's `PchipInterpolator`.
    #[must_use]
    pub fn pchip(x: &[f64], y: &[f64]) -> Self {
        let m = pchip_slopes(x, y);
        let c = (0..x.len() - 1)
            .map(|i| {
                let dx = x[i + 1] - x[i];
                let slope = (y[i + 1] - y[i]) / dx;
                let t = (m[i] + m[i + 1] - 2.0 * slope) / dx;
                vec![t / dx, (slope - m[i]) / dx - t, m[i], y[i]]
            })
            .collect();
        Self { x: x.to_vec(), c }
    }

    /// Straight segments through `(x, y)`.
    #[must_use]
    pub fn linear(x: &[f64], y: &[f64]) -> Self {
        let c = (0..x.len() - 1)
            .map(|i| vec![(y[i + 1] - y[i]) / (x[i + 1] - x[i]), y[i]])
            .collect();
        Self { x: x.to_vec(), c }
    }

    /// The breakpoints.
    #[must_use]
    pub fn breakpoints(&self) -> &[f64] {
        &self.x
    }

    /// Each interval's coefficients, highest power first (scipy `PPoly.c[:, i]`).
    #[must_use]
    pub fn coefficients(&self) -> &[Vec<f64>] {
        &self.c
    }

    /// The leading (highest-power) coefficient of each interval.
    #[must_use]
    pub fn leading(&self) -> Vec<f64> {
        self.c.iter().map(|c| c[0]).collect()
    }

    /// scipy's interval: `x_i <= xq < x_{i+1}`, the last interval at and past the right end, the
    /// first before the left.
    fn interval(&self, xq: f64) -> usize {
        self.x
            .partition_point(|&v| v <= xq)
            .clamp(1, self.x.len() - 1)
            - 1
    }

    /// The `nu`-th derivative at `xq` (Horner in the local coordinate, as scipy evaluates).
    #[must_use]
    pub fn eval(&self, xq: f64, nu: usize) -> f64 {
        let i = self.interval(xq);
        let c = &self.c[i];
        let order = c.len();
        if nu >= order {
            return 0.0;
        }
        let dx = xq - self.x[i];
        let mut acc = 0.0;
        for (k, &ck) in c[..order - nu].iter().enumerate() {
            let power = order - 1 - k;
            let factor: f64 = (power - nu + 1..=power).map(|p| p as f64).product();
            acc = acc * dx + ck * factor;
        }
        acc
    }

    /// The exact antiderivative, zero at `x_0` and continuous across every breakpoint (scipy's
    /// `PPoly.antiderivative`: coefficients over their new powers, constants fixed left to right).
    #[must_use]
    pub fn antiderivative(&self) -> Self {
        let mut c: Vec<Vec<f64>> = self
            .c
            .iter()
            .map(|ci| {
                let order = ci.len();
                let mut out: Vec<f64> = ci
                    .iter()
                    .enumerate()
                    .map(|(k, &v)| v / (order - k) as f64)
                    .collect();
                out.push(0.0);
                out
            })
            .collect();
        for i in 1..c.len() {
            let dx = self.x[i] - self.x[i - 1];
            let end = c[i - 1].iter().fold(0.0, |acc, &v| acc * dx + v);
            *c[i].last_mut().expect("a constant term") = end;
        }
        Self {
            x: self.x.clone(),
            c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_antiderivative_differentiates_back_and_is_continuous() {
        let x = [0.0, 0.5, 1.5, 2.0];
        let y = [0.0, 0.4, 1.9, 2.1];
        let p = PiecewisePoly::pchip(&x, &y);
        let f = p.antiderivative();
        for &q in &[0.1, 0.5, 0.9, 1.5, 1.99] {
            assert!((f.eval(q, 1) - p.eval(q, 0)).abs() < 1e-14);
        }
        let left = f.c[0].iter().fold(0.0, |a, &v| a * 0.5 + v);
        assert!(
            (f.eval(0.5, 0) - left).abs() < 1e-15,
            "a jump at a breakpoint"
        );
    }

    #[test]
    fn the_pchip_passes_through_its_knots_and_a_line_is_a_line() {
        let x = [0.0, 1.0, 3.0];
        let y = [1.0, 2.0, 2.5];
        let p = PiecewisePoly::pchip(&x, &y);
        for (xi, yi) in x.iter().zip(&y) {
            assert!((p.eval(*xi, 0) - yi).abs() < 1e-15);
        }
        let l = PiecewisePoly::linear(&x, &y);
        assert_eq!(l.eval(2.0, 0), 2.25);
        assert_eq!(l.eval(2.0, 2), 0.0);
    }
}
