//! Unit tests for the PCHIP slope rule (carried from motronics-studio `core_num::bh::tests`, 2026-11-19).
//!
//! These are the claims that do NOT need the golden: the PCHIP derivative RULE tested against
//! its own definition rather than against interpolated values.
//!
//! Why the derivative rule gets its own tests. A golden of `y(x)` at sample points can pass
//! while the derivative construction is wrong -- the Hermite basis reproduces the KNOT VALUES
//! exactly for ANY choice of `m`, so an interior sample only ever sees a blend of the two
//! adjacent slopes, and a derivative consumer depends on `m` directly. So `m` is asserted against the
//! Fritsch-Carlson formula written out independently here, and against the properties that
//! formula was chosen FOR.

use super::pchip_slopes;

/// Relative tolerance for a claim about closed-form arithmetic evaluated two ways.
const TOL: f64 = 1e-12;

fn close(got: f64, want: f64, what: &str) {
    let tol = TOL * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:e}, want {want:e} (tol {tol:e})"
    );
}

/// A short synthetic table with NON-UNIFORM spacing, strictly increasing and convex.
///
/// Non-uniform on purpose: with uniform `h` the weights `w1` and `w2` are equal and the
/// weighted harmonic mean collapses to the plain one, so a port that dropped the weights
/// entirely would pass. Unitless.
fn synthetic() -> (Vec<f64>, Vec<f64>) {
    let x = vec![0.0, 0.2, 0.5, 0.9, 1.4, 1.6, 1.9];
    let y = vec![0.0, 40.0, 120.0, 300.0, 1200.0, 3000.0, 12000.0];
    (x, y)
}

// ---------------------------------------------------------------------------------------------
// The derivative rule, INDEPENDENT of any interpolated value.
// ---------------------------------------------------------------------------------------------

/// The interior rule is the WEIGHTED harmonic mean, written out here from its definition.
///
/// `1/d_k = (w1/m_{k-1} + w2/m_k) / (w1 + w2)` with `w1 = 2h_k + h_{k-1}`,
/// `w2 = h_k + 2h_{k-1}` -- scipy's `_find_derivatives`, quoted in its own comment. Computed
/// here from the raw knots with no call into the implementation's helpers.
#[test]
fn the_interior_slopes_are_the_weighted_harmonic_mean_of_the_adjacent_secants() {
    let (x, y) = synthetic();
    let m = pchip_slopes(&x, &y);
    let n = x.len();
    for k in 1..n - 1 {
        let (hkm, hk) = (x[k] - x[k - 1], x[k + 1] - x[k]);
        let d0 = (y[k] - y[k - 1]) / hkm;
        let d1 = (y[k + 1] - y[k]) / hk;
        let (w1, w2) = (2.0 * hk + hkm, hk + 2.0 * hkm);
        let want = 1.0 / ((w1 / d0 + w2 / d1) / (w1 + w2));
        close(m[k], want, &format!("interior slope at knot {k}"));
    }
    // A floor: the weights must actually DIFFER somewhere, or the test above proves only the
    // unweighted mean.
    let differs = (1..n - 1).any(|k| {
        let (hkm, hk) = (x[k] - x[k - 1], x[k + 1] - x[k]);
        (hk - hkm).abs() > 1e-9
    });
    assert!(differs, "the fixture must have non-uniform spacing");
}

/// With UNIFORM spacing the rule degenerates to the plain harmonic mean `2 d0 d1 / (d0 + d1)`.
///
/// A second, independent statement of the same construction: it fixes the 2/1 weight RATIO,
/// which the general formula above would still satisfy if both weights were scaled together.
#[test]
fn uniform_spacing_reduces_the_rule_to_the_plain_harmonic_mean() {
    let x = vec![0.0, 0.25, 0.5, 0.75];
    let y = vec![0.0, 10.0, 45.0, 200.0];
    let m = pchip_slopes(&x, &y);
    for k in 1..3 {
        let d0 = (y[k] - y[k - 1]) / 0.25;
        let d1 = (y[k + 1] - y[k]) / 0.25;
        close(
            m[k],
            2.0 * d0 * d1 / (d0 + d1),
            &format!("harmonic mean at {k}"),
        );
    }
}

/// The ENDPOINT rule: scipy's `_edge_case`, the one-sided three-point estimate from Cleve
/// Moler's `pchiptx.m` (*Numerical Computing with MATLAB* ch. 3.6).
///
/// Its defining property, asserted here rather than the formula restated: when the three
/// points nearest an end lie EXACTLY on a parabola, the edge slope is that parabola's own
/// derivative at the end point. Both ends. The fixture is a monotone-increasing parabola over
/// its span so neither clamp fires -- checked below.
#[test]
fn the_endpoint_rule_is_the_three_point_parabola_derivative() {
    // y = 3x^2 + 2x + 1 on x = 0, 0.3, 1.0 (deliberately non-uniform).
    let f = |x: f64| 3.0 * x * x + 2.0 * x + 1.0;
    let fp = |x: f64| 6.0 * x + 2.0;
    let x = vec![0.0, 0.3, 1.0];
    let y: Vec<f64> = x.iter().map(|&v| f(v)).collect();
    let m = pchip_slopes(&x, &y);
    close(m[0], fp(0.0), "left endpoint slope");
    close(m[2], fp(1.0), "right endpoint slope");
    // The control: neither clamp fired. Both secants are positive (so the sign clamp cannot
    // apply) and each edge slope is within 3x its adjacent secant.
    let d0 = (y[1] - y[0]) / 0.3;
    let d1 = (y[2] - y[1]) / 0.7;
    assert!(
        d0 > 0.0 && d1 > 0.0,
        "the fixture keeps both secants positive"
    );
    assert!(m[0].abs() <= 3.0 * d0.abs() && m[2].abs() <= 3.0 * d1.abs());
}

/// A parabola with a genuinely non-uniform spacing is not reproduced by the CENTERED estimate,
/// so the test above is about the ONE-SIDED rule and not about any three-point formula.
///
/// This is the floor under it: the centered difference `(y2 - y0)/(x2 - x0)` disagrees with the
/// asserted value, so a port that used it would red.
#[test]
fn the_endpoint_rule_is_not_the_centered_difference() {
    let f = |x: f64| 3.0 * x * x + 2.0 * x + 1.0;
    let x = [0.0, 0.3, 1.0];
    let y: [f64; 3] = [f(x[0]), f(x[1]), f(x[2])];
    let centered = (y[2] - y[0]) / (x[2] - x[0]);
    let m = pchip_slopes(&x, &y);
    assert!(
        (m[0] - centered).abs() > 1.0,
        "the fixture must distinguish the one-sided rule from the centered one"
    );
}

/// Clamp 1: an edge estimate whose SIGN disagrees with the adjacent secant collapses to zero.
#[test]
fn an_edge_slope_that_disagrees_in_sign_with_its_secant_is_zeroed() {
    // Rising then rising much harder: the one-sided estimate at the left end goes NEGATIVE.
    let x = [0.0, 1.0, 1.1];
    let y = [0.0, 1.0, 100.0];
    let (h0, h1) = (1.0_f64, 0.1_f64);
    let (d0, d1) = (1.0_f64, 990.0_f64);
    let raw = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    assert!(
        raw < 0.0 && d0 > 0.0,
        "the fixture must trip the sign clamp"
    );
    let m = pchip_slopes(&x, &y);
    assert_eq!(m[0], 0.0, "the sign clamp must zero it");
}

/// Clamp 2: with the two secants of OPPOSITE sign, an edge steeper than `3|d0|` caps at `3*d0`.
#[test]
fn an_over_steep_edge_with_opposing_secants_caps_at_three_times_the_secant() {
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 1.0, -20.0];
    let (d0, d1) = (1.0_f64, -21.0_f64);
    let raw = ((2.0 + 1.0) * d0 - 1.0 * d1) / 2.0;
    assert!(
        (d0 > 0.0) != (d1 > 0.0) && raw.abs() > 3.0 * d0.abs() && (raw > 0.0) == (d0 > 0.0),
        "the fixture must reach the 3*d0 cap and not the sign clamp first"
    );
    let m = pchip_slopes(&x, &y);
    close(m[0], 3.0 * d0, "the capped edge slope");
}

/// A secant of zero, or two secants disagreeing in sign, zeroes the INTERIOR slope -- the rule
/// that makes the interpolant monotone.
#[test]
fn a_flat_or_reversing_interior_knot_gets_a_zero_slope() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let y = [0.0, 5.0, 5.0, 9.0];
    let m = pchip_slopes(&x, &y);
    assert_eq!(m[1], 0.0, "knot 1 has a zero secant to its right");
    assert_eq!(m[2], 0.0, "knot 2 has a zero secant to its left");
    let reversing = pchip_slopes(&[0.0, 1.0, 2.0], &[0.0, 5.0, 1.0]);
    assert_eq!(reversing[1], 0.0, "a local maximum gets a zero slope");
}

/// Two knots degrade to the single secant at BOTH ends -- scipy's `y.shape[0] == 2` branch.
#[test]
fn two_knots_are_the_single_secant() {
    let m = pchip_slopes(&[0.5, 1.5], &[100.0, 400.0]);
    assert_eq!(m, vec![300.0, 300.0]);
}

/// The property the rule was CHOSEN for, on a real-shaped monotone table: no slope is negative,
/// so `y` never turns over between knots.
#[test]
fn a_monotone_table_gets_no_negative_slope() {
    let (x, y) = synthetic();
    let m = pchip_slopes(&x, &y);
    assert!(m.iter().all(|&v| v >= 0.0), "slopes: {m:?}");
    assert!(m.iter().any(|&v| v > 0.0), "an all-zero result is vacuous");
}
