//! `numerics::interp` against SciPy (`PchipInterpolator`, `PPoly.antiderivative`, the `k=1` spline) through the
//! committed `goldens/interp.jsonl` that `tests/unit/test_interp_and_nnls_port_golden.py` freezes from live SciPy.
//!
//! Tolerance: `1e-12` relative, plus a dust floor of `1e-12` times the largest magnitude in the compared series
//! (exact zeros -- a clamped slope, a vanished derivative -- make a pure relative test meaningless). Unitless: the
//! interpolant carries whatever unit its table does. Planted: one moved coefficient is seen.

use numerics::interp::{pchip_slopes, PiecewisePoly};
use serde_json::Value;

const RTOL: f64 = 1e-12;

fn golden() -> Value {
    serde_json::from_str(include_str!("goldens/interp.jsonl")).expect("goldens/interp.jsonl")
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

/// `None` when every entry agrees; else the first disagreement.
fn disagreement(got: &[f64], want: &[f64]) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("length {} vs {}", got.len(), want.len()));
    }
    let scale = want.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    got.iter().zip(want).enumerate().find_map(|(i, (g, w))| {
        let tol = RTOL * w.abs() + RTOL * scale;
        ((g - w).abs() > tol).then(|| format!("[{i}] got {g:e}, want {w:e} (tol {tol:e})"))
    })
}

fn check(got: &[f64], want: &[f64], what: &str) {
    if let Some(d) = disagreement(got, want) {
        panic!("{what}: {d}");
    }
}

fn check_poly(p: &PiecewisePoly, entry: &Value, query: &[f64], what: &str) {
    let c = entry["c"].as_array().expect("c");
    assert_eq!(p.coefficients().len(), c.len(), "{what}: interval count");
    for (i, (got, want)) in p.coefficients().iter().zip(c).enumerate() {
        check(got, &nums(want), &format!("{what}.c[{i}]"));
    }
    for nu in 0..4 {
        let got: Vec<f64> = query.iter().map(|&q| p.eval(q, nu)).collect();
        check(
            &got,
            &nums(&entry["eval"][nu.to_string()]),
            &format!("{what}.eval nu={nu}"),
        );
    }
    let f = p.antiderivative();
    let got: Vec<f64> = query.iter().map(|&q| f.eval(q, 0)).collect();
    check(
        &got,
        &nums(&entry["antiderivative"]),
        &format!("{what}.antiderivative"),
    );
}

fn cases() -> Vec<Value> {
    let g = golden();
    let cases = g["cases"].as_array().expect("cases").clone();
    assert!(
        cases.len() >= 10,
        "a golden with fewer tables than the slope rule has branches proves little"
    );
    cases
}

#[test]
fn the_pchip_slopes_are_scipys() {
    for c in cases() {
        let name = c["name"].as_str().expect("name");
        check(
            &pchip_slopes(&nums(&c["x"]), &nums(&c["y"])),
            &nums(&c["slopes"]),
            name,
        );
    }
}

#[test]
fn the_pchip_polynomial_its_derivatives_and_antiderivative_are_scipys() {
    for c in cases() {
        let name = c["name"].as_str().expect("name");
        let (x, y, q) = (nums(&c["x"]), nums(&c["y"]), nums(&c["query"]));
        check_poly(
            &PiecewisePoly::pchip(&x, &y),
            &c["pchip"],
            &q,
            &format!("{name}.pchip"),
        );
    }
}

#[test]
fn the_linear_polynomial_and_its_antiderivative_are_scipys() {
    for c in cases() {
        let name = c["name"].as_str().expect("name");
        let (x, y, q) = (nums(&c["x"]), nums(&c["y"]), nums(&c["query"]));
        check_poly(
            &PiecewisePoly::linear(&x, &y),
            &c["linear"],
            &q,
            &format!("{name}.linear"),
        );
    }
}

#[test]
fn planted_a_moved_coefficient_is_seen() {
    let c = &cases()[0];
    let p = PiecewisePoly::pchip(&nums(&c["x"]), &nums(&c["y"]));
    let mut want = nums(&c["pchip"]["c"][1]);
    want[0] *= 1.0 + 1e-9;
    assert!(disagreement(&p.coefficients()[1], &want).is_some());
}
