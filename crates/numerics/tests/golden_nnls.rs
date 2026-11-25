//! `numerics::linear::nnls` against `scipy.optimize.nnls` through the committed `goldens/nnls.jsonl` that
//! `tests/unit/test_interp_and_nnls_port_golden.py` freezes from live SciPy.
//!
//! Both are Lawson-Hanson, but not the same floating-point program, so the bar is round-off, not bits:
//! `x` within `1e-10` of the problem's own scale (`max |x|`, floored at 1), `rnorm` within `1e-10` relative
//! (absolute `1e-12` for an exact fit, in `b`'s unit). The degenerate (duplicate-column) problem has a
//! non-unique `x`; it is still compared entry by entry: SciPy breaks the exact dual tie toward the later column and
//! so does this port. Planted: a moved golden entry is seen.

use numerics::linear::nnls::nnls;
use serde_json::Value;

const XTOL: f64 = 1e-10;

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

fn cases() -> Vec<Value> {
    let g: Value =
        serde_json::from_str(include_str!("goldens/nnls.jsonl")).expect("goldens/nnls.jsonl");
    let cases = g["cases"].as_array().expect("cases").clone();
    assert!(cases.len() >= 8, "the golden lost a problem");
    cases
}

fn x_disagrees(got: &[f64], want: &[f64]) -> bool {
    let scale = want.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    got.len() != want.len()
        || got
            .iter()
            .zip(want)
            .any(|(g, w)| (g - w).abs() > XTOL * scale)
}

fn solve(c: &Value) -> (Vec<f64>, f64) {
    let m = usize::try_from(c["m"].as_u64().expect("m")).expect("m");
    let n = usize::try_from(c["n"].as_u64().expect("n")).expect("n");
    let s = nnls(&nums(&c["a"]), m, n, &nums(&c["b"]), None).expect("solves");
    (s.x, s.rnorm)
}

#[test]
fn every_problem_is_scipys() {
    for c in cases() {
        let name = c["name"].as_str().expect("name");
        let (x, rnorm) = solve(&c);
        let (want_x, want_r) = (nums(&c["x"]), c["rnorm"].as_f64().expect("rnorm"));
        assert!(!x_disagrees(&x, &want_x), "{name}: x {x:?} vs {want_x:?}");
        assert!(
            (rnorm - want_r).abs() <= XTOL * want_r + 1e-12,
            "{name}: rnorm {rnorm:e} vs {want_r:e}"
        );
        assert!(x.iter().all(|&v| v >= 0.0), "{name}: a negative entry");
    }
}

#[test]
fn planted_a_moved_entry_is_seen() {
    let c = &cases()[2];
    let (x, _) = solve(c);
    let mut want = nums(&c["x"]);
    let k = want
        .iter()
        .position(|&v| v > 0.0)
        .expect("a positive entry");
    want[k] += 1e-6;
    assert!(x_disagrees(&x, &want));
}
