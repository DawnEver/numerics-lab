//! `numerics::dense::expm::expm` against `scipy.linalg.expm` (SciPy 1.18.1 over its bundled OpenBLAS, SKYLAKEX) BIT FOR BIT
//! on the fixture motronics-studio's `tests/unit/laplace/control/test_the_native_expm_is_scipys_bit_for_bit.py` wrote (carried over byte-identical, 2026-11-19) and re-checks
//! against live SciPy: every matrix size 2..12, every Pade degree, dense and triangular, stiff, nilpotent, and the
//! held march's own matrices. Sizes past `MAX_N` (16) are not bit-identical: they must agree to round-off
//! (`PAST_MAX_N_REL`), since a march larger than sixteen states (an MMC phase) has no other exponential. Planted: one
//! input ulp moved is seen.

use numerics::dense::expm::{expm, MAX_N};

/// Past `MAX_N`, the largest entry error relative to the exponential's largest entry (measured 4..512 ulp on 20..33).
const PAST_MAX_N_REL: f64 = 1e-12;

struct Case {
    name: String,
    n: usize,
    a: Vec<f64>,
    e: Vec<f64>,
}

fn bits(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .expect("a hex list")
        .iter()
        .map(|h| f64::from_bits(u64::from_str_radix(h.as_str().expect("hex"), 16).expect("a u64")))
        .collect()
}

fn cases() -> Vec<Case> {
    let text = include_str!("fixtures/expm_scipy.json");
    let rows: Vec<serde_json::Value> = serde_json::from_str(text).expect("the fixture");
    rows.iter()
        .map(|r| Case {
            name: r["name"].as_str().expect("name").to_owned(),
            n: usize::try_from(r["n"].as_u64().expect("n")).expect("n"),
            a: bits(&r["a"]),
            e: bits(&r["e"]),
        })
        .collect()
}

fn misses(moved: bool) -> Vec<String> {
    let mut out = Vec::new();
    for c in cases() {
        let mut a = c.a.clone();
        if moved {
            a[0] = f64::from_bits(a[0].to_bits() + 1);
        }
        if c.n > MAX_N {
            // Past the measured identity the same transcription runs and is SciPy's to round-off, not to the bit:
            // every entry within `PAST_MAX_N_REL` of the matrix's largest (dimensionless, as the exponential is).
            let got = expm(&a, c.n).expect("an exponential past MAX_N");
            let scale = c.e.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let worst = got
                .iter()
                .zip(&c.e)
                .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
            if !(worst <= PAST_MAX_N_REL * scale) {
                out.push(format!(
                    "{}: past MAX_N off by {worst:e} of {scale:e}",
                    c.name
                ));
            }
            continue;
        }
        let got = expm(&a, c.n).expect("an exponential");
        let worst = got
            .iter()
            .zip(&c.e)
            .map(|(x, y)| x.to_bits().abs_diff(y.to_bits()))
            .max()
            .unwrap_or(0);
        if worst != 0 {
            let k = got
                .iter()
                .zip(&c.e)
                .position(|(x, y)| x.to_bits() != y.to_bits())
                .unwrap_or(0);
            out.push(format!(
                "{}: {worst} ulp (first at {k}: {:e} vs {:e})",
                c.name, got[k], c.e[k]
            ));
        }
    }
    out
}

#[test]
fn every_fixture_exponential_is_scipys_bit_for_bit() {
    let m = misses(false);
    assert!(m.is_empty(), "{} misses: {}", m.len(), m.join("; "));
}

#[test]
fn planted_past_max_n_a_relative_move_of_1e_10_is_seen() {
    let past: Vec<Case> = cases().into_iter().filter(|c| c.n > MAX_N).collect();
    assert!(!past.is_empty(), "the fixture carries sizes past MAX_N");
    for c in past {
        let got = expm(&c.a, c.n).expect("an exponential past MAX_N");
        let scale = c.e.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&c.e)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y * (1.0 + 1e-10)).abs()));
        assert!(
            worst > PAST_MAX_N_REL * scale,
            "{}: a 1e-10 move passes",
            c.name
        );
    }
}

#[test]
fn planted_a_one_ulp_input_move_is_seen() {
    let moved = misses(true);
    let dense = moved
        .iter()
        .filter(|m| m.starts_with("dense") && !m.contains("past MAX_N"))
        .count();
    // 8 sizes x 10 scales of dense matrices are transcribed; a one-ulp entry moves half of them (40 measured)
    assert!(dense >= 30, "{dense}: {}", moved.join("; "));
}
