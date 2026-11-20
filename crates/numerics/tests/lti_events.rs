//! The exact-trajectory event kernels (`numerics::lti`), moved from motronics-studio's `laplace_net`
//! (a74923ef38) on 2026-11-19, reached through the public API: an armed guard scan and a first crossing
//! each find the zero of `1 - t` on the exponential trajectory, and a segment that ends early finds none.

use numerics::dense::matrix::Dense;
use numerics::lti::first_crossing::{first_crossing, lattice, Propagators, Segment};
use numerics::lti::guard_scan::{scan_crossings, AffineGuards, History, Scan};

#[test]
fn an_armed_guard_scan_arrives_at_its_threshold() {
    let mut guards = AffineGuards::new(
        vec![0.0, 1.0, 0.0, 0.0],
        vec![1.0, -1.0],
        vec![1.0, 0.0],
        vec![None],
        0.0,
    )
    .unwrap();
    let grid: Vec<f64> = (0..=16).map(|k| 2.0 * f64::from(k) / 16.0).collect();
    let armed = History {
        amplitude: 1.0,
        sign: 1,
        after_hit: false,
        inherited: 0,
    };
    let scan = Scan {
        grid: &grid,
        histories: &[armed],
        thresholds: &[1e-6],
        tie: 1e-9,
        stride: 1,
    };
    let found = scan_crossings(&mut guards, &scan).unwrap();
    assert_eq!(found.crossings, vec![0]);
    assert!((found.time - (1.0 + 1e-6)).abs() <= 1e-9, "{}", found.time);
}

fn ramp(span: f64) -> Option<(f64, Vec<usize>)> {
    let system = Dense::new(2, 2, vec![0.0, -1.0, 0.0, 0.0]).unwrap();
    let grid = lattice(&[0.0, 0.0], &[0.0, 0.0], span);
    let rows = Dense::new(1, 2, vec![1.0, 0.0]).unwrap();
    let cache = Propagators::default();
    let seg = Segment {
        system: &system,
        grid: &grid,
        za: &[1.0, 1.0],
        span,
        rows: &rows,
        offsets: &[0.0],
        band: &[1e-9],
        propagators: &cache,
    };
    first_crossing(&seg).unwrap()
}

#[test]
fn a_falling_row_first_crosses_at_its_zero() {
    let (time, causes) = ramp(2.0).unwrap();
    assert_eq!(causes, vec![0]);
    assert!((time - 1.0).abs() <= 1e-12, "{time}");
}

#[test]
fn planted_a_segment_that_ends_before_the_zero_finds_none() {
    assert!(ramp(0.9).is_none());
}
