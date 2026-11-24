//! The GRID-LATTICE option of [`AffineGuards`]: on a scan grid's uniform runs it propagates `y_{k+1} = E_h y_k` with
//! one `E_h = expm(M h)` per run instead of one `expm` per read. It must read what the per-step path reads, pointwise,
//! on uniform AND non-uniform grids (a run splits where the step changes), and the comparison must be able to say no.
//!
//! Units: time [s], `system` [1/s]; each guard value is in the guard's own coordinate `[g]`, so the tolerance
//! `READ_ABS_G` is in `[g]` against guards of order one.

use numerics::dense::expm::expm;
use numerics::lti::guard_scan::{scan_crossings, AffineGuards, History, Scan};
use std::time::Instant;

/// Pointwise bound [g] between the lattice and the per-step reads (round-off of a few hundred matvecs on O(1) guards).
const READ_ABS_G: f64 = 1e-10;

/// Two damped oscillators (block diagonal, 4 states) [1/s], read by two guards.
fn plant() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    #[rustfmt::skip]
    let system = vec![
        -0.3, 6.0, 0.0, 0.0,
        -6.0, -0.3, 0.0, 0.0,
        0.0, 0.0, -0.1, 2.5,
        0.0, 0.0, -2.5, -0.1,
    ];
    let state = vec![1.0, 0.0, 0.5, -0.2];
    #[rustfmt::skip]
    let guards = vec![
        1.0, 0.0, 0.3, 0.0,
        0.0, 0.5, 0.0, 1.0,
    ];
    (system, state, guards)
}

fn per_step() -> AffineGuards {
    let (m, y, g) = plant();
    AffineGuards::new(m, y, g, vec![None, None], 0.0).unwrap()
}

fn lattice(grid: &[f64]) -> AffineGuards {
    per_step().with_grid_lattice(grid)
}

fn uniform(start: f64, step: f64, count: usize) -> Vec<f64> {
    (0..count).map(|k| start + step * k as f64).collect()
}

/// Largest pointwise deviation [g] between two read sequences.
fn max_deviation(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
        .fold(0.0, f64::max)
}

fn reads(guards: &mut AffineGuards, grid: &[f64]) -> Vec<Vec<f64>> {
    grid.iter().map(|&t| guards.read(t).unwrap()).collect()
}

#[test]
fn a_uniform_grid_reads_the_per_step_values_with_one_step_exponential() {
    let grid = uniform(0.0, 0.01, 400);
    let exact = reads(&mut per_step(), &grid);
    let mut fast = lattice(&grid);
    let got = reads(&mut fast, &grid);
    let dev = max_deviation(&exact, &got);
    assert!(dev <= READ_ABS_G, "deviation {dev:e} g");
    // grid[0] is read exactly (one expm), then ONE E_h for the whole run.
    assert_eq!(fast.expm_calls(), 2);
}

#[test]
fn a_step_change_splits_the_run_and_a_ragged_stretch_falls_back_per_step() {
    // Run A (h = 0.01), run B (h = 0.004), then a ragged stretch where no two steps agree.
    let mut grid = uniform(0.0, 0.01, 100);
    let t = *grid.last().unwrap();
    grid.extend(uniform(t + 0.004, 0.004, 100));
    let mut t = *grid.last().unwrap();
    for k in 1..=10 {
        t += 0.001 * f64::from(k);
        grid.push(t);
    }
    let exact = reads(&mut per_step(), &grid);
    let mut fast = lattice(&grid);
    let got = reads(&mut fast, &grid);
    let dev = max_deviation(&exact, &got);
    assert!(dev <= READ_ABS_G, "deviation {dev:e} g");
    // The exact start, one E_h per run (2), one exact expm per ragged point (10).
    assert_eq!(fast.expm_calls(), 1 + 2 + 10);
}

#[test]
fn off_grid_reads_stay_exact() {
    let grid = uniform(0.0, 0.01, 50);
    let mut fast = lattice(&grid);
    let mut exact = per_step();
    for &t in &[0.123_456, 0.25, 0.333] {
        let dev = max_deviation(&[exact.read(t).unwrap()], &[fast.read(t).unwrap()]);
        assert!(dev <= READ_ABS_G, "t = {t}: deviation {dev:e} g");
    }
}

#[test]
fn planted_a_wrong_step_exponential_is_caught_by_the_comparison() {
    // Propagate with E_{h'} for h' = 1.001 h: the comparison at READ_ABS_G must say NO.
    let (m, y0, g) = plant();
    let h = 0.01;
    let grid = uniform(0.0, h, 400);
    let scaled: Vec<f64> = m.iter().map(|v| v * h * 1.001).collect();
    let e = expm(&scaled, 4).unwrap();
    let mut y = y0;
    let mut wrong = Vec::new();
    for _ in &grid {
        wrong.push(
            (0..2)
                .map(|r| (0..4).map(|k| g[r * 4 + k] * y[k]).sum())
                .collect::<Vec<f64>>(),
        );
        y = (0..4)
            .map(|i| (0..4).map(|k| e[i * 4 + k] * y[k]).sum())
            .collect();
    }
    let exact = reads(&mut per_step(), &grid);
    let dev = max_deviation(&exact, &wrong);
    assert!(
        dev > READ_ABS_G,
        "the planted wrong h passed: deviation {dev:e} g"
    );
}

fn scan_of() -> (Vec<History>, Vec<f64>) {
    let unarmed = History {
        amplitude: 0.0,
        sign: 0,
        after_hit: false,
        inherited: 0,
    };
    (vec![unarmed; 2], vec![0.4, 0.4])
}

#[test]
fn a_scan_finds_the_same_crossing_on_either_path() {
    let grid = uniform(0.0, 0.01, 400);
    let (histories, thresholds) = scan_of();
    let scan = Scan {
        grid: &grid,
        histories: &histories,
        thresholds: &thresholds,
        tie: 1e-9,
        stride: 1,
    };
    let a = scan_crossings(&mut per_step(), &scan).unwrap();
    let b = scan_crossings(&mut lattice(&grid), &scan).unwrap();
    assert_eq!(a.crossings, b.crossings);
    assert_eq!(a.sides, b.sides);
    assert!(
        (a.time - b.time).abs() <= 1e-9,
        "{} vs {} s",
        a.time,
        b.time
    );
    assert!(!a.crossings.is_empty());
}

/// The measurement behind the option (`cargo test --release -p numerics --test guard_scan_grid_lattice -- --ignored
/// --nocapture`): the same scans, timed on both paths.
#[test]
#[ignore = "timing measurement; run in --release with --ignored --nocapture"]
fn measure_lattice_against_per_step() {
    let grid = uniform(0.0, 0.002, 2000);
    let (histories, near) = scan_of();
    // A threshold no guard reaches walks the whole grid (the scan's worst case); `near` stops at the first arrival.
    let far = vec![10.0, 10.0];
    for (label, thresholds) in [("no arrival", &far), ("first arrival", &near)] {
        let scan = Scan {
            grid: &grid,
            histories: &histories,
            thresholds,
            tie: 1e-9,
            stride: 1,
        };
        let repeats = 200;
        let time = |make: &dyn Fn() -> AffineGuards| {
            let t0 = Instant::now();
            for _ in 0..repeats {
                std::hint::black_box(scan_crossings(&mut make(), &scan).unwrap());
            }
            t0.elapsed().as_secs_f64() * 1e3
        };
        let slow = time(&per_step);
        let fast = time(&|| lattice(&grid));
        println!(
            "{label}: {repeats} scans x {} points: per-step {slow:.1} ms, grid-lattice {fast:.1} ms, ratio {:.2}x",
            grid.len(),
            slow / fast
        );
    }
}
