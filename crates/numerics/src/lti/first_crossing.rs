//! The FIRST crossing of affine rows `G z + c` along the exact trajectory `z(s) = expm(M s) z0` of one segment: the
//! rows sampled on a spectrum-driven [`lattice`], each row's first flagged interval bracketed through degree-16
//! barycentric interpolants of exact inner samples, placed by [`brentq`], then one exact Newton polish. The
//! exponentials are cached per [`span_key`] ([`Propagators`]).
//!
//! Moved from motronics-studio's `laplace_net::control::first_crossing` (a74923ef38) on 2026-11-19: the generic half.
//! What builds the rows from a circuit's devices (`augmented`, `guard_rows`, `guard_band`) stays there.
//!
//! The algorithm is the Python's line for line: the guards sampled on the exact solution at the lattice, each guard's
//! first flagged interval (an end below `-tol`, or a slope turning `-` to `+`) bracketed through the degree-16
//! barycentric interpolants of its exact inner samples, placed by [`brentq`] with the Python's parameters, then one
//! exact Newton polish.
//!
//! # Equivalence
//!
//! * the lattice: bit for bit, given the spectrum `(|lambda|, Re lambda)` of the system the caller computed
//!   (`np.linalg.eigvals`) -- every step is scalar;
//! * the crossing's identity and side: exact; its instant: within the march's tie [s]. [`crate::dense::expm::expm`] is
//!   `scipy.linalg.expm` bit for bit (up to [`crate::dense::expm::MAX_N`]), and the propagators share the Python's span key
//!   ([`span_key`]); numpy's BLAS still sums `propagator @ z` in its own order, so a sample may differ at round-off.
//!
//! # Units
//!
//! `span`, the lattice and the instant [s]; `system` [1/s]; each guard's row, offset and band in its own coordinate
//! (`a_v` V + `a_i` A).

// The kernels index as the Python they transcribe does, so one reads against the other line by line.
#![allow(clippy::needless_range_loop)]

use crate::dense::expm::expm;
use crate::dense::matrix::{dot, vecmat, Dense, ShapeError};
use crate::nonlinear::roots::{brentq, BrentError};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::convert::Infallible;

/// One sample's `(guard values, guard slopes)`, one entry per guard row.
type Sample = (Vec<f64>, Vec<f64>);

/// `_RADIANS_PER_SAMPLE`: the phase a guard sample may advance by.
pub const RADIANS_PER_SAMPLE: f64 = 0.5;
/// `_DECAYED`: e-foldings after which a decaying mode no longer sets the lattice.
pub const DECAYED: f64 = 40.0;
/// `_BRACKET_NODES`: the intervals a bracketed lattice interval is split into, the interpolant's degree.
pub const BRACKET_NODES: usize = 16;
/// `consistency.CROSSING_XTOL` [s]: the placement's absolute tolerance.
pub const CROSSING_XTOL: f64 = 1e-18;
/// `scipy.optimize.brentq`'s default `xtol` [s], which the slope minimum's bracket uses.
const SCIPY_XTOL: f64 = 2e-12;
/// `scipy.optimize.brentq`'s default `rtol` (`4 eps`), also the placement's.
const BRENT_RTOL: f64 = 4.0 * f64::EPSILON;
/// Roots within this fraction of `max(1, span)` [s] of the first are the same instant.
const SAME_INSTANT: f64 = 1e-15;

/// Why a crossing could not be located; the Python raises the same (`ValueError` / `RuntimeError`).
#[derive(Debug, Clone, PartialEq)]
pub enum CrossingError {
    /// Inconsistent shapes.
    Malformed(String),
    /// A Brent bracket without a sign change.
    Bracket { left: f64, right: f64 },
    /// Brent's method did not converge within its `maxiter`.
    Unconverged { left: f64, right: f64 },
    /// The exponential refused its argument.
    Expm(String),
}

impl std::fmt::Display for CrossingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(why) => write!(f, "first crossing: {why}"),
            Self::Bracket { left, right } => write!(
                f,
                "f(a) and f(b) must have different signs (a = {left:e}, b = {right:e})"
            ),
            Self::Unconverged { left, right } => {
                write!(f, "Failed to converge in [{left:e}, {right:e}]")
            }
            Self::Expm(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CrossingError {}

impl From<ShapeError> for CrossingError {
    fn from(e: ShapeError) -> Self {
        Self::Malformed(e.0)
    }
}

/// The sample lattice over `[0, span]` from the spectrum of the segment's system: `size` `|lambda|`, `rate` `Re lambda`.
#[must_use]
pub fn lattice(size: &[f64], rate: &[f64], span: f64) -> Vec<f64> {
    if span <= 0.0 {
        return vec![0.0];
    }
    let stop = span * (1.0 - 1e-12);
    let mut points = vec![0.0];
    let mut step: Option<f64> = None;
    while let Some(&t) = points.last().filter(|&&t| t < stop) {
        let alive = size
            .iter()
            .zip(rate)
            .filter(|(_, &r)| r >= 0.0 || -r * t < DECAYED)
            .map(|(&s, _)| s)
            .reduce(f64::max);
        let need = match alive {
            Some(m) if m > 0.0 => RADIANS_PER_SAMPLE / m,
            _ => f64::INFINITY,
        };
        let next = step.map_or(need, |p| if 2.0 * p < need { 2.0 * p } else { need });
        step = Some(next);
        points.push(t + next);
    }
    let below = points.iter().take_while(|&&p| p < stop).count();
    let mut out = points[..below].to_vec();
    out.push(span);
    out
}

/// `_BARYCENTRIC`: `(-1)^i C(16, i)`.
fn barycentric() -> [f64; BRACKET_NODES + 1] {
    let mut out = [0.0; BRACKET_NODES + 1];
    let mut binomial = 1.0f64;
    for (i, o) in out.iter_mut().enumerate() {
        *o = if i % 2 == 0 { binomial } else { -binomial };
        binomial = binomial * (BRACKET_NODES - i) as f64 / (i + 1) as f64;
    }
    out
}

/// numpy's pairwise `sum` of a short contiguous array (eight accumulators, then the tail).
fn pairwise(a: &[f64]) -> f64 {
    if a.len() < 8 {
        return a.iter().fold(0.0, |s, x| s + x);
    }
    let mut r = [0.0; 8];
    r.copy_from_slice(&a[..8]);
    let whole = a.len() - a.len() % 8;
    for chunk in a[8..whole].chunks(8) {
        for (x, y) in r.iter_mut().zip(chunk) {
            *x += y;
        }
    }
    let mut out = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for x in &a[whole..] {
        out += x;
    }
    out
}

/// `_interpolant`: the barycentric interpolant on `[0, width]` through exact samples at its equispaced nodes.
struct Interpolant {
    at: [f64; BRACKET_NODES + 1],
    samples: [f64; BRACKET_NODES + 1],
    weights: [f64; BRACKET_NODES + 1],
}

impl Interpolant {
    fn new(
        samples: [f64; BRACKET_NODES + 1],
        width: f64,
        weights: [f64; BRACKET_NODES + 1],
    ) -> Self {
        let step = width / BRACKET_NODES as f64;
        let mut at = [0.0; BRACKET_NODES + 1];
        for (i, a) in at.iter_mut().enumerate() {
            *a = i as f64 * step + 0.0;
        }
        at[BRACKET_NODES] = width;
        Self {
            at,
            samples,
            weights,
        }
    }

    fn read(&self, x: f64) -> f64 {
        let gap: Vec<f64> = self.at.iter().map(|a| x - a).collect();
        if let Some(i) = gap.iter().position(|&g| g == 0.0) {
            return self.samples[i];
        }
        let terms: Vec<f64> = self.weights.iter().zip(&gap).map(|(w, g)| w / g).collect();
        dot(&terms, &self.samples) / pairwise(&terms)
    }
}

/// `brentq` with an infallible function, refused as scipy refuses.
fn located(
    f: impl Fn(f64) -> f64,
    left: f64,
    right: f64,
    xtol: f64,
    maxiter: usize,
) -> Result<f64, CrossingError> {
    let (root, converged) =
        brentq::<_, Infallible>(|x| Ok(f(x)), left, right, xtol, BRENT_RTOL, maxiter).map_err(
            |e| match e {
                BrentError::Bracket { left, right } => CrossingError::Bracket { left, right },
                BrentError::Function(never) => match never {},
            },
        )?;
    if converged {
        Ok(root)
    } else {
        Err(CrossingError::Unconverged { left, right })
    }
}

/// `_crossing`: where `guard` crosses zero downward on `[0, span]`, given it ends below `-tol`.
fn crossing(
    guard: &Interpolant,
    slope: &Interpolant,
    tol: f64,
    span: f64,
) -> Result<f64, CrossingError> {
    let precise = |f: &dyn Fn(f64) -> f64, a: f64, b: f64| located(f, a, b, CROSSING_XTOL, 200);
    let start = guard.read(0.0);
    if start <= -tol {
        return Ok(0.0);
    }
    let past = precise(&|s| guard.read(s) + tol, 0.0, span)?;
    if start <= 0.0 {
        if slope.read(0.0) > 0.0 && slope.read(past) < 0.0 {
            let peak = precise(&|s| slope.read(s), 0.0, past)?;
            if guard.read(peak) > 0.0 {
                return precise(&|s| guard.read(s), peak, past);
            }
        }
        return Ok(past);
    }
    if guard.read(past) < 0.0 {
        precise(&|s| guard.read(s), 0.0, past)
    } else {
        Ok(past)
    }
}

/// `expm(system * span)`.
/// # Errors
/// The exponential refused its argument.
pub fn exponential(system: &Dense, span: f64) -> Result<Dense, CrossingError> {
    let scaled: Vec<f64> = system.data.iter().map(|m| m * span).collect();
    let e = expm(&scaled, system.rows).map_err(|e| CrossingError::Expm(e.to_string()))?;
    Ok(Dense::new(system.rows, system.rows, e)?)
}

/// Significant digits of a propagator's span key: a Python march that keys its own cache through [`span_key`]
/// shares one span per key with this one.
pub const SPAN_KEY_DIGITS: usize = 11;

/// `span` rounded to [`SPAN_KEY_DIGITS`] significant digits: the propagator cache's key. Both directions round
/// correctly (Rust's `{:e}` and `str::parse`).
#[must_use]
pub fn span_key(span: f64) -> f64 {
    format!("{span:.prec$e}", prec = SPAN_KEY_DIGITS - 1)
        .parse()
        .unwrap_or(span)
}

/// One system's propagators: `expm(system * span)` cached per [`span_key`]. A span whose
/// key is cached reads the FIRST span's exponential, as the Python's dict does.
#[derive(Debug, Default)]
pub struct Propagators(std::sync::Mutex<HashMap<u64, Dense>>);

impl Propagators {
    /// The cached exponential of `span`'s key, built from `span` on a miss.
    ///
    /// # Errors
    /// The exponential refused its argument.
    pub fn get(&self, system: &Dense, span: f64) -> Result<Dense, CrossingError> {
        let key = span_key(span).to_bits();
        let mut cache = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(found) = cache.get(&key) {
            return Ok(found.clone());
        }
        let e = exponential(system, span)?;
        cache.insert(key, e.clone());
        Ok(e)
    }
}

/// `_polished`: one exact Newton step from an interpolated zero `x`, kept inside `[0, limit]` and one node spacing.
fn polished(
    system: &Dense,
    zk: &[f64],
    row: &[f64],
    offset: f64,
    x: f64,
    limit: f64,
) -> Result<f64, CrossingError> {
    if x <= 0.0 {
        return Ok(x);
    }
    let state = exponential(system, x)?.apply(zk);
    let value = dot(row, &state) + offset;
    let rate = dot(&vecmat(row, system), &state);
    if rate == 0.0 {
        return Ok(x);
    }
    let moved = x - value / rate;
    Ok(
        if (0.0..=limit).contains(&moved) && (moved - x).abs() < limit / BRACKET_NODES as f64 {
            moved
        } else {
            x
        },
    )
}

/// One segment's rows and the trajectory they are read on.
pub struct Segment<'a> {
    /// `augmented(topo, exo)`, `n x n` [1/s].
    pub system: &'a Dense,
    /// The lattice ([`lattice`]) over `[0, span]` [s].
    pub grid: &'a [f64],
    /// The augmented start state `[xi, w]`.
    pub za: &'a [f64],
    pub span: f64,
    pub rows: &'a Dense,
    pub offsets: &'a [f64],
    pub band: &'a [f64],
    /// This system's propagator cache: the lattice and bracket steps read through it.
    pub propagators: &'a Propagators,
}

/// `(s, rows)` of the EARLIEST guard crossing in `[0, span]`, or `None`.
pub fn first_crossing(seg: &Segment<'_>) -> Result<Option<(f64, Vec<usize>)>, CrossingError> {
    let (g, n) = (seg.rows, seg.za.len());
    if g.rows == 0 {
        return Ok(None);
    }
    if seg.system.rows != n
        || seg.system.cols != n
        || g.cols != n
        || seg.offsets.len() != g.rows
        || seg.band.len() != g.rows
    {
        return Err(CrossingError::Malformed(
            "system, state, rows, offsets and band do not agree".into(),
        ));
    }
    let s = seg.grid;
    if s.is_empty() {
        return Err(CrossingError::Malformed("an empty lattice".into()));
    }
    let propagator = |width: f64| seg.propagators.get(seg.system, width);
    let count = s.len() - 1;
    let mut z = vec![seg.za.to_vec()];
    for k in 0..count {
        let next = propagator(s[k + 1] - s[k])?.apply(&z[k]);
        z.push(next);
    }
    let gs = g.mul(seg.system)?;
    let read = |zk: &[f64]| -> Sample {
        let values = (0..g.rows)
            .map(|j| dot(g.row(j), zk) + seg.offsets[j])
            .collect();
        (values, gs.apply(zk))
    };
    let samples: Vec<Sample> = z.iter().map(|zk| read(zk)).collect();
    let tol = seg.band;
    let ends_below = |j: usize, k: usize| samples[k + 1].0[j] < -tol[j];
    let flagged = |j: usize, k: usize| {
        ends_below(j, k) || (samples[k].1[j] < 0.0 && samples[k + 1].1[j] > 0.0)
    };
    let weights = barycentric();
    let mut nodes: HashMap<usize, Vec<Sample>> = HashMap::new();
    let mut roots: Vec<(f64, usize)> = Vec::new();
    for j in 0..g.rows {
        for k in (0..count).filter(|&k| flagged(j, k)) {
            let width = s[k + 1] - s[k];
            let mut reach = width;
            let hit = ends_below(j, k);
            if let Entry::Vacant(vacant) = nodes.entry(k) {
                let step = propagator(width / BRACKET_NODES as f64)?;
                let mut inner = vec![samples[k].clone()];
                let mut state = step.apply(&z[k]);
                for _ in 0..BRACKET_NODES - 1 {
                    inner.push(read(&state));
                    state = step.apply(&state);
                }
                inner.push(samples[k + 1].clone());
                vacant.insert(inner);
            }
            let column = &nodes[&k];
            let pick = |which: usize| -> [f64; BRACKET_NODES + 1] {
                let mut out = [0.0; BRACKET_NODES + 1];
                for (o, sample) in out.iter_mut().zip(column) {
                    *o = if which == 0 { sample.0[j] } else { sample.1[j] };
                }
                out
            };
            let guard = Interpolant::new(pick(0), width, weights);
            let slope = Interpolant::new(pick(1), width, weights);
            if !hit {
                let ends = (slope.read(0.0), slope.read(width));
                reach = if ends.0 < 0.0 && 0.0 < ends.1 {
                    located(|x| slope.read(x), 0.0, width, SCIPY_XTOL, 100)?
                } else if guard.read(0.0) <= guard.read(width) {
                    0.0
                } else {
                    width
                };
                if guard.read(reach) >= -tol[j] {
                    continue;
                }
            }
            let mut local = crossing(&guard, &slope, tol[j], reach)?;
            if guard.read(0.0) > 0.0 || (slope.read(0.0) > 0.0 && guard.read(local) > -0.5 * tol[j])
            {
                local = polished(seg.system, &z[k], g.row(j), seg.offsets[j], local, reach)?;
            }
            roots.push((s[k] + local, j));
            break;
        }
    }
    if roots.is_empty() {
        return Ok(None);
    }
    let first = roots.iter().map(|&(r, _)| r).fold(f64::INFINITY, f64::min);
    let causes = roots
        .iter()
        .filter(|&&(r, _)| r <= first + SAME_INSTANT * seg.span.max(1.0))
        .map(|&(_, j)| j)
        .collect();
    Ok(Some((first, causes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(rows: usize, cols: usize, data: &[f64]) -> Dense {
        Dense::new(rows, cols, data.to_vec()).unwrap()
    }

    /// A row reading `1 - s` over `z = [i, w]`, `w = 1` held constant: `di/ds = -w`.
    fn ramp(span: f64, band: f64) -> Option<(f64, Vec<usize>)> {
        let system = dense(2, 2, &[0.0, -1.0, 0.0, 0.0]);
        let size = [0.0, 0.0];
        let grid = lattice(&size, &[0.0, 0.0], span);
        let rows = dense(1, 2, &[1.0, 0.0]);
        let seg = Segment {
            system: &system,
            grid: &grid,
            za: &[1.0, 1.0],
            span,
            rows: &rows,
            offsets: &[0.0],
            band: &[band],
            propagators: &Propagators::default(),
        };
        first_crossing(&seg).unwrap()
    }

    #[test]
    fn a_span_key_keeps_eleven_significant_digits_and_the_first_span_owns_it() {
        assert_eq!(span_key(0.123_456_789_012_345), 0.123_456_789_01);
        assert_eq!(span_key(1e-3 * (1.0 + 1e-14)), 1e-3);
        let system = dense(2, 2, &[-1.0, 1.0, 0.0, 0.0]);
        let cache = Propagators::default();
        let first = cache.get(&system, 0.25).unwrap();
        let twin = cache.get(&system, 0.25 * (1.0 + 4e-16)).unwrap();
        assert_eq!(first.data, twin.data);
        // planted: a span past the eleventh digit is a new key and a new exponential
        assert_ne!(cache.get(&system, 0.250_000_001).unwrap().data, first.data);
    }

    #[test]
    fn a_falling_row_crosses_at_its_zero() {
        let (time, causes) = ramp(2.0, 1e-9).unwrap();
        assert_eq!(causes, vec![0]);
        assert!((time - 1.0).abs() <= 1e-12, "{time}");
    }

    #[test]
    fn planted_a_segment_that_ends_before_the_zero_finds_none() {
        assert!(ramp(0.9, 1e-9).is_none());
    }

    #[test]
    fn the_lattice_grows_at_most_twofold_and_stops_at_its_span() {
        // One mode at |lambda| = 1 that decays (Re = -1): sampled at 0.5 s while alive, then the steps double.
        let grid = lattice(&[1.0], &[-1.0], 200.0);
        assert_eq!(&grid[..3], &[0.0, 0.5, 1.0]);
        assert_eq!(*grid.last().unwrap(), 200.0);
        assert!(grid
            .windows(3)
            .all(|w| w[2] - w[1] <= 2.0 * (w[1] - w[0]) + 1e-12));
        assert_eq!(lattice(&[1.0], &[0.0], 0.0), vec![0.0]);
        assert_eq!(lattice(&[0.0], &[0.0], 3.0), vec![0.0, 3.0]);
    }

    #[test]
    fn the_barycentric_weights_are_the_signed_binomials() {
        let w = barycentric();
        assert_eq!((w[0], w[1], w[8], w[16]), (1.0, -16.0, 12870.0, 1.0));
        assert_eq!(pairwise(&[1.0; 17]), 17.0);
    }
}
