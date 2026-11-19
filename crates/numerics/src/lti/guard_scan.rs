//! The GUARD scan of one exact segment: the first tied threshold arrivals of affine guard reads along a trajectory,
//! on any [`GuardRead`] -- [`AffineGuards`] reads `G expm(M (t - t0)) y0` exactly.
//!
//! Moved from motronics-studio's `laplace_net::control::affine_scan` (a74923ef38) on 2026-11-19, where it transcribes
//! that repo's Python crossing scan line for line; that repo's Python golden holds the equivalence through the door.
//!
//! A guard read at `t` is `G expm(M (t - t0)) y0` ([`AffineGuards`]); a scan walks the segment's crossing-sample grid,
//! arms each guard on an observed departure past its threshold, and returns the FIRST tied arrivals -- each located
//! by Brent's method to the march's own instant `tie` and then proven by an actually nonpositive arrival residual.
//! The algorithm is the Python's line for line (the same reads at the same instants, the same history transitions,
//! the same tie decisions); [`brentq`] is scipy's `brentq.c` transcribed (`crate::nonlinear::roots`).
//!
//! # Equivalence
//!
//! The exponential is SciPy's bit for bit ([`crate::dense::expm::expm`] transcribes `scipy.linalg.expm` up to
//! [`crate::dense::expm::MAX_N`], to round-off past it); a read still differs in its last bits where numpy's BLAS sums the
//! propagated state and the guard rows in its own order. A
//! crossing's identity, side and every history's arming are exact; its instant agrees within `tie` [s] (Brent's
//! iterates move with the reads, and a root is only ever located to the tie); an amplitude (a guard's own numeric
//! coordinate) agrees to the reads' round-off. motronics-studio's `test_an_affine_segment_scan_is_its_golden.py`
//! holds both bounds.
//!
//! # Units
//!
//! `start`, `grid`, `tie` [s]; `system` [1/s]; each guard's values, threshold and certificate in its own coordinate.

use crate::dense::expm::{expm, ExpmError};
use crate::nonlinear::roots::{brentq, BrentError};
use std::collections::HashMap;

/// Why a scan could not run. Each is a refusal the Python raises too, by the same condition.
#[derive(Debug, Clone, PartialEq)]
pub enum GuardScanError {
    /// Inconsistent shapes, or a grid that is not at least two strictly increasing finite instants.
    Malformed(String),
    /// A read returned an infinite value (a NaN is a false relation; an infinity is a broken read).
    Infinite { time: f64 },
    /// A history transition was asked of a guard in the wrong state (the Python's `ValueError`).
    Transition(&'static str),
    /// Brent's bracket did not change sign.
    Bracket { left: f64, right: f64 },
    /// The exponential refused its argument.
    Expm(ExpmError),
    /// A graph read refused its instant (a signal law or the plant's rows), by name.
    Read(String),
}

impl std::fmt::Display for GuardScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(why) => write!(f, "affine scan: {why}"),
            Self::Infinite { time } => {
                write!(f, "crossing trajectory must retain one finite or NaN scalar per actual guard (t = {time:e} s)")
            }
            Self::Transition(why) => write!(f, "{why}"),
            Self::Bracket { left, right } => {
                write!(
                    f,
                    "f(a) and f(b) must have different signs (a = {left:e}, b = {right:e})"
                )
            }
            Self::Expm(e) => write!(f, "{e}"),
            Self::Read(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for GuardScanError {}

impl From<BrentError<GuardScanError>> for GuardScanError {
    fn from(e: BrentError<GuardScanError>) -> Self {
        match e {
            BrentError::Bracket { left, right } => Self::Bracket { left, right },
            BrentError::Function(inner) => inner,
        }
    }
}

/// A guard's checkpointable magnitude and arming (`CrossingHistory`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct History {
    pub amplitude: f64,
    pub sign: i8,
    pub after_hit: bool,
    pub inherited: i8,
}

impl History {
    /// The Python's `__post_init__` checks.
    pub fn checked(
        amplitude: f64,
        sign: i8,
        after_hit: bool,
        inherited: i8,
    ) -> Result<Self, GuardScanError> {
        if !amplitude.is_finite() || amplitude < 0.0 {
            return Err(GuardScanError::Transition(
                "a crossing amplitude is a finite nonnegative float",
            ));
        }
        if !(-1..=1).contains(&sign) {
            return Err(GuardScanError::Transition(
                "crossing sign must be the integer -1, 0 or 1",
            ));
        }
        if !(-1..=1).contains(&inherited) || (inherited != 0 && sign != 0) {
            return Err(GuardScanError::Transition(
                "an inherited side is the integer -1, 0 or 1, held only while the guard is unarmed",
            ));
        }
        if after_hit && sign != 0 {
            return Err(GuardScanError::Transition(
                "a consumed crossing must remain unarmed until a real later departure",
            ));
        }
        Ok(Self {
            amplitude,
            sign,
            after_hit,
            inherited,
        })
    }
}

/// One segment's frozen currency (`CrossingCurrency`).
#[derive(Debug, Clone, Copy)]
struct Currency {
    history: History,
    threshold: f64,
}

impl Currency {
    /// `_point_currency`: the threshold plus a read's certificate, when it carries one.
    fn at(self, certificate: Option<f64>) -> Self {
        match certificate {
            None => self,
            Some(c) => Self {
                history: self.history,
                threshold: self.threshold + c,
            },
        }
    }

    fn arrival(self, value: f64) -> Result<f64, GuardScanError> {
        if self.history.sign == 0 {
            return Err(GuardScanError::Transition(
                "an unarmed crossing has no physical arrival event",
            ));
        }
        Ok(f64::from(self.history.sign) * value + self.threshold)
    }

    fn arm(self, value: f64, side: i8) -> Result<History, GuardScanError> {
        if self.history.sign != 0 {
            return Err(GuardScanError::Transition(
                "departure requires an unarmed crossing and an actual side",
            ));
        }
        let distance = f64::from(side) * value - self.threshold;
        if f64::from(side) * value <= 0.0 || distance < 0.0 {
            return Err(GuardScanError::Transition(
                "arming requires an actual departure to the selected side",
            ));
        }
        Ok(History {
            amplitude: self.history.amplitude.max(value.abs()),
            sign: side,
            after_hit: false,
            inherited: 0,
        })
    }

    fn consume(self, value: f64) -> Result<History, GuardScanError> {
        if self.arrival(value)? > 0.0 {
            return Err(GuardScanError::Transition(
                "a nearby value is not an arrival at the crossing threshold",
            ));
        }
        Ok(History {
            amplitude: self.history.amplitude.max(value.abs()),
            sign: 0,
            after_hit: true,
            ..self.history
        })
    }

    /// `_observed`: a NaN observes nothing; else the amplitude grows (the inherited side is dropped, as the Python's).
    fn observed(self, value: f64) -> History {
        if value.is_nan() {
            return self.history;
        }
        History {
            amplitude: self.history.amplitude.max(value.abs()),
            sign: self.history.sign,
            after_hit: self.history.after_hit,
            inherited: 0,
        }
    }
}

/// What a scan reads at an instant: every guard's value and its certificate (`CrossingSample`).
pub type Sample = (Vec<f64>, Vec<Option<f64>>);

/// A segment's guard read (`CrossingPath.read`): a pure function of the instant, so a repeat may replay its sample.
pub trait GuardRead {
    /// The guards a sample carries.
    fn count(&self) -> usize;
    /// Every guard's value at `time` and its certificate (`None`: none), added to its threshold at that read.
    ///
    /// # Errors
    /// [`GuardScanError::Read`]: a read the instant refuses.
    fn sample(&mut self, time: f64) -> Result<Sample, GuardScanError>;
}

/// `time -> G expm(M (time - start)) y0`, each instant read once (`_ReadOnce`); every read carries the start's
/// certificates (`AffineScan.evidence`).
pub struct AffineGuards {
    n: usize,
    system: Vec<f64>,
    state: Vec<f64>,
    guards: Vec<f64>,
    evidence: Vec<Option<f64>>,
    count: usize,
    start: f64,
    reads: HashMap<u64, Vec<f64>>,
}

impl AffineGuards {
    /// `system` `n x n` and `guards` `count x n`, both row-major; `state` of length `n`; `evidence` one certificate
    /// per guard; `start` [s].
    pub fn new(
        system: Vec<f64>,
        state: Vec<f64>,
        guards: Vec<f64>,
        evidence: Vec<Option<f64>>,
        start: f64,
    ) -> Result<Self, GuardScanError> {
        let n = state.len();
        if n == 0
            || system.len() != n * n
            || guards.len() % n != 0
            || evidence.len() * n != guards.len()
        {
            return Err(GuardScanError::Malformed(format!(
                "system {} entries, state {n}, guards {} entries",
                system.len(),
                guards.len()
            )));
        }
        let count = guards.len() / n;
        Ok(Self {
            n,
            system,
            state,
            guards,
            evidence,
            count,
            start,
            reads: HashMap::new(),
        })
    }

    /// The guards' values at `time`.
    pub fn read(&mut self, time: f64) -> Result<Vec<f64>, GuardScanError> {
        if let Some(values) = self.reads.get(&time.to_bits()) {
            return Ok(values.clone());
        }
        let n = self.n;
        let dt = time - self.start;
        let scaled: Vec<f64> = self.system.iter().map(|m| m * dt).collect();
        let e = expm(&scaled, n).map_err(GuardScanError::Expm)?;
        let y: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|k| e[i * n + k] * self.state[k]).sum())
            .collect();
        let values: Vec<f64> = (0..self.count)
            .map(|g| (0..n).map(|k| self.guards[g * n + k] * y[k]).sum())
            .collect();
        if values.iter().any(|v| v.is_infinite()) {
            return Err(GuardScanError::Infinite { time });
        }
        self.reads.insert(time.to_bits(), values.clone());
        Ok(values)
    }
}

impl GuardRead for AffineGuards {
    fn count(&self) -> usize {
        self.count
    }

    fn sample(&mut self, time: f64) -> Result<Sample, GuardScanError> {
        Ok((self.read(time)?, self.evidence.clone()))
    }
}

/// One scan's inputs, less the read (whose samples carry each guard's certificate).
pub struct Scan<'a> {
    pub grid: &'a [f64],
    pub histories: &'a [History],
    pub thresholds: &'a [f64],
    pub tie: f64,
    pub stride: usize,
}

/// The first tied arrivals (`CrossingScan`).
#[derive(Debug, Clone, PartialEq)]
pub struct Crossings {
    pub time: f64,
    pub crossings: Vec<usize>,
    pub histories: Vec<History>,
    pub sides: Vec<i8>,
}

/// `_arrival_root`: an actually nonpositive bound within `tie` [s] of the arrival, never Brent's bare iterate.
fn arrival_root<F>(
    mut function: F,
    mut left: f64,
    mut right: f64,
    tie: f64,
) -> Result<f64, GuardScanError>
where
    F: FnMut(f64) -> Result<f64, GuardScanError>,
{
    let half = 0.5 * tie;
    let xtol = if half > 0.0 { half } else { f64::from_bits(1) };
    let (root, converged) = brentq(&mut function, left, right, xtol, 4.0 * f64::EPSILON, 100)?;
    if converged {
        if function(root)? <= 0.0 {
            return Ok(root);
        }
        left = root;
        let later = root + half;
        if half > 0.0 && later < right {
            if function(later)? <= 0.0 {
                return Ok(later);
            }
            left = later;
        }
    }
    loop {
        let middle = left + 0.5 * (right - left);
        if !(left < middle && middle < right) || right - left <= tie {
            return Ok(right);
        }
        if function(middle)? <= 0.0 {
            right = middle;
        } else {
            left = middle;
        }
    }
}

struct Scanner<'a> {
    read: &'a mut dyn GuardRead,
    tie: f64,
}

impl Scanner<'_> {
    /// `_sample`: one read of every guard, `count` of them, none infinite (a NaN is a false relation, D45).
    fn sample(&mut self, time: f64, count: usize) -> Result<Sample, GuardScanError> {
        let (values, certificates) = self.read.sample(time)?;
        if values.len() != count || certificates.len() != count {
            return Err(GuardScanError::Malformed(
                "a read must retain one scalar per actual guard".into(),
            ));
        }
        if values.iter().any(|v| v.is_infinite()) {
            return Err(GuardScanError::Infinite { time });
        }
        Ok((values, certificates))
    }

    /// `_arrivals` of one guard: `inf` where its read is NaN (it has not arrived).
    fn arrival(
        &mut self,
        time: f64,
        currencies: &[Currency],
        guard: usize,
    ) -> Result<f64, GuardScanError> {
        let (values, certificates) = self.sample(time, currencies.len())?;
        let value = values[guard];
        if value.is_nan() {
            return Ok(f64::INFINITY);
        }
        currencies[guard].at(certificates[guard]).arrival(value)
    }

    fn arrivals(
        &mut self,
        time: f64,
        currencies: &[Currency],
        guards: &[usize],
    ) -> Result<Vec<f64>, GuardScanError> {
        guards
            .iter()
            .map(|&j| self.arrival(time, currencies, j))
            .collect()
    }

    fn located(
        &mut self,
        currencies: &[Currency],
        guard: usize,
        start: f64,
        end: f64,
    ) -> Result<f64, GuardScanError> {
        let tie = self.tie;
        arrival_root(|t| self.arrival(t, currencies, guard), start, end, tie)
    }

    fn galloped(
        &mut self,
        currencies: &[Currency],
        guard: usize,
        end: f64,
        left: f64,
    ) -> Result<f64, GuardScanError> {
        if self.tie <= 0.0 {
            return Ok(left);
        }
        let mut width = self.tie;
        loop {
            let start = (end - width).max(left);
            if start <= left || self.arrival(start, currencies, guard)? > 0.0 {
                return Ok(start);
            }
            width *= 2.0;
        }
    }

    /// `_first_arrivals`: `(root, guard)` of the guards arriving first in `(left, right]`, in guard order.
    fn first_arrivals(
        &mut self,
        currencies: &[Currency],
        arrived: &[usize],
        left: f64,
        right: f64,
    ) -> Result<Vec<(f64, usize)>, GuardScanError> {
        if arrived.is_empty() {
            return Ok(Vec::new());
        }
        let tie = self.tie;
        let mut lead = arrived[0];
        if arrived.len() > 1 {
            let ends = [
                self.arrivals(left, currencies, arrived)?,
                self.arrivals(right, currencies, arrived)?,
            ];
            let middle = self.arrivals(left + 0.5 * (right - left), currencies, arrived)?;
            if let Some(k) =
                (0..arrived.len()).find(|&k| middle[k] != ends[0][k] && middle[k] != ends[1][k])
            {
                lead = arrived[k];
            }
        }
        let mut others: Vec<usize> = arrived.iter().copied().filter(|&j| j != lead).collect();
        let mut best = self.located(currencies, lead, left, right)?;
        while !others.is_empty() {
            let late = (best + tie).min(right);
            let later = self.arrivals(late, currencies, &others)?;
            others = others
                .iter()
                .zip(&later)
                .filter(|(_, &a)| a <= 0.0)
                .map(|(&j, _)| j)
                .collect();
            let early = best - tie;
            if others.is_empty() || early <= left {
                break;
            }
            let before = self.arrivals(early, currencies, &others)?;
            let earlier: Vec<usize> = others
                .iter()
                .zip(&before)
                .filter(|(_, &a)| a <= 0.0)
                .map(|(&j, _)| j)
                .collect();
            if earlier.is_empty() {
                break;
            }
            let first = earlier[0];
            others = std::iter::once(lead)
                .chain(others.iter().copied())
                .filter(|&j| j != first)
                .collect();
            lead = first;
            let start = self.galloped(currencies, lead, early, left)?;
            best = self.located(currencies, lead, start, early)?;
        }
        let mut roots = vec![(best, lead)];
        if !others.is_empty() {
            let at_best = self.arrivals(best, currencies, &others)?;
            let late = (best + tie).min(right);
            roots.extend(
                others
                    .iter()
                    .zip(&at_best)
                    .map(|(&j, &a)| (if a <= 0.0 { best } else { late }, j)),
            );
        }
        roots.sort_by_key(|&(_, j)| j);
        Ok(roots)
    }

    /// `_margins`: each guard's distance from its next event; NaN unknown.
    fn margins(
        currencies: &[Currency],
        (values, certificates): &Sample,
    ) -> Result<Vec<f64>, GuardScanError> {
        currencies
            .iter()
            .zip(values)
            .enumerate()
            .map(|(j, (currency, &value))| {
                if value.is_nan() {
                    return Ok(f64::NAN);
                }
                let actual = currency.at(certificates[j]);
                if currency.history.sign != 0 {
                    actual.arrival(value)
                } else {
                    Ok(actual.threshold - value.abs())
                }
            })
            .collect()
    }
}

/// `_clear`: no guard can have an event in `(here, reach]`.
fn clear(behind: (f64, &[f64]), here: (f64, &[f64]), reach: f64, ahead: &[f64]) -> bool {
    let ((t0, m0), (t1, m1)) = (behind, here);
    if m0.iter().chain(m1).chain(ahead).any(|v| v.is_nan()) {
        return false;
    }
    if m1.iter().chain(ahead).any(|&v| v <= 0.0) {
        return false;
    }
    m0.iter().zip(m1).all(|(&a, &b)| {
        let slope = (b - a) / (t1 - t0);
        b + slope.min(0.0) * (reach - t1) > 0.0
    })
}

/// `scan_crossings` with frozen `thresholds`: the first tied arrivals along `scan.grid`, on any [`GuardRead`] -- the
/// affine rows ([`AffineGuards`]) or the graph (`graph_read::GraphGuards`).
pub fn scan_crossings(
    read: &mut dyn GuardRead,
    scan: &Scan<'_>,
) -> Result<Crossings, GuardScanError> {
    let grid = scan.grid;
    if grid.len() < 2
        || grid.iter().any(|t| !t.is_finite())
        || grid.windows(2).any(|w| w[1] - w[0] <= 0.0)
    {
        return Err(GuardScanError::Malformed(
            "crossing scan needs at least two strictly increasing actual instants".into(),
        ));
    }
    let count = read.count();
    if scan.histories.len() != count || scan.thresholds.len() != count {
        return Err(GuardScanError::Malformed(
            "crossing histories must match actual flattened guard identities".into(),
        ));
    }
    if scan.stride < 1 {
        return Err(GuardScanError::Malformed(format!(
            "a crossing stride must be a positive integer, got {}",
            scan.stride
        )));
    }
    let mut currencies: Vec<Currency> = scan
        .histories
        .iter()
        .zip(scan.thresholds)
        .map(|(&history, &threshold)| {
            if !threshold.is_finite() || threshold < 0.0 {
                return Err(GuardScanError::Transition(
                    "a crossing threshold is a finite nonnegative float",
                ));
            }
            Ok(Currency { history, threshold })
        })
        .collect::<Result<_, _>>()?;
    let mut s = Scanner {
        read,
        tie: scan.tie,
    };
    let first = s.sample(grid[0], count)?;
    let mut previous = grid[0];
    let mut behind: Option<(f64, Vec<f64>)> = None;
    let mut margins = Scanner::margins(&currencies, &first)?;
    let (mut index, last) = (0usize, grid.len() - 1);
    let mut current = first;
    while index < last {
        let mut step = 1;
        if scan.stride > 1 && index + 1 < last {
            if let Some((tb, mb)) = &behind {
                let reach = (index + scan.stride).min(last);
                let ahead = Scanner::margins(&currencies, &s.sample(grid[reach], count)?)?;
                if clear((*tb, mb), (previous, &margins), grid[reach], &ahead) {
                    step = reach - index;
                }
            }
        }
        index += step;
        let instant = grid[index];
        current = s.sample(instant, count)?;
        let mut arrived = Vec::new();
        let mut departures = Vec::new();
        for (j, (currency, &value)) in currencies.iter().zip(&current.0).enumerate() {
            if value.is_nan() {
                continue;
            }
            let actual = currency.at(current.1[j]);
            if currency.history.sign == 0 {
                if value.abs() > actual.threshold {
                    departures.push((j, value, if value > 0.0 { 1i8 } else { -1i8 }));
                }
                continue;
            }
            if actual.arrival(value)? <= 0.0 {
                arrived.push(j);
            }
        }
        let roots = s.first_arrivals(&currencies, &arrived, previous, instant)?;
        if !roots.is_empty() {
            let time = roots
                .iter()
                .map(|&(root, _)| root)
                .fold(f64::INFINITY, f64::min);
            let hits: Vec<usize> = roots
                .iter()
                .filter(|&&(root, _)| root - time <= scan.tie)
                .map(|&(_, j)| j)
                .collect();
            // Each tied hit is consumed at its OWN rooted read (value and certificate); the rest at the endpoint's.
            let (endpoint, at_end) = s.sample(time, count)?;
            let mut rooted: HashMap<usize, (f64, Option<f64>)> = HashMap::new();
            for &(root, j) in &roots {
                if hits.contains(&j) {
                    let (values, certificates) = s.sample(root, count)?;
                    rooted.insert(j, (values[j], certificates[j]));
                }
            }
            let mut saved = Vec::with_capacity(count);
            for (j, currency) in currencies.iter().enumerate() {
                let actual = currency.at(at_end[j]);
                if let Some(&(value, certificate)) = rooted.get(&j) {
                    saved.push(currency.at(certificate).consume(value)?);
                } else if currency.history.sign == 0 && endpoint[j].abs() > actual.threshold {
                    saved.push(actual.arm(endpoint[j], if endpoint[j] > 0.0 { 1 } else { -1 })?);
                } else {
                    saved.push(currency.observed(endpoint[j]));
                }
            }
            let sides = hits.iter().map(|&j| -currencies[j].history.sign).collect();
            return Ok(Crossings {
                time,
                crossings: hits,
                histories: saved,
                sides,
            });
        }
        for &(j, value, side) in &departures {
            let currency = currencies[j];
            currencies[j] = Currency {
                history: currency.at(current.1[j]).arm(value, side)?,
                threshold: currency.threshold,
            };
        }
        currencies = currencies
            .iter()
            .zip(&current.0)
            .map(|(currency, &value)| Currency {
                history: currency.observed(value),
                threshold: currency.threshold,
            })
            .collect();
        behind = Some((previous, margins));
        margins = Scanner::margins(&currencies, &current)?;
        previous = instant;
    }
    Ok(Crossings {
        time: grid[last],
        crossings: Vec::new(),
        histories: currencies
            .iter()
            .zip(&current.0)
            .map(|(currency, &value)| currency.observed(value))
            .collect(),
        sides: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One guard `y(t) = 1 - t` (the state `[1, -1]` under `M = [[0, 1], [0, 0]]`... read as `G = [1, 0]`): armed
    /// positive, it arrives where `1 - t + threshold <= 0`.
    fn ramp() -> AffineGuards {
        AffineGuards::new(
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, -1.0],
            vec![1.0, 0.0],
            vec![None],
            0.0,
        )
        .unwrap()
    }

    fn armed() -> History {
        History {
            amplitude: 1.0,
            sign: 1,
            after_hit: false,
            inherited: 0,
        }
    }

    #[test]
    fn an_armed_ramp_arrives_within_the_tie_of_its_threshold() {
        let grid: Vec<f64> = (0..=16).map(|k| 2.0 * f64::from(k) / 16.0).collect();
        let tie = 1e-9;
        let scan = Scan {
            grid: &grid,
            histories: &[armed()],
            thresholds: &[1e-6],
            tie,
            stride: 1,
        };
        let found = scan_crossings(&mut ramp(), &scan).unwrap();
        assert_eq!(found.crossings, vec![0]);
        assert_eq!(found.sides, vec![-1]);
        assert!((found.time - (1.0 + 1e-6)).abs() <= tie, "{}", found.time);
        assert_eq!(found.histories[0].sign, 0);
        assert!(found.histories[0].after_hit);
    }

    #[test]
    fn planted_a_grid_that_stops_before_the_arrival_finds_none() {
        let grid: Vec<f64> = (0..=16).map(|k| 0.9 * f64::from(k) / 16.0).collect();
        let scan = Scan {
            grid: &grid,
            histories: &[armed()],
            thresholds: &[1e-6],
            tie: 1e-9,
            stride: 1,
        };
        let found = scan_crossings(&mut ramp(), &scan).unwrap();
        assert!(found.crossings.is_empty());
        assert_eq!(found.time, 0.9);
    }

    #[test]
    fn an_unarmed_guard_arms_on_its_departure_and_arrives_on_its_return() {
        // y(t) = 1 - t from an UNARMED history at threshold 0.5: departs positive at once, arrives at t = 1.5.
        let grid: Vec<f64> = (0..=16).map(|k| 2.0 * f64::from(k) / 16.0).collect();
        let unarmed = History {
            amplitude: 0.0,
            sign: 0,
            after_hit: false,
            inherited: 0,
        };
        let scan = Scan {
            grid: &grid,
            histories: &[unarmed],
            thresholds: &[0.5],
            tie: 1e-9,
            stride: 1,
        };
        let found = scan_crossings(&mut ramp(), &scan).unwrap();
        assert_eq!(found.crossings, vec![0]);
        assert!((found.time - 1.5).abs() <= 1e-9, "{}", found.time);
    }

    #[test]
    fn a_grid_that_does_not_increase_is_refused() {
        let scan = Scan {
            grid: &[0.0, 0.0],
            histories: &[armed()],
            thresholds: &[0.0],
            tie: 0.0,
            stride: 1,
        };
        assert!(matches!(
            scan_crossings(&mut ramp(), &scan),
            Err(GuardScanError::Malformed(_))
        ));
    }
}
