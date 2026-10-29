//! THE CONVERGENCE RULE every strategy stops on: the norms, the energy and residual tests, and
//! the two watches (stagnation, a rising increment) that end a run which is no longer converging.
//! Split out of `mod.rs` on 2026-09-29, which crossed the 800-line band.

use super::{STAGNATION_MAX_MIN_RATIO, STAGNATION_WINDOW};
use std::collections::VecDeque;

/// Euclidean norm.
#[must_use]
pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

/// `||x_next - x|| / ||x||` — the SECOND half of [`has_converged`], on its own.
///
/// Split out so the number can be REPORTED, not only tested. A run that stops at `max_iter` used
/// to log the energy error alone, and the two states that produces are opposite: an energy plateau
/// at a genuine fixed point (the criterion is too strict) and an energy plateau the iterate is
/// still wandering across (the criterion is right and the solve has not converged). Measured
/// 2026-08-27, one heavy shard capped 104 solves at 100/100 while logging 0.00% energy error, and
/// the log could not distinguish those two — which is why this is observable now.
#[must_use]
pub fn increment_ratio(x: &[f64], x_next: &[f64]) -> f64 {
    let x_scale = norm(x).max(1e-12);
    let d = x_next
        .iter()
        .zip(x)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    d / x_scale
}

/// Relative energy change in PERCENT; infinite on the first pass, where there is no predecessor.
#[must_use]
pub fn energy_error_pct(total_energy: f64, last_energy: Option<f64>) -> f64 {
    match last_energy {
        None => f64::INFINITY,
        Some(last) => {
            let denom = total_energy.abs().max(last.abs()).max(1e-12);
            (total_energy - last).abs() / denom * 100.0
        }
    }
}

/// The ONE convergence rule: the energy has settled AND the iterate has stopped moving.
///
/// A FLAT ENERGY IS NOT A FIXED POINT. `total_energy` can sit on a near-flat plateau that is not
/// the solution — measured on a saturating-BH case, which read 0.00% energy change for 20+
/// iterations at an energy level ~14x the physical one before escaping and reaching the genuine
/// fixed point ~15 iterations later. On an energy-only test that is a `converged: true` return at
/// a wrong answer, and the caller cannot tell.
///
/// That failure mode is a property of the ENERGY CRITERION, not of any one strategy. It was
/// previously guarded inside the damped Picard alone, and even there only behind an earlier,
/// unrelated damping event — so it did not fire on an undamped run of the very strategy that owned
/// it, and the other eleven strategies never had it. Hence one function, consulted by all of them.
///
/// The second half is the direct fixed-point test `x_next == x`, immune to a flat-but-wrong energy
/// plateau, on the step the strategy is ABOUT to take. It costs nothing: every strategy has
/// already computed `x_next` by the time it asks. `rtol_energy_pct` is a PERCENT and the increment
/// is a FRACTION, so the `/ 100` is the unit conversion between them, not a second, tighter
/// tolerance.
#[must_use]
pub fn has_converged(err: f64, x: &[f64], x_next: &[f64], rtol_energy_pct: f64) -> bool {
    err < rtol_energy_pct && increment_ratio(x, x_next) < rtol_energy_pct / 100.0
}

/// The residual half of the ONE rule, for a strategy that holds the residual (the Newton family).
///
/// `residual_ratio = ||r(x_next)|| / max(||r(x_0)||, ||rhs||)` below `rtol_energy_pct / 100`,
/// AND the same increment half [`has_converged`] applies. Port of the Python `_has_converged`'s
/// `residual_ratio=` form (2026-09-25).
#[must_use]
pub fn has_converged_residual(
    x: &[f64],
    x_next: &[f64],
    rtol_energy_pct: f64,
    residual_ratio: f64,
) -> bool {
    let rtol = rtol_energy_pct / 100.0;
    residual_ratio < rtol && increment_ratio(x, x_next) < rtol
}

/// The stagnation break, consulted NEXT TO [`has_converged`] — never instead of it.
///
/// [`has_converged`] is the FAST break: energy settled AND the increment below `rtol/100`, checked
/// on the step about to be taken. It is unreachable when the increment has a stationary floor
/// ABOVE `rtol/100` that is not the iterate wandering — measured, the hanging-fold update pins it
/// at ~3.5e-4. This watch is the STAGNATION break for exactly that shape: the energy error has
/// been under rtol for [`STAGNATION_WINDOW`] CONSECUTIVE steps AND the increment is stationary
/// over the same window (`max/min <=` [`STAGNATION_MAX_MIN_RATIO`]).
///
/// The energy half is NOT weakened — every window step must independently satisfy `err < rtol`,
/// the same bar the fast break sets. What replaces the increment FLOOR is the increment
/// STATIONARITY test, and the wandering-increment false plateau fails it by construction: a
/// 20-step 0.00% energy plateau whose increment moves by more than the 2x band never fires this
/// break, which is the hazard [`has_converged`] measured and the reason the window is 25, not 5.
///
/// A sliding window, so "K consecutive" costs no explicit reset: one step over rtol (or one
/// wandering increment) sits INSIDE the window and blocks the break until it has aged out.
#[derive(Debug, Clone)]
pub struct StagnationWatch {
    rtol: f64,
    window: usize,
    errs: VecDeque<f64>,
    increments: VecDeque<f64>,
}

impl StagnationWatch {
    #[must_use]
    pub fn new(rtol_energy_pct: f64) -> Self {
        Self::with_window(rtol_energy_pct, STAGNATION_WINDOW)
    }

    #[must_use]
    pub fn with_window(rtol_energy_pct: f64, window: usize) -> Self {
        Self {
            rtol: rtol_energy_pct,
            window: window.max(1),
            errs: VecDeque::new(),
            increments: VecDeque::new(),
        }
    }

    /// Record one iteration's pair; `true` iff the full window is settled AND stationary
    /// (inside the band AND not trending upward - see the body).
    pub fn observe(&mut self, err: f64, increment: f64) -> bool {
        if self.errs.len() == self.window {
            self.errs.pop_front();
            self.increments.pop_front();
        }
        self.errs.push_back(err);
        self.increments.push_back(increment);
        if self.errs.len() < self.window {
            return false;
        }
        if !self.errs.iter().all(|&e| e < self.rtol) {
            return false;
        }
        let lo = self
            .increments
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let hi = self
            .increments
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if hi > STAGNATION_MAX_MIN_RATIO * lo.max(1e-300) {
            return false;
        }
        // STATIONARY IS NOT THE SAME AS INSIDE A BAND, and the max/min test alone cannot tell them
        // apart: an increment CREEPING UP by 0.7% a step stays inside a 2x band for 100 iterations
        // while diverging the whole time. MEASURED 2026-09-06 on `case_a mec:quad` -
        // 1.91e-3 -> 2.05e-3 across ten steps, band ratio 1.07, and the break fired and returned
        // `converged = true` at 20x the increment bar.
        let half = self.increments.len() / 2;
        let early: f64 = self.increments.iter().take(half).sum::<f64>() / half as f64;
        let late: f64 = self
            .increments
            .iter()
            .skip(self.increments.len() - half)
            .sum::<f64>()
            / half as f64;
        late <= early
    }
}

/// Consecutive steps of a RISING increment before the iterate is damped on the increment alone.
///
/// Symmetric with [`super::ALPHA_CALM_STEPS`] on purpose: the same amount of evidence that licenses a
/// recovery licenses a damping. One step is not enough - an increment jitters - and the measured
/// blow-out rose monotonically for ten.
pub const INCREMENT_RISE_STEPS: usize = 3;

/// What the ITERATE says about the step size - the channel the ENERGY goes deaf on.
///
/// THE ONE REGISTRY for "what should alpha do?" when the energy has nothing to say, consulted by
/// every damped Picard leg instead of each spelling its own half of it.
///
/// Both alpha SHRINK ratchets are gated on `meaningful` (`err >= rtol_energy_pct`) and that is
/// right and measured: under the tolerance the energy delta's SIGN is floating-point noise, and a
/// noise-driven shrink is what pinned alpha at 0.108 on `case_a`. But the recovery leg then
/// read the SAME deaf channel and drew the opposite conclusion from the same silence, so on an
/// energy plateau alpha grew every `ALPHA_CALM_STEPS` with nothing able to object.
///
/// MEASURED 2026-09-06 through `downstream run cases/case_a --select
/// mec,quad --max-num-iter 600`, the `[sparse]` leg on `picard-damped`. At iteration 119 the
/// increment was falling steadily toward the 1e-4 bar (2.03e-4 -> 1.95e-4 -> 1.89e-4) at alpha
/// 0.137. Recovery raised alpha to 0.205 then 0.308 and the increment blew out ~20x until the
/// damage finally reached the ENERGY and re-armed a shrink. At the production cap of 100 the solve
/// stops inside that cycle and reports `converged = false` on an iterate that was, twenty
/// iterations earlier, nearly converged.
///
/// BOTH DIRECTIONS, and blocking the growth alone was measured to be half a fix: with only
/// [`Self::permits_growth`] the same run parked at alpha 0.150 with the increment creeping UP
/// (1.91e-3 -> 2.05e-3 over ten steps) - no longer cycling, still not contracting, and nothing left
/// that could move alpha. So a SUSTAINED rise damps: [`Self::claim_damping`] is the increment's own
/// overshoot signal, the exact counterpart of `err > prev_err` for a plateau where `err` cannot
/// express it.
///
/// ASYMMETRIC EVIDENCE ON PURPOSE. One non-contracting step is enough to REFUSE a growth (refusing
/// costs a solve nothing but iterations at the current step size); [`INCREMENT_RISE_STEPS`]
/// consecutive rises are required to force a shrink (shrinking is what over-damping is made of).
#[derive(Debug, Clone, Copy, Default)]
pub struct IncrementTrend {
    last: Option<f64>,
    rising: usize,
}

impl IncrementTrend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record this step's increment. Called once per iteration, after it is computed.
    pub fn observe(&mut self, increment: f64) {
        self.rising = match self.last {
            Some(last) if increment > last => self.rising + 1,
            _ => 0,
        };
        self.last = Some(increment);
    }

    /// May alpha grow? Only while the last step did not move the iterate MORE than the one before.
    ///
    /// No history reads as permission: refusing a recovery for want of evidence would throttle
    /// every solve's first steps, which is the crawl the recovery leg exists to remove.
    #[must_use]
    pub fn permits_growth(&self) -> bool {
        self.rising == 0
    }

    /// Has the increment risen for [`INCREMENT_RISE_STEPS`] running? CONSUMES the demand.
    ///
    /// Consuming is what makes one sustained rise cost ONE shrink rather than one per step for as
    /// long as the rise lasts - an unconsumed demand would drive alpha to its floor in three more
    /// iterations, which is precisely the over-damping the `meaningful` guard was added to stop.
    pub fn claim_damping(&mut self) -> bool {
        if self.rising < INCREMENT_RISE_STEPS {
            return false;
        }
        self.rising = 0;
        true
    }
}
