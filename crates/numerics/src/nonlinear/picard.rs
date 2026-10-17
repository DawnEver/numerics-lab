//! The Picard family: ONE successive-substitution loop on two orthogonal axes.
//!
//! * WHAT is relaxed — the iterate (`x += alpha (solve - x)`) or the coefficients (mix the raw
//!   secant arrays before the boundary fold, then `x = solve`).
//! * HOW alpha moves — constant 1, or the damped schedule of [`AlphaSchedule`].
//!
//! Everything else — assemble at the top, the one-step-delayed break, [`has_converged`] or
//! [`StagnationWatch`], `on_iteration` — is shared.

use super::{
    energy_error_pct, has_converged, increment_ratio, IncrementTrend, IterationResult,
    IterationStrategy, NonlinearError, NonlinearSystem, RawState, ResultOf, SolveOptions,
    StagnationWatch, StateOf, ALPHA_CALM_STEPS, ALPHA_GROW, ALPHA_MIN,
};

/// What the loop under-relaxes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Relax {
    /// The iterate: `x += alpha (solve - x)`.
    X,
    /// The assembly coefficients, through the raw pre-fold arrays; then `x = solve`.
    Coefficients,
}

/// The under-relaxation factor: constant 1, or damped.
///
/// The damped rules, in order each step:
///
/// * shrink x0.5 when [`IncrementTrend`] claims damping (the iterate overshot);
/// * shrink x0.5 when the energy error grows, and x0.6 on an energy-delta sign flip — both only
///   ABOVE `rtol`: below it the energy is at its noise floor and neither is a limit cycle;
/// * after [`ALPHA_CALM_STEPS`] calm steps grow x [`ALPHA_GROW`], capped at the step size that
///   last misbehaved; once alpha has reached the cap, the cap itself grows.
///
/// A shrink that did not move alpha (already at [`ALPHA_MIN`]) sets no cap, and a step whose
/// increment grew is not calm. The fixed point and the convergence test do not depend on alpha.
#[derive(Debug, Clone)]
struct AlphaSchedule {
    damped: bool,
    alpha: f64,
    cap: f64,
    calm: usize,
    prev_err: f64,
    prev_sign: i32,
}

impl AlphaSchedule {
    fn new(damped: bool) -> Self {
        Self {
            damped,
            alpha: 1.0,
            cap: 1.0,
            calm: 0,
            prev_err: f64::INFINITY,
            prev_sign: 0,
        }
    }

    /// Update alpha from this step's energy error and energy change; return it.
    fn step(&mut self, err: f64, delta: Option<f64>, rtol: f64, trend: &mut IncrementTrend) -> f64 {
        if !self.damped {
            return self.alpha;
        }
        let meaningful = err >= rtol;
        let before = self.alpha;
        let mut disturbed = false;
        if trend.claim_damping() {
            self.alpha = ALPHA_MIN.max(self.alpha * 0.5);
            disturbed = true;
        }
        if meaningful && self.prev_err.is_finite() && err > self.prev_err {
            self.alpha = ALPHA_MIN.max(self.alpha * 0.5);
            disturbed = true;
        }
        if let Some(delta) = delta {
            let sign = i32::from(delta > 0.0) - i32::from(delta < 0.0);
            if meaningful && self.prev_sign != 0 && sign != 0 && sign != self.prev_sign {
                self.alpha = ALPHA_MIN.max(self.alpha * 0.6);
                disturbed = true;
            }
            self.prev_sign = sign;
        }
        self.prev_err = err;
        if disturbed {
            if self.alpha < before {
                self.cap = before;
            }
            self.calm = 0;
        } else if !trend.permits_growth() {
            self.calm = 0;
        } else {
            self.calm += 1;
            if self.calm >= ALPHA_CALM_STEPS {
                if self.alpha >= self.cap {
                    self.cap = 1.0_f64.min(self.cap * ALPHA_GROW);
                }
                self.alpha = self.cap.min(self.alpha * ALPHA_GROW);
                self.calm = 0;
            }
        }
        self.alpha
    }
}

/// Successive substitution, parametrised by relax target and alpha schedule.
///
/// `Relax::Coefficients` rests on one identity: every raw stiffness and remanence-source entry is
/// `nu_e * <geometric factor>` and assembly is linear, so mixing the per-element reluctivity
/// `nu_k = (1-a) nu_{k-1} + a nu(B_k)` equals mixing the assembled raw arrays
/// `S_k = (1-a) S_{k-1} + a S(nu(B_k))` (likewise the source). That holds ONLY for a GLOBAL
/// scalar alpha; a per-element alpha would break it silently. The mix must precede the boundary
/// fold, hence the `raw_state`/`fold_raw` requirement.
fn picard_loop<S: NonlinearSystem>(
    system: &mut S,
    opts: &SolveOptions,
    name: &'static str,
    relax: Relax,
    damped: bool,
) -> Result<ResultOf<S>, NonlinearError> {
    let refuse = || NonlinearError::UnsupportedSystem {
        strategy: name,
        need: "raw_state/fold_raw".to_owned(),
    };
    if relax == Relax::Coefficients && !system.supports_raw_mixing() {
        return Err(refuse());
    }
    let rtol = opts.rtol_energy_pct;
    let mut schedule = AlphaSchedule::new(damped);
    let mut trend = IncrementTrend::new();
    let mut watch = StagnationWatch::new(rtol);
    let mut x = opts.start(system);
    // Last iteration's MIXED raw arrays (`Relax::Coefficients` only); `None` before the first pass.
    let mut prev_mixed: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut state: Option<StateOf<S>> = None;
    let mut last_energy: Option<f64> = None;
    let mut converged = false;
    let mut ran = 0;
    for i in 0..opts.max_iter {
        ran = i + 1;
        let st = match relax {
            Relax::X => system.assemble(&x)?,
            Relax::Coefficients => {
                let raw = system.raw_state(&x)?.ok_or_else(refuse)?;
                let (vals, source) = match &prev_mixed {
                    None => (raw.vals, raw.source),
                    Some((pv, pt)) => (
                        mix(pv, &raw.vals, schedule.alpha),
                        mix(pt, &raw.source, schedule.alpha),
                    ),
                };
                let mixed = RawState {
                    rows: raw.rows,
                    cols: raw.cols,
                    vals,
                    source,
                    total_energy: raw.total_energy,
                };
                let st = system.fold_raw(&mixed)?.ok_or_else(refuse)?;
                prev_mixed = Some((mixed.vals, mixed.source));
                st
            }
        };
        if converged {
            state = Some(st);
            break;
        }
        let err = energy_error_pct(st.total_energy, last_energy);
        let delta = last_energy.map(|last| st.total_energy - last);
        let alpha = schedule.step(err, delta, rtol, &mut trend);
        let x_solve = system.solve_linear(&st)?;
        let increment = increment_ratio(&x, &x_solve);
        converged = has_converged(err, &x, &x_solve, rtol) || watch.observe(err, increment);
        last_energy = Some(st.total_energy);
        trend.observe(increment);
        x = if relax == Relax::X && damped {
            x.iter()
                .zip(&x_solve)
                .map(|(a, b)| a + alpha * (b - a))
                .collect()
        } else {
            x_solve
        };
        if system.is_linear() {
            converged = true;
        }
        system.on_iteration(i, &st, err, alpha, increment);
        state = Some(st);
    }
    Ok(IterationResult {
        x,
        state,
        iterations: ran,
        converged,
    })
}

/// `(1-alpha) * prev + alpha * target`, elementwise.
fn mix(prev: &[f64], target: &[f64], alpha: f64) -> Vec<f64> {
    prev.iter()
        .zip(target)
        .map(|(p, t)| (1.0 - alpha) * p + alpha * t)
        .collect()
}

/// Relax the iterate on the damped schedule — the MEC default.
#[derive(Debug, Clone, Copy, Default)]
pub struct DampedPicardStrategy;

impl<S: NonlinearSystem> IterationStrategy<S> for DampedPicardStrategy {
    fn name(&self) -> &'static str {
        "picard-damped"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        picard_loop(system, opts, "picard-damped", Relax::X, true)
    }
}

/// Relax the reluctivity on the damped schedule — the FEA default.
#[derive(Debug, Clone, Copy, Default)]
pub struct CoefficientDampedPicardStrategy;

impl<S: NonlinearSystem> IterationStrategy<S> for CoefficientDampedPicardStrategy {
    fn name(&self) -> &'static str {
        "coefficient-damped"
    }

    /// What this strategy needs of a system, as data, so a caller can ask before solving.
    fn requires(&self) -> &'static [&'static str] {
        &["raw_state", "fold_raw"]
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        picard_loop(system, opts, "coefficient-damped", Relax::Coefficients, true)
    }
}

/// Plain substitution, alpha 1. Kept so its divergence on saturating B-H stays a labelled datum.
#[derive(Debug, Clone, Copy, Default)]
pub struct PicardStrategy;

impl<S: NonlinearSystem> IterationStrategy<S> for PicardStrategy {
    fn name(&self) -> &'static str {
        "picard"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        picard_loop(system, opts, "picard", Relax::X, false)
    }
}
