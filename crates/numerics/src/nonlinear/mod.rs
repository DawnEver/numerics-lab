//! Pluggable nonlinear-iteration strategies — the Rust mirror of
//! `src/motronics/gauss/nonlinear/`.
//!
//! The governing equation knows how to ASSEMBLE (given the current iterate, produce the system
//! matrix, the energy, and whatever payload the caller returns); the STRATEGY knows how to
//! ITERATE (Picard, damped Picard, Newton, quasi-Newton). They meet at the small
//! [`NonlinearSystem`] trait, so a solver picks its nonlinear method without the equation code
//! changing. This module drives the iterate only; it never assembles physics or touches a mesh.
//!
//! First principle (`AGENTS.md`): IMPLEMENT every method and characterize it — never exclude one
//! because an experiment showed it inferior. Plain Picard's historical divergence on saturating
//! B-H is a labelled datum, not a hidden exclusion. The default is the damped Picard extracted
//! VERBATIM from the MEC/FEA inline loops, so wiring this seam leaves every golden unchanged.
//!
//! # What the Python calls a Protocol, this calls a trait
//!
//! [`NonlinearSystem`] carries two associated types. `Matrix` is the tangent / system operator,
//! DELIBERATELY opaque to every strategy: MEC hands back a sparse matrix, FEA a reduced-tangent
//! bundle only its own `solve_system` understands, and a strategy passes it back verbatim without
//! ever inspecting its shape. `Payload` is the equation-specific extras the CALLER reads off the
//! final state. Associated types rather than a boxed `dyn Any` keep that opacity at no runtime
//! cost, and let the compiler enforce that a tangent from one equation cannot reach another's
//! solver.

use std::collections::VecDeque;
use std::fmt;

mod budget;
mod newton;
mod picard;
mod quasi;

pub use budget::{with_time_budget, TimeBudgetedSystem, BUDGETED_CALLS};
pub use newton::{
    ContinuationStrategy, ModifiedNewtonStrategy, NewtonKrylovStrategy, NewtonLineSearchStrategy,
    NewtonStrategy,
};
pub use picard::{
    AdaptiveDampedPicardStrategy, DampedPicardStrategy, NuRelaxationDampedPicardStrategy,
    PicardStrategy,
};
pub use quasi::{AndersonStrategy, BfgsStrategy, BroydenStrategy, LbfgsStrategy};

// -- The damped-Picard parameters, moved verbatim from `mec/_sparse.py` -----------------------

/// Floor for the adaptive under-relaxation factor.
pub const ALPHA_MIN: f64 = 0.1;
/// Adaptive-Picard recovery factor: grow `alpha` back toward 1 on a progressing step.
pub const ALPHA_GROW: f64 = 1.5;
/// Convergence: total-energy change below this fraction of a percent.
pub const RTOL_ENERGY_PCT: f64 = 0.01;

/// Consecutive UNDISTURBED steps (no energy-error growth, no energy-delta sign flip) before the
/// damped Picard loop grows `alpha` back.
///
/// The ratchet used to be one-way, so a single early overshoot damped the whole smooth tail:
/// measured on `ipm_uv_cummins mec:quad`, alpha stuck at `1.0 * 0.5 * 0.6^3 = 0.108` and every
/// rotor position took 44-61 iterations at ~3 s each. Recovery is capped at `alpha_cap` — the
/// step size that last misbehaved is never retried — so a genuine limit cycle still settles
/// instead of oscillating between the two regimes.
pub const ALPHA_CALM_STEPS: usize = 3;

/// Consecutive steps of the stagnation window.
///
/// MEASURED (lane K, 2026-09-01): on `ipm_2u_byd_superdrive` newton's energy error passes rtol by
/// iteration ~12 while the increment pins at ~3.57e-4 — the stationary limit of the hanging-fold
/// update — so [`has_converged`]'s increment floor (`rtol/100` = 1e-4, 2-4.5x tighter than the
/// stationary increment on every measured case) is never met and the solve burns the whole
/// 100-step cap: ~85 wasted iterations, ~123 s. K must OUTLAST the documented false-plateau
/// hazard — the saturating-BH plateau that read 0.00% energy change for 20+ iterations (see
/// [`has_converged`]) — so K = 25; it fires on byd-newton at ~iteration 39.
pub const STAGNATION_WINDOW: usize = 25;

/// How flat the increment must be across the window to count as STATIONARY (max/min).
///
/// The false plateau's increment WANDERS, so this band — not the energy half — is what refuses it.
pub const STAGNATION_MAX_MIN_RATIO: f64 = 2.0;

/// Default wall-clock ceiling for ONE nonlinear solve, in seconds.
///
/// THE ARITHMETIC THAT PICKS IT is the TIER's, not the solve's. The budget is per solve and a
/// rotation sweep runs at least one per rotor position, so what has to hold is
/// `n_positions * budget < the test's own timeout`. `ipm_uv` declares `n_positions = 7` and its
/// ripple sweep carries a 5400 s timeout: `7 * 500 = 3500` fits, `7 * 900 = 6300` does not — and
/// on Windows the firing timeout takes the WORKER, losing every result queued behind it. A budget
/// too GENEROUS is therefore not the safe direction; it is no budget at all.
///
/// THE MEASUREMENTS IT SITS BETWEEN: ~200 s per solve for the cases that complete, and 15-17 min
/// for the sdm case that hangs. Measured 2026-08-27, `dcm_fdm_pear` (FEA-Galerkin) and
/// `dcm_fdm_rect` (MEC Picard) both ran past 900 s standalone — neither an sdm solve, so an
/// sdm-only budget could not have reached them.
///
/// WHAT THIS DELIBERATELY COSTS: a solve that legitimately needs longer now FAILS. That trade is
/// taken knowingly — a tier that returns INCONCLUSIVE loses EVERY result, so one named red leg is
/// strictly more information than none. A caller that knows its solve is genuinely long passes
/// its own seconds.
pub const DEFAULT_SOLVE_BUDGET_S: f64 = 500.0;

/// The registry's spellings, in one place so the resolver and its error cannot drift.
pub const KNOWN_STRATEGIES: [&str; 13] = [
    "anderson",
    "bfgs",
    "continuation",
    "l-bfgs",
    "modified-newton",
    "newton",
    "newton-krylov",
    "newton-linesearch",
    "nu-damped",
    "picard",
    "picard-adaptive",
    "picard-damped",
    "quasi-newton",
];

// -- Errors -----------------------------------------------------------------------------------

/// Every way a nonlinear solve can refuse.
#[derive(Debug, Clone, PartialEq)]
pub enum NonlinearError {
    /// A nonlinear solve ran past the wall-clock it was given, and stopped ITSELF.
    ///
    /// FAILED RATHER THAN RETURNED, and the deciding fact is not a preference:
    /// [`IterationResult`] already carries `converged`, and nothing downstream reads it. A solve
    /// that returned `converged: false` here would be "a warning on an unchanged success return",
    /// and the test of that shape is whether the CALLER can tell. It cannot.
    BudgetExceeded {
        budget_s: f64,
        elapsed_s: f64,
        /// The call the budget refused to START.
        at: &'static str,
    },
    /// A strategy needs a capability this system does not supply.
    UnsupportedSystem {
        strategy: &'static str,
        need: String,
    },
    /// An unknown strategy name.
    UnknownStrategy { name: String },
    /// The equation cannot rescale its source (the `continuation` degradation path).
    SourceNotScalable,
    /// Vectors of disagreeing length crossed the seam.
    ShapeMismatch { expected: usize, got: usize },
    /// The equation refused for a reason only it can name.
    Equation(String),
}

impl fmt::Display for NonlinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetExceeded {
                budget_s,
                elapsed_s,
                at,
            } => write!(
                f,
                "nonlinear solve exceeded its {budget_s} s budget before {at} ({elapsed_s} s \
                 elapsed). The budget stops the solve at the next expensive call rather than \
                 letting the tier be killed around it -- an unbounded solve inside a bounded tier \
                 costs every other result in the run its verdict, which is how a heavy run \
                 returns INCONCLUSIVE instead of one red test."
            ),
            Self::UnsupportedSystem { strategy, need } => write!(
                f,
                "'{strategy}' requires a system exposing {need}; this one does not. A capability \
                 boundary only the failure announces is a boundary no caller can plan around."
            ),
            Self::UnknownStrategy { name } => write!(
                f,
                "unknown iteration strategy {name:?}; known: {}",
                KNOWN_STRATEGIES.join(", ")
            ),
            Self::SourceNotScalable => write!(
                f,
                "this equation cannot rescale its source; 'continuation' degrades to a single \
                 full-source inner solve"
            ),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "expected a vector of length {expected}; got {got}")
            }
            Self::Equation(msg) => write!(f, "the equation refused: {msg}"),
        }
    }
}

impl std::error::Error for NonlinearError {}

// -- The seam ---------------------------------------------------------------------------------

/// One assembly of the governing equation at the current iterate.
#[derive(Debug, Clone, PartialEq)]
pub struct AssembledState<M, P> {
    /// The (reduced) linear system a strategy solves for the next iterate. `None` for an equation
    /// whose `solve_linear` needs no explicit operator.
    pub system_matrix: Option<M>,
    /// The scalar the convergence test tracks.
    pub total_energy: f64,
    /// Equation-specific extras the CALLER reads off the final state (e.g. the per-branch
    /// energy/flux/B for MEC); the strategy never inspects it.
    pub payload: Option<P>,
}

/// The pre-boundary-fold arrays `nu-damped` mixes.
///
/// See [`NuRelaxationDampedPicardStrategy`] for why the mixing has to happen on THESE arrays and
/// not on an already-folded [`AssembledState`].
#[derive(Debug, Clone, PartialEq)]
pub struct RawState {
    pub rows: Vec<u32>,
    pub cols: Vec<u32>,
    /// Raw stiffness values — LINEAR in the per-element reluctivity, the identity the whole
    /// strategy rests on.
    pub vals: Vec<f64>,
    /// Raw source vector, likewise linear in the reluctivity.
    pub source: Vec<f64>,
    pub total_energy: f64,
}

/// What a strategy returns: the converged iterate, the final assembly, and run metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct IterationResult<M, P> {
    pub x: Vec<f64>,
    pub state: Option<AssembledState<M, P>>,
    pub iterations: usize,
    pub converged: bool,
}

/// Shorthand for the state type a given system produces.
pub type StateOf<S> =
    AssembledState<<S as NonlinearSystem>::Matrix, <S as NonlinearSystem>::Payload>;
/// Shorthand for the result type a given system produces.
pub type ResultOf<S> =
    IterationResult<<S as NonlinearSystem>::Matrix, <S as NonlinearSystem>::Payload>;

/// The seam an equation implements so a strategy can drive it.
///
/// `assemble` / `solve_linear` are the Picard interface (fixed-point substitution
/// `x <- solve_linear(assemble(x))`). `residual` / `tangent` / `solve_system` are the Newton
/// interface (`tangent dx = -residual`); `tangent` may return `None` for a combo that cannot
/// supply a Jacobian, and the Newton family falls back to a Picard step.
///
/// Every method takes `&mut self`: an equation counts its assemblies and caches, and the Python
/// implementations mutate on exactly these calls.
pub trait NonlinearSystem: Sized {
    /// The tangent / system operator, opaque to every strategy (see the module docs).
    type Matrix;
    /// Equation-specific extras carried on the assembled state.
    type Payload;

    /// A linear system converges after one solve; every strategy short-circuits on this.
    fn is_linear(&self) -> bool {
        false
    }

    fn initial_guess(&self) -> Vec<f64>;

    /// Assemble at `x`.
    ///
    /// # Errors
    /// Whatever the equation refuses with, plus [`NonlinearError::BudgetExceeded`] under a budget.
    fn assemble(&mut self, x: &[f64]) -> Result<StateOf<Self>, NonlinearError>;

    /// One Picard image `g(x)` from an assembly at `x`.
    ///
    /// # Errors
    /// As [`Self::assemble`].
    fn solve_linear(&mut self, state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError>;

    /// Residual `r(x)` whose root is the solution (Newton drives `tangent dx = -r`).
    ///
    /// # Errors
    /// As [`Self::assemble`].
    fn residual(&mut self, x: &[f64], state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError>;

    /// Scalar convergence measure a line search minimises — the norm over the FREE DOFs only.
    ///
    /// Defaults to `||residual||` (correct for MEC, whose reduced loop residual has no pinned
    /// rows). FEA overrides it to EXCLUDE the Dirichlet rows: the full-space FEA residual
    /// `S A - T` carries a large, constant residual on the Dirichlet nodes that the increment
    /// (`dA = 0` there) can never reduce, so an unfiltered `||r||` would make a backtracking line
    /// search reject every step. Only the free-DOF residual actually drops to zero.
    ///
    /// # Errors
    /// As [`Self::residual`].
    fn residual_norm(&mut self, x: &[f64], state: &StateOf<Self>) -> Result<f64, NonlinearError> {
        Ok(norm(&self.residual(x, state)?))
    }

    /// Free-DOF norm of the right-hand side, the scale of Newton's relative residual.
    ///
    /// `None` (the default) judges Newton against the seed residual, exact for a zero seed.
    ///
    /// # Errors
    /// As [`Self::assemble`].
    fn rhs_norm(&mut self, _state: &StateOf<Self>) -> Result<Option<f64>, NonlinearError> {
        Ok(None)
    }

    /// Optional Newton tangent operator; `None` makes a Newton strategy refuse the system.
    ///
    /// # Errors
    /// As [`Self::assemble`].
    fn tangent(
        &mut self,
        _x: &[f64],
        _state: &StateOf<Self>,
    ) -> Result<Option<Self::Matrix>, NonlinearError> {
        Ok(None)
    }

    /// Solve `matrix y = rhs` with the equation's own singular-system guard.
    ///
    /// `matrix` is whatever [`Self::tangent`] returned, passed back verbatim — so only
    /// Newton-family strategies that obtained it call this. Quasi-Newton strategies needing a
    /// SECANT solve of an arbitrary rhs use [`Self::solve_secant`], which every equation
    /// implements uniformly, because the secant `system_matrix` is not always in this method's
    /// accepted form (FEA's takes the reduced-tangent bundle, not a raw matrix).
    ///
    /// # Errors
    /// As [`Self::assemble`].
    fn solve_system(
        &mut self,
        matrix: &Self::Matrix,
        rhs: &[f64],
    ) -> Result<Vec<f64>, NonlinearError>;

    /// Solve the SECANT system `S y = rhs` for an arbitrary rhs, with BC reduction folded in.
    ///
    /// The uniform `H0` action the quasi-Newton family (Broyden / BFGS / L-BFGS) needs: `S` is the
    /// assembled secant operator at the current iterate, reduced through the equation's own
    /// Dirichlet / periodic / hanging machinery so `y` lives in the full DOF space. `solve_linear`
    /// is the special case `rhs = source`.
    ///
    /// # Errors
    /// As [`Self::assemble`]; [`NonlinearError::UnsupportedSystem`] when the state carries no
    /// operator to solve against.
    fn solve_secant(
        &mut self,
        state: &StateOf<Self>,
        rhs: &[f64],
    ) -> Result<Vec<f64>, NonlinearError> {
        let matrix =
            state
                .system_matrix
                .as_ref()
                .ok_or_else(|| NonlinearError::UnsupportedSystem {
                    strategy: "quasi-newton family",
                    need: "an assembled secant operator on the state (or a solve_secant override)"
                        .to_owned(),
                })?;
        // Cannot pass `&state.system_matrix` straight through: `solve_system` takes `&mut self`
        // and `state` is borrowed from the caller, not from `self`, so this is a clean split.
        self.solve_system(matrix, rhs)
    }

    /// Optional per-iteration hook (logging); default no-op.
    ///
    /// `increment` is `||x_next - x|| / ||x||`, the half of the convergence rule the energy error
    /// cannot express. Both halves are passed because a hook that saw only `err` reported a capped
    /// solve and a converged one identically.
    fn on_iteration(
        &mut self,
        _i: usize,
        _state: &StateOf<Self>,
        _err: f64,
        _alpha: f64,
        _increment: f64,
    ) {
    }

    /// Scale this system's SOURCE by `lam` (continuation stepping), IN PLACE.
    ///
    /// DEVIATES from the Python, which returns a rescaled COPY. Rust ownership makes "return
    /// another `Self`" a `Clone` bound on every equation, and the copy is used strictly
    /// sequentially and then discarded — so mutating and finishing at `lam = 1` walks the
    /// identical ramp. [`ContinuationStrategy`] restores `lam = 1` before it returns, so a system
    /// is never left scaled.
    ///
    /// # Errors
    /// [`NonlinearError::SourceNotScalable`] when the equation cannot cheaply rescale its source
    /// (conductor current + magnet Br); the strategy then degrades to solving the full problem.
    fn scale_source(&mut self, _lam: f64) -> Result<(), NonlinearError> {
        Err(NonlinearError::SourceNotScalable)
    }

    /// Whether this system exposes the raw pre-fold pair `nu-damped` needs.
    ///
    /// Asked as DATA rather than discovered as a failure: the Python's iteration-method matrix
    /// once generated a `nu-damped` cell for the MEC system, which cannot supply either half, and
    /// read the designed refusal as a red (heavy shard 1, 2026-08-29).
    fn supports_raw_mixing(&self) -> bool {
        false
    }

    /// The raw pre-boundary-fold arrays at `x`; `None` when this system has no such seam.
    ///
    /// # Errors
    /// As [`Self::assemble`].
    fn raw_state(&mut self, _x: &[f64]) -> Result<Option<RawState>, NonlinearError> {
        Ok(None)
    }

    /// Fold a (possibly mixed) raw state through the equation's Dirichlet / periodic machinery.
    ///
    /// # Errors
    /// As [`Self::assemble`].
    fn fold_raw(&mut self, _raw: &RawState) -> Result<Option<StateOf<Self>>, NonlinearError> {
        Ok(None)
    }
}

/// What a caller hands a strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct SolveOptions {
    pub max_iter: usize,
    pub rtol_energy_pct: f64,
    /// Warm start. `None` uses [`NonlinearSystem::initial_guess`].
    ///
    /// This is the Python's `_SeededSystem` wrapper, folded into the call: a wrapper that
    /// overrides one method is a delegation Rust cannot express without naming every other, and
    /// its only caller is `continuation`'s natural warm start.
    pub seed: Option<Vec<f64>>,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            rtol_energy_pct: RTOL_ENERGY_PCT,
            seed: None,
        }
    }
}

impl SolveOptions {
    #[must_use]
    pub fn new(max_iter: usize) -> Self {
        Self {
            max_iter,
            ..Self::default()
        }
    }

    /// The iterate a solve starts from.
    pub(crate) fn start<S: NonlinearSystem>(&self, system: &S) -> Vec<f64> {
        self.seed.clone().unwrap_or_else(|| system.initial_guess())
    }
}

/// A nonlinear iteration method.
pub trait IterationStrategy<S: NonlinearSystem> {
    /// The registry spelling.
    fn name(&self) -> &'static str;

    /// The system capabilities this strategy needs, or `&[]` when it needs nothing special.
    fn requires(&self) -> &'static [&'static str] {
        &[]
    }

    /// Drive `system` to its fixed point.
    ///
    /// # Errors
    /// Whatever the equation refuses with, or [`NonlinearError::UnsupportedSystem`].
    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError>;
}

// -- The convergence rule ---------------------------------------------------------------------

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

/// Energy-AND-flux convergence (fraction of a percent each), both below tolerance to pass.
///
/// A private copy of the FEA post-step check's formula — NOT a call into it: `gauss` sits BELOW
/// `hamilton` in the dependency arrow, so this module cannot reach it without inverting the
/// layering. [`NuRelaxationDampedPicardStrategy`] needs this two-metric check (not the
/// single-metric [`energy_error_pct`] the other strategies use) to reproduce the removed FEA
/// frozen loop's exact convergence test verbatim.
#[must_use]
pub(crate) fn energy_flux_convergence(
    total_energy: f64,
    total_flux: f64,
    last_energy: Option<f64>,
    last_flux: Option<f64>,
    rtol_energy_pct: f64,
    rtol_flux_pct: f64,
) -> (f64, f64, bool) {
    let (Some(le), Some(lf)) = (last_energy, last_flux) else {
        return (f64::INFINITY, f64::INFINITY, false);
    };
    let energy_denom = total_energy.abs().max(le.abs()).max(1e-12);
    let flux_denom = total_flux.abs().max(lf.abs()).max(1e-12);
    let e = (total_energy - le).abs() / energy_denom * 100.0;
    let fl = (total_flux - lf).abs() / flux_denom * 100.0;
    (e, fl, e < rtol_energy_pct && fl < rtol_flux_pct)
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
        // while diverging the whole time. MEASURED 2026-09-06 on `ipm_uv_cummins mec:quad` -
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
/// Symmetric with [`ALPHA_CALM_STEPS`] on purpose: the same amount of evidence that licenses a
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
/// noise-driven shrink is what pinned alpha at 0.108 on `ipm_uv_cummins`. But the recovery leg then
/// read the SAME deaf channel and drew the opposite conclusion from the same silence, so on an
/// energy plateau alpha grew every `ALPHA_CALM_STEPS` with nothing able to object.
///
/// MEASURED 2026-09-06 through `python -m motronics run cases/field/pm/ipm_uv_cummins --select
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

// -- The registry -----------------------------------------------------------------------------

/// The system capabilities strategy `name` needs, or `&[]`.
///
/// Every strategy runs on the generic `assemble`/`solve_linear` seam except `nu-damped`, which
/// mixes reluctivity BEFORE the boundary fold and so needs the raw pre-fold pair. That is a
/// capability boundary, not a convergence property: a system without the pair cannot run the
/// strategy at all, and asking beforehand is the difference between a matrix that declares a cell
/// unsupported and one that discovers it as a runtime refusal.
///
/// # Errors
/// [`NonlinearError::UnknownStrategy`].
pub fn strategy_requires(name: &str) -> Result<&'static [&'static str], NonlinearError> {
    if name == "nu-damped" {
        return Ok(&["raw_state", "fold_raw"]);
    }
    if KNOWN_STRATEGIES.contains(&name) {
        return Ok(&[]);
    }
    Err(NonlinearError::UnknownStrategy {
        name: name.to_owned(),
    })
}

/// Whether `system` supplies everything strategy `name` declares it needs.
///
/// Asked of the SYSTEM rather than of the discretization's name because a budget wrapper, a
/// decorator or a future system type all answer correctly by construction, while a list of solver
/// names would have to be maintained alongside the registry it describes.
///
/// # Errors
/// [`NonlinearError::UnknownStrategy`].
pub fn strategy_runs_on<S: NonlinearSystem>(
    name: &str,
    system: &S,
) -> Result<bool, NonlinearError> {
    Ok(strategy_requires(name)?.is_empty() || system.supports_raw_mixing())
}

/// Resolve an iteration-strategy name; `None` -> the default damped Picard (today's behaviour).
///
/// # Errors
/// [`NonlinearError::UnknownStrategy`].
pub fn get_strategy<S: NonlinearSystem + 'static>(
    name: Option<&str>,
) -> Result<Box<dyn IterationStrategy<S>>, NonlinearError> {
    // `None` is Newton, mirroring the Python `DEFAULT_ITERATION_STRATEGY` (2026-09-25).
    let name = name.unwrap_or("newton");
    let s: Box<dyn IterationStrategy<S>> = match name {
        "picard-damped" => Box::new(DampedPicardStrategy),
        "nu-damped" => Box::new(NuRelaxationDampedPicardStrategy),
        "picard-adaptive" => Box::new(AdaptiveDampedPicardStrategy::default()),
        "picard" => Box::new(PicardStrategy),
        "newton" => Box::new(NewtonStrategy::default()),
        "anderson" => Box::new(AndersonStrategy::default()),
        "modified-newton" => Box::new(ModifiedNewtonStrategy::default()),
        "quasi-newton" => Box::new(BroydenStrategy::default()),
        "bfgs" => Box::new(BfgsStrategy::default()),
        "l-bfgs" => Box::new(LbfgsStrategy::default()),
        "newton-linesearch" => Box::new(NewtonLineSearchStrategy::default()),
        "newton-krylov" => Box::new(NewtonKrylovStrategy::default()),
        "continuation" => Box::new(ContinuationStrategy::<S>::default()),
        _ => {
            return Err(NonlinearError::UnknownStrategy {
                name: name.to_owned(),
            })
        }
    };
    Ok(s)
}

#[cfg(test)]
mod tests;
