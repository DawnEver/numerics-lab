//! Pluggable nonlinear-iteration strategies — the Rust mirror of
//! `python/numerics_lab/nonlinear/`.
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

use std::fmt;

mod budget;
mod convergence;
mod local;
mod newton;
mod picard;
mod quasi;

pub use budget::{with_time_budget, TimeBudgetedSystem, BUDGETED_CALLS};
pub use convergence::{
    energy_error_pct, has_converged, has_converged_residual, increment_ratio, norm, IncrementTrend,
    StagnationWatch, INCREMENT_RISE_STEPS,
};
pub use local::{local_change, newton_linear_tolerance, LocalFields};
pub use newton::{
    ContinuationStrategy, HeldFactorNewtonStrategy, ModifiedNewtonStrategy, NewtonKrylovStrategy,
    NewtonLineSearchStrategy, NewtonStrategy,
};
pub use picard::{DampedPicardStrategy, CoefficientDampedPicardStrategy, PicardStrategy};
pub use quasi::{AndersonStrategy, BfgsStrategy, BroydenStrategy, LbfgsStrategy};

// -- The damped-Picard parameters, moved verbatim from `mec/_sparse.py` -----------------------

/// Floor for the damped under-relaxation factor.
pub const ALPHA_MIN: f64 = 0.1;
/// Damped-Picard recovery factor after a calm window.
pub const ALPHA_GROW: f64 = 1.5;
/// Convergence: total-energy change below this fraction of a percent -- the FIXED-POINT bar
/// (`1e-4` relative), the default of every strategy whose [`StopRule`] is [`StopRule::Energy`].
/// The Python's `core.fixed_point_rtol`.
pub const RTOL_ENERGY_PCT: f64 = 0.01;

/// The RESIDUAL-stopped (Newton-family) bar, in the same percent unit: `1e-6` relative.
///
/// The Python's `core.newton_rtol` (its own config field since the 2026-09-26 user ruling): at the
/// fixed-point `1e-4` Newton left `dA ~ 6e-5` on `case_d`, and a quadratic iteration or two
/// buys `1e-6`. [`default_rtol_energy_pct`] routes each strategy to one of the two by its declared
/// [`StopRule`], exactly as `hamilton.nonlinear_defaults.stop_rtol_energy_pct` does.
pub const NEWTON_RTOL_PCT: f64 = 1e-4;

/// What a strategy judges convergence on -- DECLARED by the strategy
/// ([`IterationStrategy::stop_rule`]), never listed beside it, so a Newton variant registered later
/// answers by construction. Mirrors the Python's `stop_rule` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopRule {
    /// The total-energy change (plus the increment floor): every fixed-point / quasi-Newton method.
    Energy,
    /// The equation's residual against `max(seed residual, rhs)` (plus the increment): Newton.
    Residual,
}

/// Consecutive UNDISTURBED steps (no energy-error growth, no energy-delta sign flip) before the
/// damped Picard loop grows `alpha` back.
///
/// The ratchet used to be one-way, so a single early overshoot damped the whole smooth tail:
/// measured on `case_a mec:quad`, alpha stuck at `1.0 * 0.5 * 0.6^3 = 0.108` and every
/// rotor position took 44-61 iterations at ~3 s each. Recovery is capped at `alpha_cap` — the
/// step size that last misbehaved is never retried — so a genuine limit cycle still settles
/// instead of oscillating between the two regimes.
pub const ALPHA_CALM_STEPS: usize = 3;

/// Consecutive steps of the stagnation window.
///
/// MEASURED (2026-09-01): on `case_b` newton's energy error passes rtol by
/// iteration ~12 while the increment pins at ~3.57e-4 — the stationary limit of the hanging-fold
/// update — so [`has_converged`]'s increment floor (`rtol/100` = 1e-4, 2-4.5x tighter than the
/// stationary increment on every measured case) is never met and the solve burns the whole
/// 100-step cap: ~85 wasted iterations, ~123 s. K must OUTLAST the documented false-plateau
/// hazard — the saturating-BH plateau that read 0.00% energy change for 20+ iterations (see
/// [`has_converged`]) — so K = 25; it fires on case_b newton at ~iteration 39.
pub const STAGNATION_WINDOW: usize = 25;

/// How flat the increment must be across the window to count as STATIONARY (max/min).
///
/// The false plateau's increment WANDERS, so this band — not the energy half — is what refuses it.
pub const STAGNATION_MAX_MIN_RATIO: f64 = 2.0;

/// Default wall-clock ceiling for ONE nonlinear solve, in seconds.
///
/// THE ARITHMETIC THAT PICKS IT is the TIER's, not the solve's. The budget is per solve and a
/// rotation sweep runs at least one per rotor position, so what has to hold is
/// `n_positions * budget < the test's own timeout`. `case_a` declares `n_positions = 7` and its
/// ripple sweep carries a 5400 s timeout: `7 * 500 = 3500` fits, `7 * 900 = 6300` does not — and
/// on Windows the firing timeout takes the WORKER, losing every result queued behind it. A budget
/// too GENEROUS is therefore not the safe direction; it is no budget at all.
///
/// THE MEASUREMENTS IT SITS BETWEEN: ~200 s per solve for the cases that complete, and 15-17 min
/// for the sdm case that hangs. Measured 2026-08-27, `case_r` (FEA-Galerkin) and
/// `case_h` (MEC Picard) both ran past 900 s standalone — neither an sdm solve, so an
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
    "newton-held-factor",
    "newton-krylov",
    "newton-linesearch",
    "coefficient-damped",
    "picard",
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

/// The pre-boundary-fold arrays `coefficient-damped` mixes.
///
/// See [`CoefficientDampedPicardStrategy`] for why the mixing has to happen on THESE arrays and
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

    /// Independent fields, each storing one component vector per element/DOF.
    /// Empty means the equation supplies no additional local convergence criterion.
    fn local_values(&self, _state: &StateOf<Self>) -> Result<LocalFields, NonlinearError> {
        Ok(Vec::new())
    }

    /// Absolute inner linear tolerance, scaled to the fixed original RHS reference.
    /// A direct-only equation may ignore this policy; iterative equations consume it.
    fn set_linear_tolerance(&mut self, _atol: f64) -> Result<(), NonlinearError> {
        Ok(())
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

    /// Whether this system exposes the raw pre-fold pair `coefficient-damped` needs.
    ///
    /// Asked as DATA rather than discovered as a failure: the Python's iteration-method matrix
    /// once generated a `coefficient-damped` cell for the MEC system, which cannot supply either half, and
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

    /// `max_iter` iterations at the default bar of strategy `name` ([`default_rtol_energy_pct`]).
    ///
    /// # Errors
    /// [`NonlinearError::UnknownStrategy`].
    pub fn for_strategy(name: &str, max_iter: usize) -> Result<Self, NonlinearError> {
        Ok(Self {
            max_iter,
            rtol_energy_pct: default_rtol_energy_pct(name)?,
            ..Self::default()
        })
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

    /// What this strategy stops on. The energy change unless the strategy declares otherwise.
    fn stop_rule(&self) -> StopRule {
        StopRule::Energy
    }

    /// Drive `system` to its fixed point.
    ///
    /// # Errors
    /// Whatever the equation refuses with, or [`NonlinearError::UnsupportedSystem`].
    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError>;
}

// -- The registry -----------------------------------------------------------------------------

/// The system capabilities strategy `name` needs, or `&[]`.
///
/// Every strategy runs on the generic `assemble`/`solve_linear` seam except `coefficient-damped`, which
/// mixes reluctivity BEFORE the boundary fold and so needs the raw pre-fold pair. That is a
/// capability boundary, not a convergence property: a system without the pair cannot run the
/// strategy at all, and asking beforehand is the difference between a matrix that declares a cell
/// unsupported and one that discovers it as a runtime refusal.
///
/// # Errors
/// [`NonlinearError::UnknownStrategy`].
pub fn strategy_requires(name: &str) -> Result<&'static [&'static str], NonlinearError> {
    if name == "coefficient-damped" {
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

/// A system with no equation behind it: only a strategy's DECLARATIONS are ever read through it.
struct Declarations;

impl NonlinearSystem for Declarations {
    type Matrix = ();
    type Payload = ();

    fn initial_guess(&self) -> Vec<f64> {
        Vec::new()
    }

    fn assemble(&mut self, _x: &[f64]) -> Result<StateOf<Self>, NonlinearError> {
        Err(NonlinearError::Equation("declarations only".to_owned()))
    }

    fn solve_linear(&mut self, _state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        Err(NonlinearError::Equation("declarations only".to_owned()))
    }

    fn residual(&mut self, _x: &[f64], _state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        Err(NonlinearError::Equation("declarations only".to_owned()))
    }

    fn solve_system(&mut self, _matrix: &(), _rhs: &[f64]) -> Result<Vec<f64>, NonlinearError> {
        Err(NonlinearError::Equation("declarations only".to_owned()))
    }
}

/// Whether strategy `name` judges convergence on the equation's residual (the Newton family).
///
/// Read off the strategy's own [`IterationStrategy::stop_rule`]; the port of the Python
/// `strategy_stops_on_residual`.
///
/// # Errors
/// [`NonlinearError::UnknownStrategy`].
pub fn strategy_stops_on_residual(name: &str) -> Result<bool, NonlinearError> {
    Ok(get_strategy::<Declarations>(Some(name))?.stop_rule() == StopRule::Residual)
}

/// The default stop bar of strategy `name`, in percent: [`NEWTON_RTOL_PCT`] for a residual stop,
/// [`RTOL_ENERGY_PCT`] otherwise. A caller with its own bar (a golden above all) states it.
///
/// # Errors
/// [`NonlinearError::UnknownStrategy`].
pub fn default_rtol_energy_pct(name: &str) -> Result<f64, NonlinearError> {
    Ok(if strategy_stops_on_residual(name)? {
        NEWTON_RTOL_PCT
    } else {
        RTOL_ENERGY_PCT
    })
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
        "coefficient-damped" => Box::new(CoefficientDampedPicardStrategy),
        "picard" => Box::new(PicardStrategy),
        "newton" => Box::new(NewtonStrategy::default()),
        "anderson" => Box::new(AndersonStrategy::default()),
        "modified-newton" => Box::new(ModifiedNewtonStrategy::default()),
        "quasi-newton" => Box::new(BroydenStrategy::default()),
        "bfgs" => Box::new(BfgsStrategy::default()),
        "l-bfgs" => Box::new(LbfgsStrategy::default()),
        "newton-linesearch" => Box::new(NewtonLineSearchStrategy::default()),
        "newton-held-factor" => Box::new(HeldFactorNewtonStrategy::default()),
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
