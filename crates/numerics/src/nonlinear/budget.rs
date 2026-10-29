//! The wall-clock budget: a system that refuses to START another expensive call once its time is
//! spent.

use super::{NonlinearError, NonlinearSystem, RawState, StateOf};

/// The calls the wrapper refuses to START once the budget is gone.
///
/// Declared as data so the scope claim in [`TimeBudgetedSystem`] is checkable instead of merely
/// asserted.
pub const BUDGETED_CALLS: [&str; 5] = [
    "assemble",
    "solve_linear",
    "solve_system",
    "solve_secant",
    "tangent",
];

/// A [`NonlinearSystem`] that refuses to START another expensive call once its time is spent.
///
/// WRAPPING THE SYSTEM, NOT THE STRATEGY, is the whole design. The cost lives in `solve_linear`
/// (for sdm, a dense truncated-SVD least squares) and `assemble`; putting the check there is
/// putting it where the money is spent, and it leaves all thirteen strategies and their trait
/// untouched. A budget threaded through thirteen loops would be thirteen places to forget it.
///
/// OVERSHOOT IS BOUNDED BY ONE CALL, said plainly because the alternative is a promise the code
/// cannot keep: no check at this level can unwind a call already inside a factorization. The
/// difference from a process-level timeout is that the refusal arrives as a typed error the caller
/// can attribute, instead of as a missing summary nobody can attribute to anything.
///
/// THE BUDGETED SET IS EVERY EXPENSIVE ENTRY POINT, not just the two the Picard default happens to
/// use: `assemble`, `solve_linear`, `solve_system`, `solve_secant` and `tangent`. Leaving the
/// Newton-family calls unbudgeted would put a hole in the guard exactly where a caller who
/// switched strategy would not think to look, and a scope claim with an unbudgeted path in it is
/// the "declaration that lies" shape. [`BUDGETED_CALLS`] names the set so a test can hold this
/// paragraph to it rather than trusting it.
///
/// Everything else delegates, so a budgeted system is the same system: `is_linear`,
/// `initial_guess`, `residual`, `on_iteration` and the raw-mixing pair all pass straight through —
/// none of them solves anything.
#[derive(Debug)]
pub struct TimeBudgetedSystem<S, C> {
    inner: S,
    budget_s: f64,
    clock: C,
    started: f64,
}

impl<S, C: Fn() -> f64> TimeBudgetedSystem<S, C> {
    /// Wrap `inner`, starting the clock now.
    pub fn new(inner: S, budget_s: f64, clock: C) -> Self {
        let started = clock();
        Self {
            inner,
            budget_s,
            clock,
            started,
        }
    }

    /// The system this one wraps — what a diagnostic should NAME.
    ///
    /// Present because a caller reaching through a private field is a caller that breaks when the
    /// private moves, and every diagnostic that wants to say WHICH system lacks a capability needs
    /// this. Measured 2026-08-29: a `coefficient-damped` refusal read "got `_TimeBudgetedSystem`" and sent
    /// the reader to the budget wrapper, which is not the reason the combination cannot run.
    pub fn unwrapped(&self) -> &S {
        &self.inner
    }

    fn check(&self, at: &'static str) -> Result<(), NonlinearError> {
        let elapsed = (self.clock)() - self.started;
        if elapsed > self.budget_s {
            return Err(NonlinearError::BudgetExceeded {
                budget_s: self.budget_s,
                elapsed_s: elapsed,
                at,
            });
        }
        Ok(())
    }
}

/// `system`, refusing to start another expensive call after `seconds`.
///
/// The Python returns the SAME OBJECT for a `None` budget, so that opting out is
/// indistinguishable from never having opted in. Rust says that with the type: no budget means no
/// wrapper, so a caller that passes none cannot be running through a wrapper that might behave
/// differently in some path nobody exercised — the caller simply holds `S`.
pub fn with_time_budget<S, C: Fn() -> f64>(
    system: S,
    seconds: f64,
    clock: C,
) -> TimeBudgetedSystem<S, C> {
    TimeBudgetedSystem::new(system, seconds, clock)
}

impl<S: NonlinearSystem, C: Fn() -> f64> NonlinearSystem for TimeBudgetedSystem<S, C> {
    type Matrix = S::Matrix;
    type Payload = S::Payload;

    fn is_linear(&self) -> bool {
        self.inner.is_linear()
    }

    fn initial_guess(&self) -> Vec<f64> {
        self.inner.initial_guess()
    }

    fn assemble(&mut self, x: &[f64]) -> Result<StateOf<Self>, NonlinearError> {
        self.check("assemble")?;
        self.inner.assemble(x)
    }

    fn solve_linear(&mut self, state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        self.check("solve_linear")?;
        self.inner.solve_linear(state)
    }

    fn residual(&mut self, x: &[f64], state: &StateOf<Self>) -> Result<Vec<f64>, NonlinearError> {
        self.inner.residual(x, state)
    }

    fn residual_norm(&mut self, x: &[f64], state: &StateOf<Self>) -> Result<f64, NonlinearError> {
        self.inner.residual_norm(x, state)
    }

    fn rhs_norm(&mut self, state: &StateOf<Self>) -> Result<Option<f64>, NonlinearError> {
        self.inner.rhs_norm(state)
    }

    fn local_values(&self, state: &StateOf<Self>) -> Result<super::LocalFields, NonlinearError> {
        self.inner.local_values(state)
    }

    fn set_linear_tolerance(&mut self, atol: f64) -> Result<(), NonlinearError> {
        self.inner.set_linear_tolerance(atol)
    }

    fn tangent(
        &mut self,
        x: &[f64],
        state: &StateOf<Self>,
    ) -> Result<Option<Self::Matrix>, NonlinearError> {
        self.check("tangent")?;
        self.inner.tangent(x, state)
    }

    fn solve_system(
        &mut self,
        matrix: &Self::Matrix,
        rhs: &[f64],
    ) -> Result<Vec<f64>, NonlinearError> {
        self.check("solve_system")?;
        self.inner.solve_system(matrix, rhs)
    }

    fn solve_secant(
        &mut self,
        state: &StateOf<Self>,
        rhs: &[f64],
    ) -> Result<Vec<f64>, NonlinearError> {
        self.check("solve_secant")?;
        self.inner.solve_secant(state, rhs)
    }

    fn on_iteration(
        &mut self,
        i: usize,
        state: &StateOf<Self>,
        err: f64,
        alpha: f64,
        increment: f64,
    ) {
        self.inner.on_iteration(i, state, err, alpha, increment);
    }

    fn scale_source(&mut self, lam: f64) -> Result<(), NonlinearError> {
        self.inner.scale_source(lam)
    }

    fn supports_raw_mixing(&self) -> bool {
        self.inner.supports_raw_mixing()
    }

    fn raw_state(&mut self, x: &[f64]) -> Result<Option<RawState>, NonlinearError> {
        self.check("assemble")?;
        self.inner.raw_state(x)
    }

    fn fold_raw(&mut self, raw: &RawState) -> Result<Option<StateOf<Self>>, NonlinearError> {
        self.inner.fold_raw(raw)
    }
}
