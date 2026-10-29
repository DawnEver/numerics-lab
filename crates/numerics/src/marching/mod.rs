//! Event location, checkpoint/rollback and the variable-step BDF2 form — the Rust mirror of
//! `python/numerics_lab/time_integrators/events.py` and its driver's step machinery.
//!
//! A switch, a comparator, a contact: several physics have instants where the problem itself
//! changes, and a fixed-grid march past such an instant integrates a smeared version of it.
//! This module adds the machinery in the only shape `gauss` may honestly carry it — everything
//! here is OPAQUE to what it is marching:
//!
//! * **checkpoint/rollback of opaque state** — an [`IntegratorState`] plus the solution length
//!   and the last step size taken at the same instant. `gauss` owns the integrator history, so
//!   it owns rolling it back; the caller's own state (a device's on/off flag) is the caller's
//!   to snapshot through its own seam. Neither half knows anything about the other's.
//! * **a root-find on an opaque scalar event function** `g(u, t) -> f64`, whose zero is "an
//!   instant worth stopping at". The locator brackets a sign change across a step, then
//!   bisects by RE-STEPPING from the checkpoint with a candidate step size — the probe states
//!   are marched, never interpolated, so the located state is one the integrator could
//!   actually have produced.
//!
//! # The nonuniform-history trap, and why this module carries its own BDF2 step form
//!
//! Landing on an event at `t*` and marching on means steps whose size differs from the spacing
//! of the history they differentiate against. Fixed-weight BDF2 assumes uniform spacing: with
//! step `dt` and history spacing `h` its derivative coefficient is off by a factor
//! `(3dt - h)/(2dt)` — an O(1) WRONG EQUATION, not an order reduction. It was measured in
//! development as a probe bisection converging to the bracket edge, locating a 1.0 s crossing
//! at 0.9 s.
//!
//! So any step whose size differs from the previous spacing is prepared with the standard
//! VARIABLE-step BDF2 formula, the derivative of the quadratic through three unevenly spaced
//! points (`Bdf2::prepare` reads it off `IntegratorState::h`), which reduces to the uniform weights at `h == dt` exactly.
//! Only BDF2 carries the assumption: BDF1 has no second state, and the second-order structural
//! schemes propagate `v`/`a` with formulas already valid at arbitrary `dt`. The variable form
//! is used ONLY when the spacings differ, so an event-free march is bit-identical to a plain
//! one.
//!
//! # The companion trap, which the variable form does NOT fix
//!
//! An event is an instant where the problem CHANGES, so the pre-event history estimates the
//! post-event derivative with contaminated data — the quadratic through the pre-toggle states
//! drags the OLD slope into the first post-event step. Measured on a slope-toggling ramp: the
//! step after the toggle errs by +9.5% of the true value; on a switched buck the inductor
//! ripple loses 1.4%.
//!
//! The honest treatment of a model discontinuity is to RESTART the history at the event. The
//! first post-event step is then BDF1, whose estimate `(u_new - u*) / h` carries no pre-event
//! slope at all — exact for the piecewise-linear-between-events physics a switch produces, and
//! one bounded O(dt^2) step for smooth physics. A ceremonial event (a callback that changes
//! nothing) pays that one BDF1 step for no benefit; that is the price of the semantics,
//! stated rather than hidden. For a second-order integrator the restart CARRIES the landing
//! step's `v`/`a` — continuous state is preserved, only the derivative HISTORY is dropped.

use crate::integrators::{IntegratorError, IntegratorState, TimeIntegrator};
use std::fmt;

/// Events tolerated inside ONE nominal cell before the function is declared chattering.
///
/// A genuine circuit sees a handful of crossings per cell at most (a PWM edge and a state
/// ripple); past this the event function is re-triggering on its own zero, which no march can
/// progress past. Sized for the continuous-PI reproduction, where a drive samples its PI at
/// the march's step rate — ~200 legitimately-located samples per nominal cell, a dense but
/// genuine event stream the locator resolves rather than chatter.
pub const MAX_EVENTS_PER_CELL: usize = 512;

/// Every way a march can refuse.
#[derive(Debug, Clone, PartialEq)]
pub enum MarchError {
    /// The event function re-triggered on its own zero past the ceiling.
    Chattering {
        function: usize,
        events_in_cell: usize,
        ceiling: usize,
    },
    /// A bisection that could not narrow the bracket — a guard against an infinite loop, and
    /// a real answer: the crossing is not locatable to the requested tolerance.
    BisectionExhausted { function: usize, iterations: usize },
    /// A step the integrator itself refused.
    Integrator(IntegratorError),
}

impl fmt::Display for MarchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chattering {
                function,
                events_in_cell,
                ceiling,
            } => write!(
                f,
                "event function {function} located {events_in_cell} crossings inside one \
                 nominal cell (ceiling {ceiling}); it is re-triggering on its own zero, which \
                 no march can progress past. Widen the function's hysteresis rather than the \
                 ceiling -- a ceiling raised to admit chatter admits a march that never ends."
            ),
            Self::BisectionExhausted {
                function,
                iterations,
            } => write!(
                f,
                "event function {function}: the bracket did not narrow in {iterations} \
                 bisections, so the crossing is not locatable to the requested tolerance"
            ),
            Self::Integrator(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MarchError {}

impl From<IntegratorError> for MarchError {
    fn from(e: IntegratorError) -> Self {
        Self::Integrator(e)
    }
}

/// The march state at an instant, deep enough that probing past it cannot alias into it.
///
/// The previous spacing is part of the HISTORY — BDF2's weights read it — and it travels inside
/// the state (`IntegratorState::h`), so a rollback cannot restore the right numbers against the
/// wrong spacing: there is no second copy of the spacing to forget.
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    state: IntegratorState,
    n_points: usize,
}

impl Checkpoint {
    /// Snapshot the integrator history.
    ///
    /// The vectors are CLONED. Rust would not let a rollback restore an alias the way Python
    /// can, but the clone is still load-bearing: a probe step mutates its own copy, and a
    /// checkpoint sharing storage with the live march would be rewritten by the thing it
    /// exists to undo.
    #[must_use]
    pub fn take(state: &IntegratorState, n_points: usize) -> Self {
        Self {
            state: state.clone(),
            n_points,
        }
    }

    /// The history to resume from, its spacing included.
    #[must_use]
    pub fn restore(&self) -> IntegratorState {
        self.state.clone()
    }

    /// How many solution points existed when this checkpoint was taken — everything past it is
    /// retracted on rollback, because a retracted point is no longer a step OF the solution.
    #[must_use]
    pub const fn n_points(&self) -> usize {
        self.n_points
    }
}

/// The bisection's sign bit: `g >= 0`. A landing exactly on zero reads as the FAR side, so a
/// probe that lands precisely on the crossing is accepted rather than bisected forever.
#[must_use]
pub fn event_sign(value: f64) -> bool {
    value >= 0.0
}

/// One located crossing: which event function, and when.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Event {
    pub function: usize,
    pub time: f64,
}

/// How hard the locator is allowed to work, and how close it must get.
///
/// A struct rather than two more positional floats: `tol` and a probe count are both numbers,
/// and a call site that transposed them would compile and then locate every crossing at the
/// bracket edge — the exact failure mode the variable-step form above exists to prevent, wearing
/// a different hat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bisection {
    /// Bracket width at which the crossing is considered located, in units of TIME.
    pub tol: f64,
    /// Probes allowed before the crossing is declared unlocatable. Bisection halves the bracket
    /// each probe, so this is `log2(dt_full / tol)` rounded up, plus headroom.
    pub max_probes: usize,
}

impl Default for Bisection {
    fn default() -> Self {
        Self {
            tol: 1e-12,
            max_probes: 200,
        }
    }
}

/// Locate the earliest crossing of any event function inside a step, by RE-STEPPING.
///
/// `step_to(dt) -> u` marches one candidate step of size `dt` from the checkpointed history and
/// returns the resulting state; `g(u, t, i)` evaluates event function `i`. The caller owns both,
/// and this function learns nothing about either.
///
/// Bisection here is on the STEP SIZE, not on an interpolant. That is the whole point: the
/// located state is one the integrator could actually have produced, so the caller's `on_event`
/// sees a state consistent with the scheme rather than a linear blend of two that straddle it.
///
/// Returns `None` when no function changes sign across the full step.
///
/// # Errors
/// [`MarchError::BisectionExhausted`] when the bracket does not narrow to `tol` within
/// `max_bisections`.
pub fn locate_event<S, G>(
    t0: f64,
    dt_full: f64,
    g_at_start: &[f64],
    mut step_to: S,
    mut g: G,
    n_functions: usize,
    bisection: Bisection,
) -> Result<Option<Event>, MarchError>
where
    S: FnMut(f64) -> Result<Vec<f64>, MarchError>,
    G: FnMut(&[f64], f64, usize) -> f64,
{
    let u_end = step_to(dt_full)?;
    let t_end = t0 + dt_full;
    // The earliest crossing wins: two functions can both flip across one step and only the
    // first is a step the march may take.
    let crossed: Vec<usize> = (0..n_functions)
        .filter(|&i| event_sign(g_at_start[i]) != event_sign(g(&u_end, t_end, i)))
        .collect();
    if crossed.is_empty() {
        return Ok(None);
    }

    let mut best: Option<Event> = None;
    for i in crossed {
        let (mut lo, mut hi) = (0.0_f64, dt_full);
        let sign_lo = event_sign(g_at_start[i]);
        let mut narrowed = false;
        for _ in 0..bisection.max_probes {
            if hi - lo <= bisection.tol {
                narrowed = true;
                break;
            }
            let mid = 0.5 * (lo + hi);
            let u_mid = step_to(mid)?;
            if event_sign(g(&u_mid, t0 + mid, i)) == sign_lo {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if !narrowed && hi - lo > bisection.tol {
            return Err(MarchError::BisectionExhausted {
                function: i,
                iterations: bisection.max_probes,
            });
        }
        // `hi` is the first step size at which the sign has FLIPPED, so it is the crossing.
        let candidate = Event {
            function: i,
            time: t0 + hi,
        };
        if best.is_none_or(|b| candidate.time < b.time) {
            best = Some(candidate);
        }
    }
    Ok(best)
}

/// Counts crossings inside one nominal cell and refuses a function that re-triggers on its own
/// zero.
///
/// A ceiling rather than a bare counter, because the failure it catches is unbounded: without
/// it a chattering function makes the march take smaller and smaller steps forever, which reads
/// from outside as a hang rather than as a wrong answer.
#[derive(Debug, Clone)]
pub struct ChatterGuard {
    ceiling: usize,
    cell_index: usize,
    seen: usize,
}

impl Default for ChatterGuard {
    fn default() -> Self {
        Self::new(MAX_EVENTS_PER_CELL)
    }
}

impl ChatterGuard {
    #[must_use]
    pub const fn new(ceiling: usize) -> Self {
        Self {
            ceiling,
            cell_index: 0,
            seen: 0,
        }
    }

    /// Record one located event in nominal cell `cell_index`.
    ///
    /// # Errors
    /// [`MarchError::Chattering`] once one cell exceeds the ceiling.
    pub fn observe(&mut self, cell_index: usize, function: usize) -> Result<(), MarchError> {
        if cell_index != self.cell_index {
            self.cell_index = cell_index;
            self.seen = 0;
        }
        self.seen += 1;
        if self.seen > self.ceiling {
            return Err(MarchError::Chattering {
                function,
                events_in_cell: self.seen,
                ceiling: self.ceiling,
            });
        }
        Ok(())
    }

    /// Events counted so far in the current cell — the number the error would report.
    #[must_use]
    pub const fn seen_in_cell(&self) -> usize {
        self.seen
    }
}

/// Restart the history at an event, preserving continuous state and dropping the derivative
/// history.
///
/// This is the companion-trap treatment from the module docs, as a function rather than as a
/// paragraph: the first post-event step becomes BDF1, so no pre-event slope reaches it. For a
/// second-order integrator `v`/`a` are CARRIED — the physical state did not jump, only the
/// model did.
///
/// # Errors
/// Whatever the integrator's `start` refuses.
pub fn restart_at_event(
    integrator: &dyn TimeIntegrator,
    landed: &IntegratorState,
    t_event: f64,
) -> Result<IntegratorState, IntegratorError> {
    integrator.start(
        landed.last_u().to_vec(),
        t_event,
        landed.v.clone(),
        landed.a.clone(),
    )
}

#[cfg(test)]
mod tests;
