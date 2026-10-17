//! Time integrators — the Rust mirror of `python/numerics_lab/time_integrators/`.
//!
//! # The affine-weights contract, which is the whole design
//!
//! Every implicit derivative is AFFINE in the unknown `u_{n+1}`:
//!
//! ```text
//! u_arg   = alpha0 * u + h_u        (the state the residual is evaluated at)
//! u_dot   = alpha1 * u + h_dot
//! u_ddot  = alpha2 * u + h_ddot
//! ```
//!
//! The slopes are exactly the [`DerivativeWeights`] the jacobian consumes; the offsets are the
//! history. That is what lets one driver march BDF1, BDF2, Newmark and generalized-alpha
//! without knowing which it holds — an integrator contributes three numbers and three vectors,
//! never a solve.
//!
//! `u_arg` is the identity `u` for BDF and Newmark (`alpha0 = 1`, `h_u = 0`) and an
//! intermediate point for generalized-alpha, which is the only reason `alpha0` and `h_u` exist
//! at all.

use crate::weights::{DerivativeWeights, NEWMARK_BETA, NEWMARK_GAMMA};
use std::fmt;

/// Every way an integrator can refuse.
#[derive(Debug, Clone, PartialEq)]
pub enum IntegratorError {
    /// A spectral radius outside `[0, 1]`.
    SpectralRadiusOutOfRange { rho_inf: f64 },
    /// A second-order scheme asked to step from a state carrying no velocity/acceleration.
    MissingSecondOrderHistory { name: &'static str },
    /// A non-positive step.
    NonPositiveStep { dt: f64 },
    /// An unknown integrator name.
    UnknownKind { kind: String, known: &'static str },
    /// Vectors of disagreeing length.
    ShapeMismatch { expected: usize, got: usize },
}

impl fmt::Display for IntegratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpectralRadiusOutOfRange { rho_inf } => write!(
                f,
                "generalized-alpha spectral radius rho_inf must be in [0, 1], got {rho_inf}"
            ),
            Self::MissingSecondOrderHistory { name } => write!(
                f,
                "{name} is a second-order scheme and needs (u, v, a); this state carries no \
                 velocity/acceleration. Start it with `start(u0, t0, Some(v0), Some(a0))` -- \
                 defaulting them to zero here would silently change the trajectory."
            ),
            Self::NonPositiveStep { dt } => {
                write!(f, "the step dt must be positive; got {dt}")
            }
            Self::UnknownKind { kind, known } => {
                write!(f, "unknown integrator {kind:?}; known: {known}")
            }
            Self::ShapeMismatch { expected, got } => {
                write!(f, "expected a vector of length {expected}; got {got}")
            }
        }
    }
}

impl std::error::Error for IntegratorError {}

/// The marching history an integrator carries between steps.
///
/// `us` holds past state vectors most-recent-LAST (BDF2 reads the last two); `v`/`a` are the
/// velocity and acceleration at the last step — the second-order schemes propagate them, the
/// first-order schemes leave them `None`.
///
/// `h` is the spacing of the last step (`None` right after a start/restart): it IS history,
/// because a multistep form differentiates against it, so it lives here and is checkpointed and
/// rolled back with the vectors rather than tracked by each marcher.
#[derive(Debug, Clone, PartialEq)]
pub struct IntegratorState {
    pub us: Vec<Vec<f64>>,
    pub t: f64,
    pub v: Option<Vec<f64>>,
    pub a: Option<Vec<f64>>,
    pub h: Option<f64>,
}

impl IntegratorState {
    /// The most recent state vector.
    ///
    /// # Panics
    /// If the state was built with no history at all, which no constructor here permits.
    #[must_use]
    pub fn last_u(&self) -> &[f64] {
        self.us
            .last()
            .expect("an IntegratorState always carries at least u0")
    }
}

/// The affine per-step form an integrator hands the driver for the unknown `u_{n+1}`.
///
/// `t_eval` is the time the residual/source is sampled at — shifted by `alpha_f * dt` for
/// generalized-alpha, and `t_new` for everything else.
#[derive(Debug, Clone, PartialEq)]
pub struct StepForm {
    pub weights: DerivativeWeights,
    pub h_u: Vec<f64>,
    pub h_dot: Vec<f64>,
    pub h_ddot: Vec<f64>,
    pub t_eval: f64,
}

/// A time integrator: it supplies the affine step form and folds the solved state back into
/// its own history. It never solves.
///
/// `Debug` is a supertrait so a `Box<dyn TimeIntegrator>` can appear in a `Result` a test
/// unwraps -- and, more usefully, so a driver that panics mid-march names WHICH scheme it was
/// holding rather than an opaque pointer.
pub trait TimeIntegrator: fmt::Debug {
    /// The registry spelling.
    fn name(&self) -> &'static str;
    /// Formal order of accuracy.
    fn order(&self) -> u8;

    /// The jacobian weights for a step of size `dt`.
    ///
    /// # Errors
    /// [`IntegratorError::NonPositiveStep`].
    fn weights(&self, dt: f64) -> Result<DerivativeWeights, IntegratorError>;

    /// The initial state.
    ///
    /// # Errors
    /// [`IntegratorError::ShapeMismatch`] when `v0`/`a0` disagree with `u0`.
    fn start(
        &self,
        u0: Vec<f64>,
        t0: f64,
        v0: Option<Vec<f64>>,
        a0: Option<Vec<f64>>,
    ) -> Result<IntegratorState, IntegratorError>;

    /// The affine form for the next step.
    ///
    /// # Errors
    /// [`IntegratorError::NonPositiveStep`] or [`IntegratorError::MissingSecondOrderHistory`].
    fn prepare(
        &self,
        state: &IntegratorState,
        dt: f64,
        t_new: f64,
    ) -> Result<StepForm, IntegratorError>;

    /// Fold the solved `u_new` into the history.
    ///
    /// # Errors
    /// As [`Self::prepare`], plus [`IntegratorError::ShapeMismatch`].
    fn advance(
        &self,
        state: &IntegratorState,
        u_new: Vec<f64>,
        dt: f64,
        t_new: f64,
    ) -> Result<IntegratorState, IntegratorError>;
}

fn check_dt(dt: f64) -> Result<(), IntegratorError> {
    // NaN must be refused too, which is why this is not a bare `dt <= 0.0`.
    if dt.is_nan() || dt <= 0.0 {
        return Err(IntegratorError::NonPositiveStep { dt });
    }
    Ok(())
}

fn check_len(reference: &[f64], other: &[f64]) -> Result<(), IntegratorError> {
    if reference.len() != other.len() {
        return Err(IntegratorError::ShapeMismatch {
            expected: reference.len(),
            got: other.len(),
        });
    }
    Ok(())
}

/// The first-order start every scheme shares: history is `u0` alone.
fn start_first_order(u0: Vec<f64>, t0: f64) -> IntegratorState {
    IntegratorState {
        us: vec![u0],
        t: t0,
        v: None,
        a: None,
        h: None,
    }
}

/// The second-order start: `v0`/`a0` default to zero, matching the Python.
fn start_second_order(
    u0: Vec<f64>,
    t0: f64,
    v0: Option<Vec<f64>>,
    a0: Option<Vec<f64>>,
) -> Result<IntegratorState, IntegratorError> {
    let n = u0.len();
    let v = v0.unwrap_or_else(|| vec![0.0; n]);
    let a = a0.unwrap_or_else(|| vec![0.0; n]);
    check_len(&u0, &v)?;
    check_len(&u0, &a)?;
    Ok(IntegratorState {
        us: vec![u0],
        t: t0,
        v: Some(v),
        a: Some(a),
        h: None,
    })
}

/// Backward Euler (BDF1) — first order, unconditionally stable, for the `u_dot` (eddy) form.
///
/// `u_dot = (u_{n+1} - u_n)/dt` gives slope `alpha1 = 1/dt` and history `h_dot = -u_n/dt`;
/// `alpha2 = 0`. One implicit solve per step; local truncation error `O(dt^2)`, so the global
/// error is `O(dt)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bdf1;

impl TimeIntegrator for Bdf1 {
    fn name(&self) -> &'static str {
        "bdf1"
    }
    fn order(&self) -> u8 {
        1
    }

    fn weights(&self, dt: f64) -> Result<DerivativeWeights, IntegratorError> {
        check_dt(dt)?;
        Ok(DerivativeWeights::real(1.0, 1.0 / dt, 0.0))
    }

    fn start(
        &self,
        u0: Vec<f64>,
        t0: f64,
        _v0: Option<Vec<f64>>,
        _a0: Option<Vec<f64>>,
    ) -> Result<IntegratorState, IntegratorError> {
        Ok(start_first_order(u0, t0))
    }

    fn prepare(
        &self,
        state: &IntegratorState,
        dt: f64,
        t_new: f64,
    ) -> Result<StepForm, IntegratorError> {
        let w = self.weights(dt)?;
        let u_n = state.last_u();
        Ok(StepForm {
            weights: w,
            h_u: vec![0.0; u_n.len()],
            h_dot: u_n.iter().map(|&u| -u / dt).collect(),
            h_ddot: vec![0.0; u_n.len()],
            t_eval: t_new,
        })
    }

    fn advance(
        &self,
        _state: &IntegratorState,
        u_new: Vec<f64>,
        dt: f64,
        t_new: f64,
    ) -> Result<IntegratorState, IntegratorError> {
        check_dt(dt)?;
        Ok(IntegratorState {
            h: Some(dt),
            ..start_first_order(u_new, t_new)
        })
    }
}

/// The variable-spacing BDF2 step form; `h` is the PREVIOUS step's size.
///
/// ```text
/// u_dot = u * (2dt + h)/(dt(dt + h))  -  u_n * (dt + h)/(dt h)  +  u_{n-1} * dt/(h(dt + h))
/// ```
///
/// The derivative at `t_new` of the quadratic through three unevenly spaced points. At
/// `h == dt` it is exactly the uniform-weight BDF2, which is why [`Bdf2`] takes it only when the
/// spacings differ and an event-free march stays bit-identical. Fixed weights against an uneven
/// history are an O(1) WRONG EQUATION, not an order loss (`marching` records the measurement).
/// Reached only through `Bdf2::prepare`, so a marcher cannot choose a form: the state carries
/// the spacing and the integrator decides.
fn variable_spacing_bdf2(
    state: &IntegratorState,
    dt: f64,
    h: f64,
    t_new: f64,
) -> Result<StepForm, IntegratorError> {
    for step in [dt, h] {
        if step.is_nan() || step <= 0.0 {
            return Err(IntegratorError::NonPositiveStep { dt: step });
        }
    }
    let u_n = &state.us[state.us.len() - 1];
    let u_nm1 = &state.us[state.us.len() - 2];
    let c_n = -(dt + h) / (dt * h);
    let c_nm1 = dt / (h * (dt + h));
    Ok(StepForm {
        weights: DerivativeWeights::real(1.0, (2.0 * dt + h) / (dt * (dt + h)), 0.0),
        h_u: vec![0.0; u_n.len()],
        h_dot: u_n
            .iter()
            .zip(u_nm1)
            .map(|(&a, &b)| c_n * a + c_nm1 * b)
            .collect(),
        h_ddot: vec![0.0; u_n.len()],
        t_eval: t_new,
    })
}

/// Second-order backward difference (BDF2) — `u_dot = (3u_{n+1} - 4u_n + u_{n-1})/(2 dt)`.
///
/// Slope `alpha1 = 3/(2 dt)`, history `h_dot = (-4u_n + u_{n-1})/(2 dt)`; `alpha2 = 0`.
/// Uniform-step BDF2 needs two past states, so the FIRST step falls back to a single BDF1
/// step — the standard bootstrap, which keeps the scheme second order globally.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bdf2;

impl TimeIntegrator for Bdf2 {
    fn name(&self) -> &'static str {
        "bdf2"
    }
    fn order(&self) -> u8 {
        2
    }

    fn weights(&self, dt: f64) -> Result<DerivativeWeights, IntegratorError> {
        check_dt(dt)?;
        Ok(DerivativeWeights::real(1.0, 1.5 / dt, 0.0))
    }

    fn start(
        &self,
        u0: Vec<f64>,
        t0: f64,
        _v0: Option<Vec<f64>>,
        _a0: Option<Vec<f64>>,
    ) -> Result<IntegratorState, IntegratorError> {
        Ok(start_first_order(u0, t0))
    }

    fn prepare(
        &self,
        state: &IntegratorState,
        dt: f64,
        t_new: f64,
    ) -> Result<StepForm, IntegratorError> {
        if state.us.len() < 2 {
            // Bootstrap: not enough history yet -> one BDF1 step.
            return Bdf1.prepare(state, dt, t_new);
        }
        if let Some(h) = state.h {
            if h != dt {
                return variable_spacing_bdf2(state, dt, h, t_new);
            }
        }
        let w = self.weights(dt)?;
        let u_n = &state.us[state.us.len() - 1];
        let u_nm1 = &state.us[state.us.len() - 2];
        Ok(StepForm {
            weights: w,
            h_u: vec![0.0; u_n.len()],
            h_dot: u_n
                .iter()
                .zip(u_nm1)
                .map(|(&a, &b)| (-4.0 * a + b) / (2.0 * dt))
                .collect(),
            h_ddot: vec![0.0; u_n.len()],
            t_eval: t_new,
        })
    }

    fn advance(
        &self,
        state: &IntegratorState,
        u_new: Vec<f64>,
        dt: f64,
        t_new: f64,
    ) -> Result<IntegratorState, IntegratorError> {
        check_dt(dt)?;
        let mut us = state.us.clone();
        us.push(u_new);
        while us.len() > 2 {
            us.remove(0); // keep only the last two states
        }
        Ok(IntegratorState {
            us,
            t: t_new,
            v: None,
            a: None,
            h: Some(dt),
        })
    }
}

/// The Newmark end-point slopes and offsets for `a_{n+1}` and `v_{n+1}`.
///
/// Shared by [`Newmark`] and [`GeneralizedAlpha`], which differ only in WHERE the residual is
/// sampled, not in how the end-point state relates to the unknown.
fn newmark_form(
    state: &IntegratorState,
    dt: f64,
    beta: f64,
    gamma: f64,
    name: &'static str,
) -> Result<(f64, Vec<f64>, f64, Vec<f64>), IntegratorError> {
    check_dt(dt)?;
    let (Some(v_n), Some(a_n)) = (state.v.as_ref(), state.a.as_ref()) else {
        return Err(IntegratorError::MissingSecondOrderHistory { name });
    };
    let u_n = state.last_u();
    let a_slope = 1.0 / (beta * dt * dt);
    let a_off: Vec<f64> = (0..u_n.len())
        .map(|i| {
            -(u_n[i] + dt * v_n[i] + 0.5 * dt * dt * (1.0 - 2.0 * beta) * a_n[i]) / (beta * dt * dt)
        })
        .collect();
    let v_slope = gamma / (beta * dt);
    let v_off: Vec<f64> = (0..u_n.len())
        .map(|i| v_n[i] + dt * (1.0 - gamma) * a_n[i] + dt * gamma * a_off[i])
        .collect();
    Ok((a_slope, a_off, v_slope, v_off))
}

/// Newmark-beta for the second-order (structural `u_ddot`) system `M ddot + C dot + K u = f`.
///
/// The Newmark relations make `a_{n+1}` and `v_{n+1}` affine in the unknown `u_{n+1}`:
///
/// ```text
/// a_{n+1} = (u_{n+1} - u_n - dt v_n - dt^2/2 (1-2 beta) a_n) / (beta dt^2)
/// v_{n+1} = v_n + dt[(1-gamma) a_n + gamma a_{n+1}]
/// ```
///
/// so `alpha2 = 1/(beta dt^2)` and `alpha1 = gamma/(beta dt)`. The average-acceleration default
/// gives `alpha1 = 2/dt`, `alpha2 = 4/dt^2`.
#[derive(Debug, Clone, Copy)]
pub struct Newmark {
    pub beta: f64,
    pub gamma: f64,
}

impl Default for Newmark {
    fn default() -> Self {
        Self {
            beta: NEWMARK_BETA,
            gamma: NEWMARK_GAMMA,
        }
    }
}

impl TimeIntegrator for Newmark {
    fn name(&self) -> &'static str {
        "newmark"
    }
    fn order(&self) -> u8 {
        2
    }

    fn weights(&self, dt: f64) -> Result<DerivativeWeights, IntegratorError> {
        check_dt(dt)?;
        Ok(DerivativeWeights::real(
            1.0,
            self.gamma / (self.beta * dt),
            1.0 / (self.beta * dt * dt),
        ))
    }

    fn start(
        &self,
        u0: Vec<f64>,
        t0: f64,
        v0: Option<Vec<f64>>,
        a0: Option<Vec<f64>>,
    ) -> Result<IntegratorState, IntegratorError> {
        start_second_order(u0, t0, v0, a0)
    }

    fn prepare(
        &self,
        state: &IntegratorState,
        dt: f64,
        t_new: f64,
    ) -> Result<StepForm, IntegratorError> {
        let (_a_slope, a_off, _v_slope, v_off) =
            newmark_form(state, dt, self.beta, self.gamma, "newmark")?;
        Ok(StepForm {
            weights: self.weights(dt)?,
            h_u: vec![0.0; a_off.len()],
            h_dot: v_off,
            h_ddot: a_off,
            t_eval: t_new,
        })
    }

    fn advance(
        &self,
        state: &IntegratorState,
        u_new: Vec<f64>,
        dt: f64,
        t_new: f64,
    ) -> Result<IntegratorState, IntegratorError> {
        let (a_slope, a_off, v_slope, v_off) =
            newmark_form(state, dt, self.beta, self.gamma, "newmark")?;
        check_len(&u_new, &a_off)?;
        let a_new = u_new
            .iter()
            .zip(&a_off)
            .map(|(&u, &o)| a_slope * u + o)
            .collect();
        let v_new = u_new
            .iter()
            .zip(&v_off)
            .map(|(&u, &o)| v_slope * u + o)
            .collect();
        Ok(IntegratorState {
            us: vec![u_new],
            t: t_new,
            v: Some(v_new),
            a: Some(a_new),
            h: Some(dt),
        })
    }
}

/// Chung-Hulbert generalized-alpha — second order with controllable high-frequency damping.
///
/// The residual is evaluated at the intermediate states `u_{n+1-af}`, `v_{n+1-af}`,
/// `a_{n+1-am}` (each affine in the unknown, via the same Newmark relations) and the time is
/// sampled at `t_{n+1} - af*dt`. The effective jacobian weights are
///
/// ```text
/// alpha0 = 1 - af
/// alpha1 = (1 - af) gamma / (beta dt)
/// alpha2 = (1 - am) / (beta dt^2)
/// ```
///
/// with `am = (2 rho - 1)/(rho + 1)`, `af = rho/(rho + 1)`,
/// `beta = 1/4 (1 - am + af)^2`, `gamma = 1/2 - am + af` for a chosen spectral radius
/// `rho = rho_inf` in `[0, 1]`. `rho = 1` recovers undamped Newmark average-acceleration;
/// `rho = 0` gives asymptotic annihilation.
///
/// This is the only scheme for which `alpha0 != 1` and `h_u != 0`, and that is the reason both
/// exist in [`StepForm`] at all.
#[derive(Debug, Clone, Copy)]
pub struct GeneralizedAlpha {
    pub rho_inf: f64,
    alpha_m: f64,
    alpha_f: f64,
    beta: f64,
    gamma: f64,
}

impl Default for GeneralizedAlpha {
    fn default() -> Self {
        Self::new(0.8).expect("0.8 is in [0, 1]")
    }
}

impl GeneralizedAlpha {
    /// Build for a chosen spectral radius.
    ///
    /// # Errors
    /// [`IntegratorError::SpectralRadiusOutOfRange`].
    pub fn new(rho_inf: f64) -> Result<Self, IntegratorError> {
        if !(0.0..=1.0).contains(&rho_inf) {
            return Err(IntegratorError::SpectralRadiusOutOfRange { rho_inf });
        }
        let alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0);
        let alpha_f = rho_inf / (rho_inf + 1.0);
        Ok(Self {
            rho_inf,
            alpha_m,
            alpha_f,
            beta: 0.25 * (1.0 - alpha_m + alpha_f).powi(2),
            gamma: 0.5 - alpha_m + alpha_f,
        })
    }
}

impl TimeIntegrator for GeneralizedAlpha {
    fn name(&self) -> &'static str {
        "generalized_alpha"
    }
    fn order(&self) -> u8 {
        2
    }

    fn weights(&self, dt: f64) -> Result<DerivativeWeights, IntegratorError> {
        check_dt(dt)?;
        Ok(DerivativeWeights::real(
            1.0 - self.alpha_f,
            (1.0 - self.alpha_f) * self.gamma / (self.beta * dt),
            (1.0 - self.alpha_m) / (self.beta * dt * dt),
        ))
    }

    fn start(
        &self,
        u0: Vec<f64>,
        t0: f64,
        v0: Option<Vec<f64>>,
        a0: Option<Vec<f64>>,
    ) -> Result<IntegratorState, IntegratorError> {
        start_second_order(u0, t0, v0, a0)
    }

    fn prepare(
        &self,
        state: &IntegratorState,
        dt: f64,
        t_new: f64,
    ) -> Result<StepForm, IntegratorError> {
        let (af, am) = (self.alpha_f, self.alpha_m);
        let (_a_slope, a_off, _v_slope, v_off) =
            newmark_form(state, dt, self.beta, self.gamma, "generalized_alpha")?;
        let u_n = state.last_u();
        let v_n = state.v.as_ref().expect("newmark_form checked it");
        let a_n = state.a.as_ref().expect("newmark_form checked it");
        Ok(StepForm {
            weights: self.weights(dt)?,
            // u_{n+1-af} = (1-af) u_{n+1} + af u_n
            h_u: u_n.iter().map(|&u| af * u).collect(),
            // v_{n+1-af} = (1-af) v_{n+1} + af v_n
            h_dot: (0..u_n.len())
                .map(|i| (1.0 - af) * v_off[i] + af * v_n[i])
                .collect(),
            // a_{n+1-am} = (1-am) a_{n+1} + am a_n
            h_ddot: (0..u_n.len())
                .map(|i| (1.0 - am) * a_off[i] + am * a_n[i])
                .collect(),
            t_eval: t_new - af * dt,
        })
    }

    fn advance(
        &self,
        state: &IntegratorState,
        u_new: Vec<f64>,
        dt: f64,
        t_new: f64,
    ) -> Result<IntegratorState, IntegratorError> {
        let (a_slope, a_off, v_slope, v_off) =
            newmark_form(state, dt, self.beta, self.gamma, "generalized_alpha")?;
        check_len(&u_new, &a_off)?;
        let a_new = u_new
            .iter()
            .zip(&a_off)
            .map(|(&u, &o)| a_slope * u + o)
            .collect();
        let v_new = u_new
            .iter()
            .zip(&v_off)
            .map(|(&u, &o)| v_slope * u + o)
            .collect();
        Ok(IntegratorState {
            us: vec![u_new],
            t: t_new,
            v: Some(v_new),
            a: Some(a_new),
            h: Some(dt),
        })
    }
}

/// The registry's spellings, in one place so the error message and the resolver cannot drift.
pub const KNOWN_INTEGRATORS: [&str; 4] = ["bdf1", "bdf2", "newmark", "generalized_alpha"];

/// Resolve a kind to its implementation.
///
/// # Errors
/// [`IntegratorError::UnknownKind`].
pub fn get_integrator(kind: &str) -> Result<Box<dyn TimeIntegrator>, IntegratorError> {
    match kind {
        "bdf1" => Ok(Box::new(Bdf1)),
        "bdf2" => Ok(Box::new(Bdf2)),
        "newmark" => Ok(Box::new(Newmark::default())),
        "generalized_alpha" => Ok(Box::new(GeneralizedAlpha::default())),
        _ => Err(IntegratorError::UnknownKind {
            kind: kind.to_owned(),
            known: "bdf1, bdf2, generalized_alpha, newmark",
        }),
    }
}

#[cfg(test)]
mod tests;
