//! Newton-family, quasi-Newton and continuation iteration strategies.

use super::{
    energy_error_pct, has_converged, has_converged_residual, increment_ratio, local_change,
    newton_linear_tolerance, norm, DampedPicardStrategy, IterationResult, IterationStrategy,
    NonlinearError, NonlinearSystem, ResultOf, SolveOptions, StagnationWatch, StateOf, StopRule,
};

/// `a + s * b`, elementwise.
pub(crate) fn axpy(a: &[f64], s: f64, b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(p, q)| p + s * q).collect()
}

pub(crate) fn neg(v: &[f64]) -> Vec<f64> {
    v.iter().map(|a| -a).collect()
}

pub(crate) fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(p, q)| p - q).collect()
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// The scale Newton's relative residual is taken against: `max(||r(x_0)||, ||rhs||)`.
///
/// Mirrors the Python `_residual_reference`: the seed residual alone is the right-hand side only
/// for a ZERO seed, so a warm start near the root would otherwise be held to a bar that tightens
/// the better the guess. A system that knows its free-DOF source norm returns it from
/// [`NonlinearSystem::rhs_norm`]; one that does not is judged against its seed residual.
fn residual_reference<S: NonlinearSystem>(
    system: &mut S,
    state: &StateOf<S>,
    seed_norm: f64,
) -> Result<f64, NonlinearError> {
    Ok(match system.rhs_norm(state)? {
        Some(rhs) => seed_norm.max(rhs),
        None => seed_norm,
    })
}

/// Newton-Raphson with a backtracking line search -- THE Newton every family runs.
///
/// `J(x_k) dx = -r(x_k)`, then `x_{k+1} = x_k + lam dx` with the first `lam` in
/// `1, shrink, shrink^2, ...` meeting the Armijo decrease `||r(x + lam dx)|| <= (1 - c lam)
/// ||r(x)||` on the free-DOF residual norm. No accepted `lam` takes the FULL step (plain Newton's
/// move, so the search can only ever help). Convergence is the relative residual
/// `||r(x_k)|| / max(||r(x_0)||, ||rhs||)` AND the increment, both below `rtol_energy_pct / 100`
/// ([`has_converged_residual`]); the stagnation watch stays as the floor escape. A system with no
/// tangent is REFUSED ([`NonlinearError::UnsupportedSystem`]) -- the Picard step this used to take
/// ran a different algorithm under Newton's name. Port of the Python `NewtonStrategy` of
/// 2026-09-25; the golden trajectories are regenerated from it.
#[derive(Debug, Clone, Copy)]
pub struct NewtonStrategy {
    pub max_backtracks: usize,
    pub shrink: f64,
    pub armijo: f64,
    /// `None` is full Newton; `Some(k)` HOLDS the tangent for at most `k` iterations (the
    /// modified Newton, `newton-held-factor`, user ruling 2026-09-26). Mirrors the Python field.
    pub refactor_every: Option<usize>,
    /// A held tangent is refreshed for the next iteration when a step contracts the residual by
    /// less than this.
    pub stall_ratio: f64,
}

impl Default for NewtonStrategy {
    fn default() -> Self {
        Self {
            max_backtracks: 12,
            shrink: 0.5,
            armijo: 1e-4,
            refactor_every: None,
            stall_ratio: 0.5,
        }
    }
}

/// One Armijo search: `(lam, accepted, full-step trial, full-step norm)`.
type Search<S> = (
    f64,
    Option<(Vec<f64>, StateOf<S>, f64)>,
    Option<Vec<f64>>,
    f64,
);

impl NewtonStrategy {
    fn tangent_of<S: NonlinearSystem>(
        name: &'static str,
        system: &mut S,
        x: &[f64],
        state: &StateOf<S>,
    ) -> Result<S::Matrix, NonlinearError> {
        system.tangent(x, state)?.ok_or_else(|| NonlinearError::UnsupportedSystem {
            strategy: name,
            need: "a tangent (the Jacobian of the residual); select a fixed-point strategy by name instead"
                .to_owned(),
        })
    }

    fn search<S: NonlinearSystem>(
        &self,
        system: &mut S,
        x: &[f64],
        dx: &[f64],
        r_norm: f64,
    ) -> Result<Search<S>, NonlinearError> {
        let mut lam = 1.0_f64;
        let mut full_norm = f64::NAN;
        let mut full_trial: Option<Vec<f64>> = None;
        for _ in 0..=self.max_backtracks {
            let trial = axpy(x, lam, dx);
            let trial_state = system.assemble(&trial)?;
            let trial_norm = system.residual_norm(&trial, &trial_state)?;
            if full_trial.is_none() {
                full_trial = Some(trial.clone());
                full_norm = trial_norm;
            }
            if trial_norm.is_finite() && trial_norm <= (1.0 - self.armijo * lam) * r_norm {
                return Ok((
                    lam,
                    Some((trial, trial_state, trial_norm)),
                    full_trial,
                    full_norm,
                ));
            }
            lam *= self.shrink;
        }
        Ok((lam, None, full_trial, full_norm))
    }

    fn drive<S: NonlinearSystem>(
        &self,
        name: &'static str,
        system: &mut S,
        opts: &SolveOptions,
    ) -> Result<ResultOf<S>, NonlinearError> {
        let rtol = opts.rtol_energy_pct;
        let mut x = opts.start(system);
        if opts.max_iter == 0 {
            // NO ITERATION RUNS NO ASSEMBLY, the contract every strategy keeps.
            return Ok(IterationResult {
                x,
                state: None,
                iterations: 0,
                converged: false,
            });
        }
        let mut state = system.assemble(&x)?;
        let mut r_norm = system.residual_norm(&x, &state)?;
        let reference = residual_reference(system, &state, r_norm)?;
        if r_norm == 0.0 || !reference.is_finite() {
            return Ok(IterationResult {
                x,
                state: Some(state),
                iterations: 1,
                converged: r_norm == 0.0,
            });
        }
        let mut last_energy = Some(state.total_energy);
        // THE TRACE STARTS AT THE SEED (user ruling 2026-10-04): `on_iteration` reports the state an
        // iteration STARTS from, beside the step out of it -- the Python `NewtonStrategy.solve`.
        // `start_err` is that state's own energy change against its predecessor, `inf` for the seed.
        let mut start_err = f64::INFINITY;
        let mut converged = false;
        let mut ran = 0;
        let mut watch = StagnationWatch::new(rtol);
        let mut held: Option<S::Matrix> = None;
        let mut age = 0_usize;
        let mut stale = false;
        for i in 0..opts.max_iter {
            ran = i + 1;
            let previous_local = system.local_values(&state)?;
            system
                .set_linear_tolerance(reference * newton_linear_tolerance(r_norm / reference)?)?;
            // FULL NEWTON refreshes every iteration; the held (modified) one when its tangent is
            // `k` old or the last step stalled -- the Python `NewtonStrategy.solve`, rule for rule.
            let fresh = held.is_none() || self.refactor_every.is_none_or(|k| age >= k) || stale;
            if fresh {
                held = Some(Self::tangent_of(name, system, &x, &state)?);
                age = 0;
            }
            let r = system.residual(&x, &state)?;
            let rhs = neg(&r);
            let dx = system.solve_system(held.as_ref().expect("refreshed above"), &rhs)?;
            let x_prev = x.clone();
            let (mut lam, mut accepted, mut full_trial, mut full_norm) =
                self.search(system, &x, &dx, r_norm)?;
            if !(fresh || accepted.is_some() && lam == 1.0) {
                // A held tangent that needs a line search is no longer a Newton model here:
                // refresh and retake the step.
                held = Some(Self::tangent_of(name, system, &x, &state)?);
                age = 0;
                let dx = system.solve_system(held.as_ref().expect("refreshed above"), &rhs)?;
                (lam, accepted, full_trial, full_norm) = self.search(system, &x, &dx, r_norm)?;
            }
            age += 1;
            let (next, next_state, next_norm) = if let Some(hit) = accepted {
                hit
            } else {
                // RE-ASSEMBLED at the full step: the last assembly was the smallest trial, and
                // an equation may keep per-element state from its latest assembly.
                lam = 1.0;
                let full = full_trial.expect("the search assembles at least one trial");
                let full_state = system.assemble(&full)?;
                (full, full_state, full_norm)
            };
            // A NaN norm counts as stale: only a norm that compares <= the bar keeps the factor.
            stale = self.refactor_every.is_some()
                && !matches!(
                    next_norm.partial_cmp(&(self.stall_ratio * r_norm)),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                );
            x = next;
            r_norm = next_norm;
            let err = energy_error_pct(next_state.total_energy, last_energy);
            last_energy = Some(next_state.total_energy);
            let increment = increment_ratio(&x_prev, &x);
            converged = has_converged_residual(&x_prev, &x, rtol, r_norm / reference)
                || watch.observe(err, increment);
            if system.is_linear() {
                converged = true;
            }
            let current_local = system.local_values(&next_state)?;
            if !system.is_linear() && (!previous_local.is_empty() || !current_local.is_empty()) {
                converged =
                    converged && local_change(&previous_local, &current_local)? < rtol / 100.0;
            }
            system.on_iteration(i, &state, start_err, lam, increment);
            state = next_state;
            start_err = err;
            if converged {
                break;
            }
        }
        Ok(IterationResult {
            x,
            state: Some(state),
            iterations: ran,
            converged,
        })
    }
}

impl<S: NonlinearSystem> IterationStrategy<S> for NewtonStrategy {
    fn name(&self) -> &'static str {
        "newton"
    }

    fn stop_rule(&self) -> StopRule {
        StopRule::Residual
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        self.drive("newton", system, opts)
    }
}

/// Newton reusing the cached tangent for `refresh_every` steps.
///
/// A full Newton re-forms and re-factorizes `tangent` every step; the modified Newton holds the
/// Jacobian fixed for `k` residual/solve steps between refreshes, trading quadratic convergence
/// for a much cheaper step. Uses only `tangent` / `residual` / `solve_system`; falls back to a
/// Picard step when `tangent` returns `None`. Converges to the same root as Newton; same
/// delayed-break energy bookkeeping.
///
/// MANUAL-20260721-419: a HELD tangent step is only amortized cheaply while it is still a good
/// local model. On the sharper-curvature FEMM-matching BH spline the tangent goes stale mid-hold
/// and the step over-shoots into a spurious root — a near-zero field on the sdm map, an outright
/// energy blow-up on MEC/FEA (`newton` / `newton-linesearch`, which refresh every step, land the
/// correct root on the same maps). So every HELD (non-refresh) step is checked against the RAW
/// (unfiltered) `residual` norm before/after: if the step would GROW `||residual||`, the held
/// tangent is stale for the current curvature — refresh it right there and retake the step with
/// the fresh one, instead of marching toward the spurious root.
///
/// Deliberately the raw residual, not [`NonlinearSystem::residual_norm`]: that hook is
/// scale-RELATIVE, which is exactly blind to this failure mode — once a stale step has pushed `x`
/// far outside the physical range, `g(x)` saturates to a value that no longer scales with `x`, so
/// the RATIO plateaus near a fixed asymptote even as the ABSOLUTE residual keeps growing without
/// bound (measured: a stale sdm step took `||x||` 1.6e3 -> 1.4e8 while the relative residual read
/// a near-constant ~1.0 throughout). The raw norm has no such blind spot.
///
/// NOT PORTED: the Python caches an `splu` FACTOR of the held tangent and reuses it across the
/// held steps. Its own docstring records that this is bit-for-bit the iterate a per-step solve
/// produces — it buys TIME, not accuracy — so the port calls `solve_system` on the held tangent
/// each step and reaches the identical sequence. The factor cache belongs with the factorization,
/// which lives in `numerics::linear`, not here.
#[derive(Debug, Clone, Copy)]
pub struct ModifiedNewtonStrategy {
    pub refresh_every: usize,
}

impl Default for ModifiedNewtonStrategy {
    fn default() -> Self {
        Self { refresh_every: 3 }
    }
}

impl<S: NonlinearSystem> IterationStrategy<S> for ModifiedNewtonStrategy {
    fn name(&self) -> &'static str {
        "modified-newton"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        let rtol = opts.rtol_energy_pct;
        let every = self.refresh_every.max(1);
        let mut x = opts.start(system);
        let mut state: Option<StateOf<S>> = None;
        let mut last_energy: Option<f64> = None;
        let mut converged = false;
        let mut ran = 0;
        let mut watch = StagnationWatch::new(rtol);
        let mut cached_jac: Option<S::Matrix> = None;
        for i in 0..opts.max_iter {
            ran = i + 1;
            let st = system.assemble(&x)?;
            if converged {
                state = Some(st);
                break;
            }
            let err = energy_error_pct(st.total_energy, last_energy);
            last_energy = Some(st.total_energy);
            let x_prev = x.clone();
            let refresh = i % every == 0 || cached_jac.is_none();
            if refresh {
                cached_jac = system.tangent(&x, &st)?;
            }
            match &cached_jac {
                // No Jacobian -> Picard substitution.
                None => x = system.solve_linear(&st)?,
                Some(jac) => {
                    let r = system.residual(&x, &st)?;
                    let dx = system.solve_system(jac, &neg(&r))?;
                    if refresh {
                        // Freshly-refreshed tangent: trusted, no staleness check needed.
                        x = axpy(&x, 1.0, &dx);
                    } else {
                        let r0 = norm(&r);
                        let trial = axpy(&x, 1.0, &dx);
                        let trial_state = system.assemble(&trial)?;
                        let r1 = norm(&system.residual(&trial, &trial_state)?);
                        if r1.is_finite() && r1 <= r0 {
                            x = trial;
                        } else {
                            // Stale: refresh right here and retake the step.
                            cached_jac = system.tangent(&x, &st)?;
                            match &cached_jac {
                                None => x = system.solve_linear(&st)?,
                                Some(fresh) => {
                                    let dx = system.solve_system(fresh, &neg(&r))?;
                                    x = axpy(&x, 1.0, &dx);
                                }
                            }
                        }
                    }
                }
            }
            let increment = increment_ratio(&x_prev, &x);
            converged = has_converged(err, &x_prev, &x, rtol) || watch.observe(err, increment);
            if system.is_linear() {
                converged = true;
            }
            system.on_iteration(i, &st, err, 1.0, increment);
            state = Some(st);
        }
        Ok(IterationResult {
            x,
            state,
            iterations: ran,
            converged,
        })
    }
}

/// The registry name `newton-linesearch` -- and it IS [`NewtonStrategy`], under its own name.
///
/// It was a second Newton loop until 2026-09-25 (the same backtracking search on the energy-change
/// stop, beside a full-step `newton`). The search is now part of `NewtonStrategy`, so this only
/// reports a different name; mirrors the Python subclass.
#[derive(Debug, Clone, Copy, Default)]
pub struct NewtonLineSearchStrategy {
    pub newton: NewtonStrategy,
}

impl<S: NonlinearSystem> IterationStrategy<S> for NewtonLineSearchStrategy {
    fn name(&self) -> &'static str {
        "newton-linesearch"
    }

    fn stop_rule(&self) -> StopRule {
        StopRule::Residual
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        self.newton.drive("newton-linesearch", system, opts)
    }
}

/// The registry name `newton-held-factor`: [`NewtonStrategy`] holding its tangent for 4
/// iterations (the modified Newton, user ruling 2026-09-26). Mirrors the Python
/// `HeldFactorNewtonStrategy`. The Python side additionally FACTORS a sparse tangent once (`splu`
/// or the system's `factorize`); this crate's opaque `Matrix` has no factor seam, so a held tangent
/// is re-solved through `solve_system` -- exactly what the Python does on the golden's plain-array
/// tangent, so the trajectories are the same.
#[derive(Debug, Clone, Copy)]
pub struct HeldFactorNewtonStrategy {
    pub newton: NewtonStrategy,
}

impl Default for HeldFactorNewtonStrategy {
    fn default() -> Self {
        Self {
            newton: NewtonStrategy {
                refactor_every: Some(4),
                ..NewtonStrategy::default()
            },
        }
    }
}

impl<S: NonlinearSystem> IterationStrategy<S> for HeldFactorNewtonStrategy {
    fn name(&self) -> &'static str {
        "newton-held-factor"
    }

    fn stop_rule(&self) -> StopRule {
        StopRule::Residual
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        self.newton.drive("newton-held-factor", system, opts)
    }
}

/// Newton whose LINEAR system is solved by a Krylov method — the path to large 3D / eddy-current
/// systems.
///
/// PARTIALLY PORTED, and the gap is named rather than papered over. The Python has two branches:
/// an OPAQUE tangent (FEA's reduced-increment bundle) routes through the equation's own
/// `solve_system`, and a scipy-operator tangent — or none at all, the Jacobian-free JFNK case —
/// goes to GMRES, with a Picard substitution on Krylov breakdown.
///
/// This port keeps the first branch and the breakdown fallback and does NOT implement GMRES. The
/// tangent here is opaque by construction (an associated type no strategy may inspect), so the
/// `solve_system` route is the correct one for every equation that supplies a tangent, and the
/// resulting iterate is an exact Newton step converging to the same root — which is precisely what
/// the Python's own comment says about that branch. What is missing is the JACOBIAN-FREE path,
/// which needs a Krylov backend; that lives in `numerics::linear`, not here, and wiring it in is a
/// follow-on rather than a rewrite of this file. Until then a system with no tangent takes the
/// Picard substitution — the Python's own breakdown behaviour, reached one step earlier.
///
/// `eps` is retained (unused today) so the JFNK differencing parameter does not have to be
/// re-derived when the backend arrives.
#[derive(Debug, Clone, Copy)]
pub struct NewtonKrylovStrategy {
    pub eps: f64,
}

impl Default for NewtonKrylovStrategy {
    fn default() -> Self {
        Self { eps: 1e-7 }
    }
}

impl<S: NonlinearSystem> IterationStrategy<S> for NewtonKrylovStrategy {
    fn name(&self) -> &'static str {
        "newton-krylov"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        let rtol = opts.rtol_energy_pct;
        let mut x = opts.start(system);
        let mut state: Option<StateOf<S>> = None;
        let mut last_energy: Option<f64> = None;
        let mut converged = false;
        let mut ran = 0;
        let mut watch = StagnationWatch::new(rtol);
        for i in 0..opts.max_iter {
            ran = i + 1;
            let st = system.assemble(&x)?;
            if converged {
                state = Some(st);
                break;
            }
            let err = energy_error_pct(st.total_energy, last_energy);
            last_energy = Some(st.total_energy);
            let x_prev = x.clone();
            let r = system.residual(&x, &st)?;
            let dx = match system.tangent(&x, &st)? {
                Some(jac) => system.solve_system(&jac, &neg(&r))?,
                // No operator to apply: Picard substitution, expressed as an increment.
                None => sub(&system.solve_linear(&st)?, &x),
            };
            x = axpy(&x, 1.0, &dx);
            let increment = increment_ratio(&x_prev, &x);
            converged = has_converged(err, &x_prev, &x, rtol) || watch.observe(err, increment);
            if system.is_linear() {
                converged = true;
            }
            system.on_iteration(i, &st, err, 1.0, increment);
            state = Some(st);
        }
        Ok(IterationResult {
            x,
            state,
            iterations: ran,
            converged,
        })
    }
}

/// Load / source stepping: solve a ramp `0 < lam_1 < ... < lam_n = 1` of the source.
///
/// An OUTER driver for strong-saturation / large-current operating points where a single solve of
/// the full source diverges. It scales the source (conductor current + magnet Br) by `lam` via
/// [`NonlinearSystem::scale_source`], solving each sub-problem seeded by the previous solution
/// (natural continuation). The last sub-solve (`lam = 1`) is the full problem, so the converged
/// field is the true solution. Degrades to a single full-source inner solve when the equation
/// cannot rescale its source.
///
/// The Python builds a rescaled COPY per step and a `_SeededSystem` wrapper around it; this scales
/// IN PLACE and passes the seed through [`SolveOptions::seed`]. Same ramp, same warm start, and
/// the system is restored to `lam = 1` before the final re-assembly — so a caller never gets a
/// scaled system back.
pub struct ContinuationStrategy<S: NonlinearSystem> {
    pub steps: usize,
    pub inner: Box<dyn IterationStrategy<S>>,
}

impl<S: NonlinearSystem + 'static> Default for ContinuationStrategy<S> {
    fn default() -> Self {
        Self {
            steps: 4,
            inner: Box::new(DampedPicardStrategy),
        }
    }
}

impl<S: NonlinearSystem + 'static> ContinuationStrategy<S> {
    #[must_use]
    pub fn new(steps: usize, inner: Box<dyn IterationStrategy<S>>) -> Self {
        Self {
            steps: steps.max(1),
            inner,
        }
    }
}

impl<S: NonlinearSystem> std::fmt::Debug for ContinuationStrategy<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuationStrategy")
            .field("steps", &self.steps)
            .field("inner", &self.inner.name())
            .finish()
    }
}

impl<S: NonlinearSystem> IterationStrategy<S> for ContinuationStrategy<S> {
    fn name(&self) -> &'static str {
        "continuation"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        let steps = self.steps.max(1);
        let mut result: Option<ResultOf<S>> = None;
        let mut seed: Option<Vec<f64>> = opts.seed.clone();
        for k in 0..steps {
            let lam = (k + 1) as f64 / steps as f64;
            if !(lam == 1.0 && steps == 1) {
                match system.scale_source(lam) {
                    Ok(()) => {}
                    Err(NonlinearError::SourceNotScalable) => {
                        // The equation cannot rescale its source: solve the full problem directly.
                        return self.inner.solve(system, opts);
                    }
                    Err(e) => return Err(e),
                }
            }
            let sub_opts = SolveOptions {
                seed,
                ..opts.clone()
            };
            let r = self.inner.solve(system, &sub_opts)?;
            seed = Some(r.x.clone());
            result = Some(r);
        }
        let result = result.expect("steps >= 1, so at least one sub-solve ran");
        // The last sub-state came from the scaled sub-system; restore the full source and
        // re-assemble at the converged iterate so the returned payload/energy are the true
        // (lam = 1) fields.
        let _ = system.scale_source(1.0);
        let final_state = system.assemble(&result.x)?;
        Ok(IterationResult {
            x: result.x,
            state: Some(final_state),
            iterations: result.iterations,
            converged: result.converged,
        })
    }
}
