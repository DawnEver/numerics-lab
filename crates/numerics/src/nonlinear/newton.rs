//! Newton-family, quasi-Newton and continuation iteration strategies.

use super::{
    energy_error_pct, has_converged, increment_ratio, norm, DampedPicardStrategy, IterationResult,
    IterationStrategy, NonlinearError, NonlinearSystem, ResultOf, SolveOptions, StagnationWatch,
    StateOf,
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

/// Newton-Raphson: `J(x_k) dx = -r(x_k); x_{k+1} = x_k + dx`.
///
/// Uses the equation's `tangent` (secant + differential `dnu` term); at `x = 0` the field is zero
/// so `dnu = 0` and the first Newton step equals the first Picard iterate, then the true Jacobian
/// gives quadratic convergence near the solution (Picard is linear and needs damping on saturating
/// B-H). Falls back to a Picard step whenever `tangent` returns `None` (a combo with no Jacobian),
/// so wiring Newton never breaks a solver that cannot supply one. Same assemble-at-top /
/// one-step-delayed-break / energy-change convergence bookkeeping as the Picard strategies.
#[derive(Debug, Clone, Copy, Default)]
pub struct NewtonStrategy;

impl<S: NonlinearSystem> IterationStrategy<S> for NewtonStrategy {
    fn name(&self) -> &'static str {
        "newton"
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
            match system.tangent(&x, &st)? {
                // No Jacobian available -> Picard substitution.
                None => x = system.solve_linear(&st)?,
                Some(jac) => {
                    let r = system.residual(&x, &st)?;
                    let dx = system.solve_system(&jac, &neg(&r))?;
                    x = axpy(&x, 1.0, &dx);
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
/// which lives in `gauss::linear`, not here.
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

/// Damped Newton with a backtracking line search on `||residual||`.
///
/// `newton` takes the full step `x += dx`; deep saturation can make that overshoot and diverge.
/// This scales the Newton step by `lam` from an Armijo backtracking line search that shrinks `lam`
/// until `||r(x + lam dx)||` decreases — the robust globaliser. Needs `residual` plus `tangent`
/// (falls back to a Picard step when `tangent` returns `None`). Converges to the same root as
/// Newton (at `lam = 1` near the solution it IS Newton).
#[derive(Debug, Clone, Copy)]
pub struct NewtonLineSearchStrategy {
    pub max_backtracks: usize,
    pub shrink: f64,
}

impl Default for NewtonLineSearchStrategy {
    fn default() -> Self {
        Self {
            max_backtracks: 12,
            shrink: 0.5,
        }
    }
}

impl<S: NonlinearSystem> IterationStrategy<S> for NewtonLineSearchStrategy {
    fn name(&self) -> &'static str {
        "newton-linesearch"
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
            match system.tangent(&x, &st)? {
                None => x = system.solve_linear(&st)?,
                Some(jac) => {
                    let r = system.residual(&x, &st)?;
                    let dx = system.solve_system(&jac, &neg(&r))?;
                    // Minimise the FREE-DOF residual norm, NOT the raw ||r||: FEA's full residual
                    // carries a large constant Dirichlet-row term the increment cannot reduce,
                    // which would make every backtracking step look like an increase and stall the
                    // line search.
                    let r0 = system.residual_norm(&x, &st)?;
                    let mut lam = 1.0_f64;
                    for _ in 0..self.max_backtracks {
                        // r(x + lam dx) is assembled implicitly: re-assemble at the trial point
                        // and read its residual (the equation owns assembly). Accept the first lam
                        // that reduces it.
                        let trial = axpy(&x, lam, &dx);
                        let trial_state = system.assemble(&trial)?;
                        if system.residual_norm(&trial, &trial_state)? < r0 {
                            break;
                        }
                        lam *= self.shrink;
                    }
                    x = axpy(&x, lam, &dx);
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
/// which needs a Krylov backend; that lives in `gauss::linear`, not here, and wiring it in is a
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
