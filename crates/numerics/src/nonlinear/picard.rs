//! Picard-family iteration strategies (damped / adaptive / nu-relaxation / plain).

use super::{
    energy_error_pct, energy_flux_convergence, has_converged, increment_ratio, IterationResult,
    IterationStrategy, NonlinearError, NonlinearSystem, ResultOf, SolveOptions, StagnationWatch,
    StateOf, ALPHA_CALM_STEPS, ALPHA_GROW, ALPHA_MIN,
};

/// Successive substitution with adaptive under-relaxation — the current MEC/FEA production loop.
///
/// Extracted VERBATIM from `mec/_sparse.py`'s inline loop: assemble at the top, break on a
/// one-step-delayed convergence flag (so the returned state is the converged iterate's), halve
/// `alpha` (floored at [`ALPHA_MIN`](super::ALPHA_MIN)) whenever the energy error grows, converge
/// when the energy change drops below the tolerance, and damp the update
/// `x += alpha * (solve - x)`. A linear system converges after one solve.
///
/// Also ratchets `alpha` down (x0.6, floored) on an energy-delta SIGN FLIP — the same class of
/// damping the FEA frozen-damped loop applies. The `err > prev_err` rule only fires on a
/// MONOTONICALLY growing relative error; a period-2 limit cycle (energy alternating between two
/// values every step, e.g. a sharp BH knee overshooting each iteration) has a roughly CONSTANT
/// relative error, so that rule never trips and alpha stays at 1.0 forever (MEC's Picard loop
/// never converged on the FEMM-matching PCHIP BH spline). The sign-flip trigger catches exactly
/// that oscillation.
///
/// Convergence is [`has_converged`](super::has_converged) — energy AND iterate, UNCONDITIONALLY.
/// That test used to live here, behind "did we ratchet earlier", on the reasoning that only a
/// damped iterate can sit on a flat-but-wrong energy plateau. Both halves of that were wrong: the
/// plateau is a property of the ENERGY CRITERION (so the other eleven strategies needed it too),
/// and gating it on an earlier, unrelated damping event meant the strategy that OWNED the
/// protection did not apply it on an undamped run — which its own text conceded was "today's
/// common case".
///
/// ALPHA RECOVERS. The ratchet was one-way, so a single early overshoot damped the entire smooth
/// tail: measured on `ipm_uv_cummins mec:quad`, alpha sat at `1.0 * 0.5 * 0.6^3 = 0.108` while the
/// energy error read 0.00%, and each rotor position spent 44-61 iterations at ~3 s crawling to the
/// increment bar. After [`ALPHA_CALM_STEPS`](super::ALPHA_CALM_STEPS) undisturbed steps alpha
/// grows back by [`ALPHA_GROW`](super::ALPHA_GROW), capped at `alpha_cap` — the step size that
/// last misbehaved, never retried. This changes the RATE, not the criterion: the fixed point and
/// the test are unchanged.
#[derive(Debug, Clone, Copy, Default)]
pub struct DampedPicardStrategy;

impl<S: NonlinearSystem> IterationStrategy<S> for DampedPicardStrategy {
    fn name(&self) -> &'static str {
        "picard-damped"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        let rtol = opts.rtol_energy_pct;
        let mut x = opts.start(system);
        let mut state: Option<StateOf<S>> = None;
        let mut last_energy: Option<f64> = None;
        let mut converged = false;
        let (mut alpha, mut prev_err) = (1.0_f64, f64::INFINITY);
        let mut prev_delta_sign = 0_i32;
        // Never recover back to a step size that already misbehaved.
        let mut alpha_cap = 1.0_f64;
        // Consecutive steps with neither an error growth nor a sign flip.
        let mut calm = 0_usize;
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
            // BOTH ratchets need the oscillation to be BIGGER THAN THE CONVERGENCE TOLERANCE.
            // Below it the energy is at its own floating-point noise floor, where `delta`'s sign
            // flips at random — and a sign flip is what the second ratchet reads as a limit cycle.
            // Measured on `ipm_uv_cummins mec:quad`: the energy error reached 0.00% early, the
            // sign then alternated on noise for the rest of the solve, alpha was driven to 0.108
            // and pinned there, and each rotor position spent 44-61 iterations crawling to the
            // increment bar at a tenth of the step size. A limit cycle is BY DEFINITION an
            // oscillation that does not decay; one already inside the tolerance is not one.
            let meaningful = err >= rtol;
            let alpha_before = alpha;
            let mut disturbed = false;
            if meaningful && prev_err.is_finite() && err > prev_err {
                alpha = ALPHA_MIN.max(alpha * 0.5);
                disturbed = true;
            }
            if let Some(last) = last_energy {
                let delta = st.total_energy - last;
                let sign = i32::from(delta > 0.0) - i32::from(delta < 0.0);
                if meaningful && prev_delta_sign != 0 && sign != 0 && sign != prev_delta_sign {
                    alpha = ALPHA_MIN.max(alpha * 0.6);
                    disturbed = true;
                }
                prev_delta_sign = sign;
            }
            if disturbed {
                // ONLY when the shrink actually MOVED alpha. At the floor `max(ALPHA_MIN, alpha*k)`
                // is a no-op, and recording the cap anyway sets `alpha_cap = ALPHA_MIN` — after
                // which the recovery gate `alpha < alpha_cap` reads `0.1 < 0.1` and is false for
                // the rest of the solve. That is the one-way ratchet this recovery leg was ADDED
                // to remove, surviving one level down: measured on `dcm_fdm_rect`, 557 of 605
                // iterations pinned at alpha 0.100 with the energy error already at 0.00%, every
                // solve running to its 100-iteration cap because a tenth-sized step cannot reach
                // the increment bar. A step size that never actually applied did not misbehave, so
                // it forbids nothing.
                if alpha < alpha_before {
                    alpha_cap = alpha_before; // the step size that MISBEHAVED, never retried
                }
                calm = 0;
            } else {
                calm += 1;
                if calm >= ALPHA_CALM_STEPS {
                    if alpha < alpha_cap {
                        alpha = alpha_cap.min(alpha * ALPHA_GROW);
                    } else {
                        // THE CAP ITSELF RELEASES, once alpha has climbed back to it AND the solve
                        // has stayed calm for another full window. Without this the cap is a
                        // second one-way ratchet behind the first: a step size that misbehaved FAR
                        // FROM THE SOLUTION stays forbidden for the whole solve, even after the
                        // iterate has settled and the energy sits at its noise floor. The physics
                        // near convergence is not the physics that misbehaved at iteration 3.
                        //
                        // MEASURED (`dcm_fdm_rect`) after releasing only the alpha FLOOR: 557
                        // iterations pinned at 0.100 became 447 at 0.108 — recovery fired and the
                        // sweep still took 658 iterations, because 0.108 was now the ceiling.
                        // `mec-picard-runs-to-its-cap-while-already-converged` (2026-08-26)
                        // predicted it: "the `meaningful` guard prevents NEW reductions; it does
                        // not release an old cap."
                        //
                        // SAFE BECAUSE THE GLOBALISER IS UNCHANGED: any real energy growth or sign
                        // flip above the convergence tolerance re-arms both ratchets on the very
                        // next step, dropping alpha AND resetting the cap to whatever misbehaved.
                        // A limit cycle is still damped; what is no longer damped is the smooth
                        // tail after one early transient.
                        alpha_cap = 1.0_f64.min(alpha_cap * ALPHA_GROW);
                        alpha = alpha_cap.min(alpha * ALPHA_GROW);
                    }
                    calm = 0;
                }
            }
            prev_err = err;
            let x_solve = system.solve_linear(&st)?;
            let increment = increment_ratio(&x, &x_solve);
            converged = has_converged(err, &x, &x_solve, rtol) || watch.observe(err, increment);
            last_energy = Some(st.total_energy);
            x = x
                .iter()
                .zip(&x_solve)
                .map(|(a, b)| a + alpha * (b - a))
                .collect();
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
}

/// Self-tuning damped Picard — alpha both shrinks on overshoot AND recovers on progress.
///
/// The default [`DampedPicardStrategy`] only ever RATCHETS alpha DOWN on the energy-error growth
/// rule: safe, but once a hard early step forces alpha small it stays small, over-damping the easy
/// tail. This variant self-tunes in BOTH directions — it still halves alpha when the energy error
/// grows (the robust globaliser behaviour), but it also GROWS alpha back toward 1 (by `grow`,
/// capped at 1) on a step that reduced the error, so a case that starts stiff and relaxes recovers
/// full Picard steps and finishes in fewer iterations. It converges to the SAME fixed point damped
/// Picard reaches (same map, only the relaxation schedule differs); on a smoothly contractive
/// network the two are step-for-step identical (alpha never has to drop, so it never grows).
///
/// Uses ONLY `assemble` + `solve_linear` (no tangent), so it drops onto BOTH MEC and FEA. Same
/// delayed-break energy bookkeeping.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveDampedPicardStrategy {
    /// Multiplicative alpha recovery on a step that reduced the error.
    pub grow: f64,
    /// Multiplicative alpha cut on a step that grew the error.
    pub shrink: f64,
}

impl Default for AdaptiveDampedPicardStrategy {
    fn default() -> Self {
        Self {
            grow: ALPHA_GROW,
            shrink: 0.5,
        }
    }
}

impl<S: NonlinearSystem> IterationStrategy<S> for AdaptiveDampedPicardStrategy {
    fn name(&self) -> &'static str {
        "picard-adaptive"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        let rtol = opts.rtol_energy_pct;
        let mut x = opts.start(system);
        let mut state: Option<StateOf<S>> = None;
        let mut last_energy: Option<f64> = None;
        let mut converged = false;
        let (mut alpha, mut prev_err) = (1.0_f64, f64::INFINITY);
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
            if prev_err.is_finite() {
                if err > prev_err {
                    alpha = ALPHA_MIN.max(alpha * self.shrink); // overshoot -> damp harder
                } else if err < prev_err {
                    alpha = 1.0_f64.min(alpha * self.grow); // progress -> recover the full step
                }
            }
            prev_err = err;
            let x_solve = system.solve_linear(&st)?;
            let increment = increment_ratio(&x, &x_solve);
            converged = has_converged(err, &x, &x_solve, rtol) || watch.observe(err, increment);
            last_energy = Some(st.total_energy);
            x = x
                .iter()
                .zip(&x_solve)
                .map(|(a, b)| a + alpha * (b - a))
                .collect();
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
}

/// The FEA reluctivity-under-relaxed Picard loop.
///
/// The FEA frozen loop damped the RELUCTIVITY, not the solution vector — under-relaxing `nu`
/// inside the assembly kernel, per element, before the stiffness/source were formed
/// (`nu_used = (1-alpha) nu_prev + alpha nu(B)`). [`DampedPicardStrategy`] damps the SOLUTION
/// instead and cannot reproduce that iteration map (measured ~2.6e-5 relative on the field,
/// ~5.5e-5 on energy — close, but a DIFFERENT fixed-point trajectory), which is why the frozen
/// loop stayed a separate, inline, kernel-coupled code path for as long as it did.
///
/// **The move this strategy makes possible.** Assembly and the reluctivity's per-element
/// under-relaxation turn out to be exactly the SAME linear operation reordered. Every raw
/// stiffness entry and every PM-remanence source entry is `reluctivity_e * <fixed geometric
/// factor>` — LINEAR in the per-element reluctivity scalar — and assembly is a LINEAR map. So
/// mixing the per-element reluctivity BEFORE assembly (what the deleted kernel did) and mixing the
/// assembled RAW arrays AFTER assembly (what this strategy does) are the same operator:
///
/// ```text
/// nu_k = (1-alpha) nu_{k-1} + alpha nu_target_k
///   ==(linear assembly)==>
/// S_k = (1-alpha) S_{k-1} + alpha S(nu_target_k),  T_k = (1-alpha) T_{k-1} + alpha T(nu_target_k)
/// ```
///
/// where `S_{k-1}`/`T_{k-1}` are literally last iteration's already-mixed raw arrays — no
/// per-element state crosses the kernel boundary, only the previous raw pair. A conductor-current
/// source term (nu-independent, identical every iteration) mixes back to itself trivially since
/// the weights always sum to 1.
///
/// **This identity REQUIRES alpha to be a single GLOBAL scalar, uniform across every element**,
/// exactly as the deleted loop always used it: `(1-alpha) S(w) != S((1-alpha) w)` in general, but
/// IS an identity when the SAME scalar multiplies the whole raw array. A future optimisation that
/// let alpha vary PER ELEMENT would silently invalidate this equivalence with nothing else going
/// red — there is deliberately no runtime assertion (alpha is a private `f64` local, never a
/// per-element array, so nothing today CAN pass a non-uniform alpha through this seam); this
/// paragraph is the guard until a per-element-alpha strategy is actually proposed.
///
/// Not bit-for-bit with the deleted loop (floating-point reassociation), but the SAME operator in
/// exact arithmetic.
///
/// Reached through [`NonlinearSystem::raw_state`] / [`NonlinearSystem::fold_raw`], which default
/// to `None` — the mixing has to happen BEFORE the boundary fold to match the deleted loop's order
/// of operations exactly, and most equations have no such seam.
#[derive(Debug, Clone, Copy, Default)]
pub struct NuRelaxationDampedPicardStrategy;

/// Frozen floor for the reluctivity under-relaxation factor. Not [`ALPHA_MIN`]: the deleted FEA
/// loop used 0.05, and reproducing it verbatim is the whole point of this strategy.
const NU_ALPHA_MIN: f64 = 0.05;

impl<S: NonlinearSystem> IterationStrategy<S> for NuRelaxationDampedPicardStrategy {
    fn name(&self) -> &'static str {
        "nu-damped"
    }

    /// WHAT THIS STRATEGY NEEDS OF A SYSTEM, as data. The requirement used to be enforced only
    /// inside the loop and stated in three docstrings, and nothing could ASK it — so the
    /// iteration-method matrix generated a `nu-damped` cell for the MEC system, which cannot
    /// supply either half, and read the designed refusal as a red (heavy shard 1, 2026-08-29).
    fn requires(&self) -> &'static [&'static str] {
        &["raw_state", "fold_raw"]
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        if !system.supports_raw_mixing() {
            return Err(NonlinearError::UnsupportedSystem {
                strategy: "nu-damped",
                need: "raw_state/fold_raw".to_owned(),
            });
        }
        let rtol = opts.rtol_energy_pct;
        let mut x = opts.start(system);
        let mut alpha = 1.0_f64;
        let mut prev_delta_sign = 0_i32;
        let mut last_energy: Option<f64> = None;
        let mut last_flux: Option<f64> = None;
        // Last iteration's MIXED raw arrays; `None` = no prior iterate yet.
        let mut prev_mixed: Option<(Vec<f64>, Vec<f64>)> = None;
        let mut state: Option<StateOf<S>> = None;
        // One-step-delayed break: the pass AFTER convergence only refreshes state.
        let mut is_convergence = false;
        let mut reached_tol = false;
        let mut ran = 0;
        for i in 0..opts.max_iter {
            let raw = system
                .raw_state(&x)?
                .ok_or_else(|| NonlinearError::UnsupportedSystem {
                    strategy: "nu-damped",
                    need: "raw_state/fold_raw".to_owned(),
                })?;
            let (mixed_vals, mixed_source) = match &prev_mixed {
                // First pass: nu_prev < 0 -> reluctivity = nu_target, i.e. no mixing at all.
                None => (raw.vals.clone(), raw.source.clone()),
                Some((pv, pt)) => (mix(pv, &raw.vals, alpha), mix(pt, &raw.source, alpha)),
            };
            let mixed = super::RawState {
                rows: raw.rows,
                cols: raw.cols,
                vals: mixed_vals.clone(),
                source: mixed_source.clone(),
                total_energy: raw.total_energy,
            };
            let st = system
                .fold_raw(&mixed)?
                .ok_or_else(|| NonlinearError::UnsupportedSystem {
                    strategy: "nu-damped",
                    need: "raw_state/fold_raw".to_owned(),
                })?;
            if is_convergence {
                // Refresh-only pass: state now reflects the converged iterate; stop without
                // solving.
                state = Some(st);
                break;
            }
            let total_flux = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let (err, _flux_err, is_converged) = energy_flux_convergence(
                raw.total_energy,
                total_flux,
                last_energy,
                last_flux,
                rtol,
                rtol,
            );
            if is_converged {
                is_convergence = true;
                reached_tol = true;
            }
            // Frozen alpha adaptation: shrink (x0.6, floored at NU_ALPHA_MIN) on an energy-delta
            // SIGN FLIP — verbatim from the deleted loop, never raised back up. Note the absent
            // `meaningful` guard: this strategy reproduces the frozen loop, which had none.
            if let Some(last) = last_energy {
                let delta = raw.total_energy - last;
                let sign = i32::from(delta > 0.0) - i32::from(delta < 0.0);
                if prev_delta_sign != 0 && sign != 0 && sign != prev_delta_sign {
                    alpha = (alpha * 0.6).max(NU_ALPHA_MIN);
                }
                prev_delta_sign = sign;
            }
            last_energy = Some(raw.total_energy);
            last_flux = Some(total_flux);
            prev_mixed = Some((mixed_vals, mixed_source));
            let x_prev = x;
            x = system.solve_linear(&st)?;
            // This strategy judges on the energy/flux PAIR, not on `has_converged`, so it has no
            // increment of its own; it is measured here purely so the observer sees the same two
            // numbers everywhere and a capped run is readable without knowing which strategy ran.
            let increment = increment_ratio(&x_prev, &x);
            ran = i + 1;
            if system.is_linear() {
                is_convergence = true;
            }
            system.on_iteration(i, &st, err, alpha, increment);
            state = Some(st);
        }
        let converged = reached_tol || system.is_linear();
        Ok(IterationResult {
            x,
            state,
            iterations: ran,
            converged,
        })
    }
}

/// `(1-alpha) * prev + alpha * target`, elementwise.
fn mix(prev: &[f64], target: &[f64], alpha: f64) -> Vec<f64> {
    prev.iter()
        .zip(target)
        .map(|(p, t)| (1.0 - alpha) * p + alpha * t)
        .collect()
}

/// Plain successive substitution — NO damping (`x_{k+1} = solve`).
///
/// The naive baseline that historically diverged/oscillated on strongly-saturating B-H (which is
/// exactly WHY the damped variant exists). First principle: KEEP it, so its failure mode is a
/// characterized, labelled datum (a convergence-matrix `xfail`), not a silent exclusion. Same
/// assemble-at-top / delayed-break convergence bookkeeping as the damped variant, minus the alpha.
#[derive(Debug, Clone, Copy, Default)]
pub struct PicardStrategy;

impl<S: NonlinearSystem> IterationStrategy<S> for PicardStrategy {
    fn name(&self) -> &'static str {
        "picard"
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
            let x_solve = system.solve_linear(&st)?;
            let increment = increment_ratio(&x, &x_solve);
            converged = has_converged(err, &x, &x_solve, rtol) || watch.observe(err, increment);
            last_energy = Some(st.total_energy);
            x = x_solve; // undamped: take the full substitution
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
