//! Multi-secant and quasi-Newton strategies: Anderson acceleration and the Broyden / BFGS /
//! L-BFGS family. Split from `newton.rs`, which crossed the 800-line band.

use super::newton::{axpy, dot, neg, sub};
use super::{
    energy_error_pct, has_converged, increment_ratio, IterationResult, IterationStrategy,
    NonlinearError, NonlinearSystem, ResultOf, SolveOptions, StagnationWatch, StateOf,
};

/// Anderson acceleration of the Picard fixed-point map.
///
/// The Picard map is `g(x) = solve_linear(assemble(x))` and its fixed-point residual is
/// `f(x) = g(x) - x`. Plain/damped Picard mixes only the LAST iterate; Anderson mixes the last `m`
/// iterates by the least-squares combination of their residuals that minimises the mixed residual,
/// giving multi-secant (quasi-Newton-like) acceleration while using ONLY `assemble` +
/// `solve_linear` — so it drops onto BOTH MEC and FEA with no tangent. Converges to the SAME fixed
/// point damped Picard reaches, in fewer iterations on a contractive map.
///
/// Bookkeeping matches the Picard strategies (assemble-at-top, one-step-delayed break) so the
/// returned state is the converged iterate's and the reference-energy gate holds. `beta` is the
/// mixing (relaxation) factor; `m` the history depth.
///
/// DEVIATION FROM THE PYTHON, named because it can matter. The Python solves the inner
/// least-squares with `numpy.linalg.lstsq`, an SVD that returns the MINIMUM-NORM solution when
/// `dF` is rank-deficient. This uses a Householder QR, which returns a basic solution instead and
/// falls back to the plain Picard image `gx` when the triangular factor is numerically singular.
/// On a full-column-rank `dF` — every step where the history is genuinely informative — the two
/// agree to rounding. They can differ on a degenerate history, and that is a real difference, not
/// a rounding one.
#[derive(Debug, Clone, Copy)]
pub struct AndersonStrategy {
    pub m: usize,
    pub beta: f64,
}

impl Default for AndersonStrategy {
    fn default() -> Self {
        Self { m: 5, beta: 1.0 }
    }
}

impl<S: NonlinearSystem> IterationStrategy<S> for AndersonStrategy {
    fn name(&self) -> &'static str {
        "anderson"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        let rtol = opts.rtol_energy_pct;
        let mut x = opts.start(system);
        let mut state: Option<StateOf<S>> = None;
        let mut last_energy: Option<f64> = None;
        let mut converged = false;
        let mut ran = 0;
        let mut watch = StagnationWatch::new(rtol);
        let mut x_hist: Vec<Vec<f64>> = Vec::new(); // past g(x_k) images
        let mut f_hist: Vec<Vec<f64>> = Vec::new(); // past residuals f_k = g(x_k) - x_k
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
            let gx = system.solve_linear(&st)?; // one Picard image g(x_k)
            let f = sub(&gx, &x); // fixed-point residual
            x_hist.push(gx.clone());
            f_hist.push(f.clone());
            if f_hist.len() > self.m + 1 {
                x_hist.remove(0);
                f_hist.remove(0);
            }
            if system.is_linear() || f_hist.len() == 1 {
                // Nothing to mix yet (or nothing to gain): the plain Picard image.
                x = gx;
            } else {
                // Column-stacked first differences of the two histories.
                let k = f_hist.len() - 1;
                let d_f: Vec<Vec<f64>> = (0..k).map(|j| sub(&f_hist[j + 1], &f_hist[j])).collect();
                let d_x: Vec<Vec<f64>> = (0..k).map(|j| sub(&x_hist[j + 1], &x_hist[j])).collect();
                match lstsq(&d_f, &f) {
                    None => x = gx,
                    Some(gamma) => {
                        // x = gx - dX gamma, then the beta relaxation on the mixed residual.
                        let mut next = gx;
                        for (col, g) in d_x.iter().zip(&gamma) {
                            next = axpy(&next, -*g, col);
                        }
                        if (self.beta - 1.0).abs() > 0.0 {
                            let mut mixed_f = f;
                            for (col, g) in d_f.iter().zip(&gamma) {
                                mixed_f = axpy(&mixed_f, -*g, col);
                            }
                            next = axpy(&next, self.beta - 1.0, &mixed_f);
                        }
                        x = next;
                    }
                }
            }
            let increment = increment_ratio(&x_prev, &x);
            converged = has_converged(err, &x_prev, &x, rtol) || watch.observe(err, increment);
            if system.is_linear() {
                converged = true;
            }
            system.on_iteration(i, &st, err, self.beta, increment);
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

/// Least-squares `min ||A gamma - b||` by Householder QR; `None` when `R` is singular.
///
/// `a` is COLUMN-major (each entry is one column), which is how Anderson builds it. Sized by the
/// history depth, so `k` is a handful and a dense QR is the right shape.
fn lstsq(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let k = a.len();
    let n = b.len();
    if k == 0 || n < k {
        return None;
    }
    // Work on a row-major copy of A augmented with b, so the reflectors apply to both at once.
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|r| {
            let mut row: Vec<f64> = (0..k).map(|c| a[c][r]).collect();
            row.push(b[r]);
            row
        })
        .collect();
    for c in 0..k {
        let alpha = {
            let s: f64 = (c..n).map(|r| m[r][c] * m[r][c]).sum::<f64>().sqrt();
            if m[c][c] > 0.0 {
                -s
            } else {
                s
            }
        };
        if alpha.abs() < 1e-300 {
            return None;
        }
        let mut v: Vec<f64> = (0..n).map(|r| if r < c { 0.0 } else { m[r][c] }).collect();
        v[c] -= alpha;
        let vv = dot(&v, &v);
        if vv < 1e-300 {
            continue;
        }
        // Apply the reflector to every remaining column INCLUDING the augmented rhs at index `k`,
        // which is why this walks `c..=k` and not `c..k`.
        for col in c..=k {
            let scale: f64 = (c..n).map(|r| v[r] * m[r][col]).sum::<f64>() * 2.0 / vv;
            for (r, row) in m.iter_mut().enumerate().take(n).skip(c) {
                row[col] -= scale * v[r];
            }
        }
    }
    // Back-substitute the upper-triangular k x k block against the transformed rhs.
    let mut gamma = vec![0.0; k];
    for c in (0..k).rev() {
        let diag = m[c][c];
        if diag.abs() < 1e-12 {
            return None;
        }
        let s: f64 = ((c + 1)..k).map(|j| m[c][j] * gamma[j]).sum();
        gamma[c] = (m[c][k] - s) / diag;
    }
    Some(gamma)
}

/// Quasi-Newton (Broyden good) — an approximate inverse Jacobian from residual history.
///
/// Builds the Newton direction from a low-rank secant approximation of the inverse Jacobian,
/// updated each step from the `(dx, df)` pair — so it needs only `residual` and an initial linear
/// apply, positioning between Picard and Newton where the exact `tangent` is unavailable. The
/// initial inverse-Jacobian action is one solve of the secant `system_matrix` (available on every
/// equation); subsequent steps apply the accumulated rank-1 Sherman-Morrison updates with no
/// further factorization. Converges to the same root; same delayed-break energy bookkeeping.
#[derive(Debug, Clone, Copy)]
pub struct BroydenStrategy {
    /// Cap on the stored rank-1 update pairs.
    pub m: usize,
}

impl Default for BroydenStrategy {
    fn default() -> Self {
        Self { m: 10 }
    }
}

/// One stored rank-1 inverse-Jacobian update: `v <- v + (s . v) * u`.
type Rank1 = (Vec<f64>, Vec<f64>);

fn apply_broyden_inverse<S: NonlinearSystem>(
    system: &mut S,
    state: &StateOf<S>,
    r: &[f64],
    updates: &[Rank1],
) -> Result<Vec<f64>, NonlinearError> {
    let mut v = system.solve_secant(state, r)?;
    for (s_vec, u) in updates {
        let c = dot(s_vec, &v);
        v = axpy(&v, c, u);
    }
    Ok(v)
}

impl<S: NonlinearSystem> IterationStrategy<S> for BroydenStrategy {
    fn name(&self) -> &'static str {
        "quasi-newton"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        let rtol = opts.rtol_energy_pct;
        let mut x = opts.start(system);
        let mut state: Option<StateOf<S>> = None;
        let mut last_energy: Option<f64> = None;
        let mut converged = false;
        let mut ran = 0;
        let mut watch = StagnationWatch::new(rtol);
        let mut updates: Vec<Rank1> = Vec::new();
        let mut prev: Option<(Vec<f64>, Vec<f64>)> = None; // (r, dx) of the last step
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
            if let Some((prev_r, prev_dx)) = &prev {
                let df = sub(&r, prev_r);
                let h_df = apply_broyden_inverse(system, &st, &df, &updates)?;
                let denom = dot(prev_dx, &h_df);
                if denom.abs() > 1e-30 {
                    let u: Vec<f64> = prev_dx
                        .iter()
                        .zip(&h_df)
                        .map(|(a, b)| (a - b) / denom)
                        .collect();
                    updates.push((prev_dx.clone(), u));
                    if updates.len() > self.m {
                        updates.remove(0);
                    }
                }
            }
            let dx = neg(&apply_broyden_inverse(system, &st, &r, &updates)?);
            x = axpy(&x, 1.0, &dx);
            prev = Some((r, dx));
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

/// One BFGS secant pair `(s = dx, y = dr, rho = 1/(y . s))`.
type SecantPair = (Vec<f64>, Vec<f64>, f64);

/// The L-BFGS two-loop recursion: backward pass, `H0` apply, forward pass.
///
/// Mathematically identical to the dense `H <- (I - rho s y^T) H (I - rho y s^T) + rho s s^T`
/// update, applied matrix-free so no dense `H` is ever formed. Shared by [`BfgsStrategy`] and
/// [`LbfgsStrategy`], which differ ONLY in how deep the retained history is.
fn two_loop<S: NonlinearSystem>(
    system: &mut S,
    state: &StateOf<S>,
    r: &[f64],
    pairs: &[SecantPair],
) -> Result<Vec<f64>, NonlinearError> {
    let mut q = r.to_vec();
    let mut alphas = Vec::with_capacity(pairs.len());
    for (s_vec, y_vec, rho) in pairs.iter().rev() {
        let a = rho * dot(s_vec, &q);
        q = axpy(&q, -a, y_vec);
        alphas.push(a);
    }
    let mut z = system.solve_secant(state, &q)?; // H0 z = q (one secant solve)
                                                 // `alphas` was pushed in reverse order, so walking it backwards pairs it with `pairs` forwards
                                                 // -- the Python spells this `zip(pairs, reversed(alphas))`.
    for ((s_vec, y_vec, rho), a) in pairs.iter().zip(alphas.iter().rev()) {
        let b = rho * dot(y_vec, &z);
        z = axpy(&z, a - b, s_vec);
    }
    Ok(z)
}

/// The body shared by full BFGS and limited-memory BFGS: only `m` differs.
fn run_bfgs<S: NonlinearSystem>(
    system: &mut S,
    opts: &SolveOptions,
    m: usize,
) -> Result<ResultOf<S>, NonlinearError> {
    let rtol = opts.rtol_energy_pct;
    let mut x = opts.start(system);
    let mut state: Option<StateOf<S>> = None;
    let mut last_energy: Option<f64> = None;
    let mut converged = false;
    let mut ran = 0;
    let mut watch = StagnationWatch::new(rtol);
    let mut pairs: Vec<SecantPair> = Vec::new();
    let mut prev: Option<(Vec<f64>, Vec<f64>)> = None;
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
        if let Some((prev_r, prev_dx)) = &prev {
            let y = sub(&r, prev_r);
            let ys = dot(&y, prev_dx);
            // Curvature condition; skip a non-informative (near-zero) pair.
            if ys.abs() > 1e-30 {
                pairs.push((prev_dx.clone(), y, 1.0 / ys));
                if pairs.len() > m {
                    pairs.remove(0);
                }
            }
        }
        let dx = neg(&two_loop(system, &st, &r, &pairs)?);
        x = axpy(&x, 1.0, &dx);
        prev = Some((r, dx));
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

/// Quasi-Newton (full BFGS) — a SYMMETRIC low-rank inverse Jacobian from secant pairs.
///
/// Broyden's update is the general (unsymmetric) secant; BFGS is its symmetric-positive-definite
/// sibling, the right choice for the SPD-ish magnetostatic root `r(x) = S(x) x - b` whose true
/// tangent is symmetric. It maintains `H ~ J^-1` by the classic rank-2 BFGS inverse update from
/// each `(s = dx, y = r_{k+1} - r_k)` pair, applied matrix-free through the stored history on top
/// of an initial `H0` action = one secant solve (the same seed Broyden uses). Needs only
/// `residual` + that one linear apply — no exact `tangent`. `m` caps the stored pairs (full BFGS =
/// the whole history).
#[derive(Debug, Clone, Copy)]
pub struct BfgsStrategy {
    pub m: usize,
}

impl Default for BfgsStrategy {
    fn default() -> Self {
        Self { m: 20 }
    }
}

impl<S: NonlinearSystem> IterationStrategy<S> for BfgsStrategy {
    fn name(&self) -> &'static str {
        "bfgs"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        run_bfgs(system, opts, self.m)
    }
}

/// Quasi-Newton (limited-memory BFGS) — the two-loop recursion on a bounded secant history.
///
/// Identical inverse-Hessian model to [`BfgsStrategy`] but capping the stored history to the last
/// `m` `(s, y)` pairs, so the per-step cost and memory stay `O(m n)` regardless of iteration count
/// — the form used for large systems where keeping the full history is prohibitive. Converges to
/// the same root as full BFGS / Newton on the SPD-ish magnetostatic problem.
#[derive(Debug, Clone, Copy)]
pub struct LbfgsStrategy {
    /// Limited-memory depth (the last `m` secant pairs).
    pub m: usize,
}

impl Default for LbfgsStrategy {
    fn default() -> Self {
        Self { m: 8 }
    }
}

impl<S: NonlinearSystem> IterationStrategy<S> for LbfgsStrategy {
    fn name(&self) -> &'static str {
        "l-bfgs"
    }

    fn solve(&self, system: &mut S, opts: &SolveOptions) -> Result<ResultOf<S>, NonlinearError> {
        run_bfgs(system, opts, self.m)
    }
}
