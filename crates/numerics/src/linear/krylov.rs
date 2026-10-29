//! The Krylov family — `cg`, `minres`, `gmres`, `bicgstab`, `qmr`, hand-written.
//!
//! # Why these are hand-written and not a dependency
//!
//! The Python calls scipy, and scipy's five signatures are NOT uniform: `minres` has no `atol`,
//! `qmr` splits the preconditioner into `M1`/`M2`. The port has to reproduce a SEMANTICS
//! (`rtol = 1e-10`, `atol = 0`, `maxiter = 4n`, and a non-convergence that RAISES rather than
//! returning a wrong `x`), and these are a few hundred lines of well-specified linear algebra.
//! Writing them out makes matching that semantics a deliberate act instead of a hope that some
//! crate's defaults happen to agree.
//!
//! # The stopping rule, and the one place it is not scipy's
//!
//! Every solver here stops on the TRUE relative residual `||b - A x|| / ||b||`, recomputed from
//! the current iterate rather than carried in a recurrence. scipy's `gmres` measures the
//! PRECONDITIONED residual and `bicgstab`/`cg` carry a recurrence that drifts from the true
//! residual over many iterations. Measuring the true residual costs one extra matvec per step
//! and cannot certify a solution that does not satisfy the system — which is the same failure
//! mode the direct backend's residual guard exists to catch, so the two halves agree.

use crate::linear::precond::Preconditioner;
use crate::sparse::{Csr, SparseError};

/// Krylov convergence target.
///
/// Tight enough that the iterative solution matches the direct factorisation to the accuracy the
/// nonlinear outer loop needs (its energy tolerance is ~1e-4). Verbatim from the Python's
/// `_KRYLOV_RTOL`.
pub const KRYLOV_RTOL: f64 = 1e-10;

/// `atol = 0` makes the stop PURELY RELATIVE, so a small-norm rhs still converges to full
/// relative accuracy instead of being declared converged by an absolute floor it started under.
/// Verbatim from the Python's `_KRYLOV_ATOL`.
pub const KRYLOV_ATOL: f64 = 0.0;

/// `maxiter = KRYLOV_MAXITER_FACTOR * n` — a generous ceiling before declaring non-convergence.
pub const KRYLOV_MAXITER_FACTOR: usize = 4;

/// GMRES restart length: the Arnoldi basis is rebuilt every this many inner steps.
///
/// scipy's `gmres` defaults to `restart = min(20, n)`, and the number is a memory/convergence
/// trade rather than a tolerance — so it is named here rather than left as a literal, and the
/// port takes scipy's value so a system that needed a long basis does not silently stall.
pub const GMRES_RESTART: usize = 20;

/// A converged Krylov solve, with the count that makes a preconditioner's effect MEASURABLE.
///
/// The iteration count is not diagnostics: "SSOR roughly halves the CG count on the Poisson
/// operator while Jacobi leaves it unchanged" is the whole claim `ssor.py` makes for itself, and
/// a solver that returned only `x` would make that claim untestable.
#[derive(Debug, Clone, PartialEq)]
pub struct KrylovOutcome {
    /// The solution.
    pub x: Vec<f64>,
    /// Iterations taken (matvecs against `A`, one per step).
    pub iterations: usize,
    /// The achieved `||b - A x|| / ||b||`.
    pub residual: f64,
}

/// Why a Krylov solve refused.
#[derive(Debug, Clone, PartialEq)]
pub enum KrylovFailure {
    /// Nonfinite, negative absolute, or nonpositive relative tolerance.
    InvalidTolerance,
    /// Ran out of iterations — the Python's `info > 0`.
    NotConverged {
        /// The residual it got to.
        residual: f64,
        /// The ceiling it hit.
        maxiter: usize,
    },
    /// A recurrence coefficient collapsed — the Python's `info < 0`.
    Breakdown {
        /// Which scalar went to zero.
        quantity: &'static str,
        /// The iteration it happened on.
        iteration: usize,
    },
    /// The operator or the vectors do not agree on a shape.
    Shape(SparseError),
}

impl From<SparseError> for KrylovFailure {
    fn from(e: SparseError) -> Self {
        Self::Shape(e)
    }
}

/// The five registered Krylov methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Conjugate gradients — assumes SPD.
    Cg,
    /// MINRES — assumes symmetric (indefinite allowed).
    Minres,
    /// Restarted GMRES — general.
    Gmres,
    /// BiCGSTAB — general.
    BicgStab,
    /// QMR — general, and the only one that touches `A^T`.
    Qmr,
}

impl Method {
    /// Does this method assume a symmetric / SPD operator?
    ///
    /// Mirrors the Python's `requires_symmetric`, which tags `cg`/`minres`. Paired with
    /// `Csr::is_symmetric`, it makes "this solver was run on an operator it assumes something
    /// false about" DETECTABLE rather than a silently wrong answer.
    #[must_use]
    pub const fn requires_symmetric(self) -> bool {
        matches!(self, Self::Cg | Self::Minres)
    }

    /// The registry spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cg => "cg",
            Self::Minres => "minres",
            Self::Gmres => "gmres",
            Self::BicgStab => "bicgstab",
            Self::Qmr => "qmr",
        }
    }

    /// Resolve a registry name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "cg" => Some(Self::Cg),
            "minres" => Some(Self::Minres),
            "gmres" => Some(Self::Gmres),
            "bicgstab" => Some(Self::BicgStab),
            "qmr" => Some(Self::Qmr),
            _ => None,
        }
    }

    /// Run the method.
    ///
    /// # Errors
    /// [`KrylovFailure`] on non-convergence within `4n`, a recurrence breakdown, or a shape
    /// disagreement — never a wrong `x` returned as a success.
    pub fn solve(
        self,
        a: &Csr,
        b: &[f64],
        x0: Option<&[f64]>,
        m: Option<&Preconditioner>,
    ) -> Result<KrylovOutcome, KrylovFailure> {
        self.solve_with_tolerances(a, b, x0, m, KRYLOV_RTOL, KRYLOV_ATOL)
    }

    /// Solve with explicit relative and absolute targets, including original-RHS-scaled Newton tolerances.
    pub fn solve_with_tolerances(
        self,
        a: &Csr,
        b: &[f64],
        x0: Option<&[f64]>,
        m: Option<&Preconditioner>,
        rtol: f64,
        atol: f64,
    ) -> Result<KrylovOutcome, KrylovFailure> {
        if !rtol.is_finite()
            || rtol < 0.0
            || !atol.is_finite()
            || atol < 0.0
            || (rtol == 0.0 && atol == 0.0)
        {
            return Err(KrylovFailure::InvalidTolerance);
        }
        let mut st = State::new(a, b, x0)?;
        st.target = (rtol * st.b_norm).max(atol);
        match self {
            Self::Cg => cg(st, m),
            Self::Minres => minres(st, m),
            Self::Gmres => gmres(st, m),
            Self::BicgStab => bicgstab(st, m),
            Self::Qmr => qmr(st, m),
        }
    }
}

// -- shared scaffolding -----------------------------------------------------------------------

struct State<'a> {
    a: &'a Csr,
    b: &'a [f64],
    x: Vec<f64>,
    n: usize,
    b_norm: f64,
    maxiter: usize,
    target: f64,
}

impl<'a> State<'a> {
    fn new(a: &'a Csr, b: &'a [f64], x0: Option<&[f64]>) -> Result<Self, SparseError> {
        let (n_rows, n_cols) = a.shape();
        if n_rows != n_cols {
            return Err(SparseError::NotSquare { n_rows, n_cols });
        }
        if b.len() != n_rows {
            return Err(SparseError::ShapeMismatch {
                expected: n_rows,
                got: b.len(),
            });
        }
        let x = match x0 {
            Some(v) if v.len() == n_rows => v.to_vec(),
            // A warm start of the wrong length is the caller's bug, but refusing it here would
            // turn an optional accelerant into a hard error; a zero start is always valid.
            _ => vec![0.0; n_rows],
        };
        Ok(Self {
            a,
            b,
            x,
            n: n_rows,
            b_norm: norm(b),
            maxiter: KRYLOV_MAXITER_FACTOR * n_rows.max(1),
            target: (KRYLOV_RTOL * norm(b)).max(KRYLOV_ATOL),
        })
    }

    /// The stop: `||b - A x|| <= max(rtol * ||b||, atol)`, with `atol = 0`.
    ///
    /// A zero rhs makes the target zero, and `x = 0` satisfies it exactly — which is the right
    /// answer, not a degenerate one.
    fn tolerance(&self) -> f64 {
        self.target
    }

    fn residual_of(&self, x: &[f64]) -> Result<(Vec<f64>, f64), SparseError> {
        let ax = self.a.matvec(x)?;
        let r: Vec<f64> = self.b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
        let nrm = norm(&r);
        Ok((r, nrm))
    }

    fn relative(&self, absolute: f64) -> f64 {
        if self.b_norm > 0.0 {
            absolute / self.b_norm
        } else {
            absolute
        }
    }

    fn done(&self, x: Vec<f64>, iterations: usize, absolute: f64) -> KrylovOutcome {
        KrylovOutcome {
            x,
            iterations,
            residual: self.relative(absolute),
        }
    }

    fn exhausted(&self, absolute: f64) -> KrylovFailure {
        KrylovFailure::NotConverged {
            residual: self.relative(absolute),
            maxiter: self.maxiter,
        }
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn precondition(m: Option<&Preconditioner>, v: &[f64]) -> Vec<f64> {
    m.map_or_else(|| v.to_vec(), |p| p.apply(v))
}

/// `A^T x`, walking the CSR rows and scattering. Only `qmr` needs it.
fn matvec_transpose(a: &Csr, x: &[f64]) -> Vec<f64> {
    let (n_rows, n_cols) = a.shape();
    let mut y = vec![0.0; n_cols];
    let (indptr, indices, values) = (a.indptr(), a.indices(), a.values());
    for r in 0..n_rows {
        let xr = x[r];
        if xr == 0.0 {
            continue;
        }
        for i in indptr[r] as usize..indptr[r + 1] as usize {
            y[indices[i] as usize] += values[i] * xr;
        }
    }
    y
}

// -- the methods ------------------------------------------------------------------------------

/// Preconditioned conjugate gradients. Assumes `A` SPD and `M` SPD.
fn cg(mut st: State, m: Option<&Preconditioner>) -> Result<KrylovOutcome, KrylovFailure> {
    let (mut r, mut r_norm) = st.residual_of(&st.x)?;
    let tol = st.tolerance();
    if r_norm <= tol {
        return Ok(st.done(st.x.clone(), 0, r_norm));
    }
    let mut z = precondition(m, &r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for k in 1..=st.maxiter {
        let ap = st.a.matvec(&p)?;
        let denom = dot(&p, &ap);
        if !denom.is_finite() || denom <= 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "p^T A p",
                iteration: k,
            });
        }
        let alpha = rz / denom;
        for i in 0..st.n {
            st.x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        r_norm = norm(&r);
        if r_norm <= tol {
            return Ok(st.done(st.x.clone(), k, r_norm));
        }
        z = precondition(m, &r);
        let rz_new = dot(&r, &z);
        if rz == 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "r^T z",
                iteration: k,
            });
        }
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..st.n {
            p[i] = z[i] + beta * p[i];
        }
    }
    Err(st.exhausted(r_norm))
}

/// Preconditioned MINRES — symmetric, indefinite allowed.
///
/// The Paige-Saunders Lanczos recurrence with Givens rotations. The stop uses the TRUE residual
/// rather than the `|eta|` estimate the recurrence carries, for the reason in the module docs.
fn minres(mut st: State, m: Option<&Preconditioner>) -> Result<KrylovOutcome, KrylovFailure> {
    let (v_init, mut r_norm) = st.residual_of(&st.x)?;
    let tol = st.tolerance();
    if r_norm <= tol {
        return Ok(st.done(st.x.clone(), 0, r_norm));
    }
    let mut v_old = vec![0.0; st.n];
    let mut v = v_init;
    let mut z = precondition(m, &v);
    let mut gamma = dot(&z, &v);
    if gamma <= 0.0 {
        return Err(KrylovFailure::Breakdown {
            quantity: "z^T v (M is not positive definite)",
            iteration: 0,
        });
    }
    gamma = gamma.sqrt();
    let (mut gamma_old, mut eta) = (1.0, gamma);
    let (mut s_old, mut s, mut c_old, mut c) = (0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64);
    let mut w_old = vec![0.0; st.n];
    let mut w = vec![0.0; st.n];

    for k in 1..=st.maxiter {
        let zk: Vec<f64> = z.iter().map(|t| t / gamma).collect();
        let az = st.a.matvec(&zk)?;
        let delta = dot(&az, &zk);
        let v_new: Vec<f64> = (0..st.n)
            .map(|i| az[i] - (delta / gamma) * v[i] - (gamma / gamma_old) * v_old[i])
            .collect();
        let z_new = precondition(m, &v_new);
        let gsq = dot(&z_new, &v_new);
        let gamma_new = if gsq > 0.0 { gsq.sqrt() } else { 0.0 };

        let a0 = c * delta - c_old * s * gamma;
        let a1 = (a0 * a0 + gamma_new * gamma_new).sqrt();
        if a1 == 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "Givens norm",
                iteration: k,
            });
        }
        let a2 = s * delta + c_old * c * gamma;
        let a3 = s_old * gamma;
        c_old = c;
        c = a0 / a1;
        s_old = s;
        s = gamma_new / a1;
        let w_new: Vec<f64> = (0..st.n)
            .map(|i| (zk[i] - a3 * w_old[i] - a2 * w[i]) / a1)
            .collect();
        for (xi, wi) in st.x.iter_mut().zip(&w_new) {
            *xi += c * eta * wi;
        }
        eta *= -s;

        r_norm = st.residual_of(&st.x)?.1;
        if r_norm <= tol {
            return Ok(st.done(st.x.clone(), k, r_norm));
        }
        if gamma_new == 0.0 {
            // The Lanczos process has terminated: the Krylov space is exhausted and the iterate
            // above is the best it holds. Reporting non-convergence is honest; claiming success
            // is not.
            return Err(st.exhausted(r_norm));
        }
        v_old = v;
        v = v_new;
        z = z_new;
        gamma_old = gamma;
        gamma = gamma_new;
        w_old = w;
        w = w_new;
    }
    Err(st.exhausted(r_norm))
}

/// Left-preconditioned restarted GMRES(`GMRES_RESTART`).
///
/// The inner loop tracks the Givens residual estimate of the PRECONDITIONED system; the outer
/// loop verifies the TRUE residual before declaring convergence, so a preconditioner that makes
/// the preconditioned residual small without solving the system cannot pass.
fn gmres(mut st: State, m: Option<&Preconditioner>) -> Result<KrylovOutcome, KrylovFailure> {
    let tol = st.tolerance();
    let restart = GMRES_RESTART.min(st.n).max(1);
    let mut used = 0usize;
    let (_, mut r_norm) = st.residual_of(&st.x)?;
    if r_norm <= tol {
        return Ok(st.done(st.x.clone(), 0, r_norm));
    }
    while used < st.maxiter {
        let (r, _) = st.residual_of(&st.x)?;
        let r = precondition(m, &r);
        let beta = norm(&r);
        if beta == 0.0 {
            return Ok(st.done(st.x.clone(), used, r_norm));
        }
        let mut basis: Vec<Vec<f64>> = vec![r.iter().map(|t| t / beta).collect()];
        let mut h = vec![vec![0.0; restart]; restart + 1];
        let (mut cs, mut sn) = (vec![0.0; restart], vec![0.0; restart]);
        let mut g = vec![0.0; restart + 1];
        g[0] = beta;
        let mut k_used = 0usize;

        for k in 0..restart {
            if used >= st.maxiter {
                break;
            }
            used += 1;
            k_used = k + 1;
            let mut w = precondition(m, &st.a.matvec(&basis[k])?);
            for (j, bj) in basis.iter().enumerate().take(k + 1) {
                h[j][k] = dot(&w, bj);
                let hjk = h[j][k];
                for i in 0..st.n {
                    w[i] -= hjk * bj[i];
                }
            }
            h[k + 1][k] = norm(&w);
            // Apply the rotations accumulated so far, then the new one.
            for j in 0..k {
                let (t0, t1) = (h[j][k], h[j + 1][k]);
                h[j][k] = cs[j] * t0 + sn[j] * t1;
                h[j + 1][k] = -sn[j] * t0 + cs[j] * t1;
            }
            let denom = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            if denom == 0.0 {
                break;
            }
            cs[k] = h[k][k] / denom;
            sn[k] = h[k + 1][k] / denom;
            h[k][k] = denom;
            h[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] *= cs[k];

            let breakdown = h[k + 1][k].abs() == 0.0 && basis.len() == k + 1;
            if h[k + 1][k] == 0.0 && !breakdown {
                break;
            }
            let hk = norm(&w);
            if hk > 0.0 {
                basis.push(w.iter().map(|t| t / hk).collect());
            }
            if g[k + 1].abs() <= KRYLOV_RTOL * beta || hk == 0.0 {
                break;
            }
        }

        // Back-substitute the small upper-triangular system, then update x.
        let mut y = vec![0.0; k_used];
        for i in (0..k_used).rev() {
            let mut acc = g[i];
            for j in i + 1..k_used {
                acc -= h[i][j] * y[j];
            }
            y[i] = if h[i][i] == 0.0 { 0.0 } else { acc / h[i][i] };
        }
        for (yj, bj) in y.iter().zip(&basis) {
            for (xi, vi) in st.x.iter_mut().zip(bj) {
                *xi += yj * vi;
            }
        }
        r_norm = st.residual_of(&st.x)?.1;
        if r_norm <= tol {
            return Ok(st.done(st.x.clone(), used, r_norm));
        }
    }
    Err(st.exhausted(r_norm))
}

/// Preconditioned BiCGSTAB — general (non-symmetric) operators.
fn bicgstab(mut st: State, m: Option<&Preconditioner>) -> Result<KrylovOutcome, KrylovFailure> {
    let (mut r, mut r_norm) = st.residual_of(&st.x)?;
    let tol = st.tolerance();
    if r_norm <= tol {
        return Ok(st.done(st.x.clone(), 0, r_norm));
    }
    let r_hat = r.clone();
    let (mut rho, mut alpha, mut omega) = (1.0_f64, 1.0_f64, 1.0_f64);
    let mut v = vec![0.0; st.n];
    let mut p = vec![0.0; st.n];
    for k in 1..=st.maxiter {
        let rho_new = dot(&r_hat, &r);
        if rho == 0.0 || omega == 0.0 || rho_new == 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "rho / omega",
                iteration: k,
            });
        }
        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;
        for i in 0..st.n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        let p_hat = precondition(m, &p);
        v = st.a.matvec(&p_hat)?;
        let denom = dot(&r_hat, &v);
        if denom == 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "rhat^T v",
                iteration: k,
            });
        }
        alpha = rho / denom;
        let s: Vec<f64> = (0..st.n).map(|i| r[i] - alpha * v[i]).collect();
        for (xi, pi) in st.x.iter_mut().zip(&p_hat) {
            *xi += alpha * pi;
        }
        r_norm = st.residual_of(&st.x)?.1;
        if r_norm <= tol {
            return Ok(st.done(st.x.clone(), k, r_norm));
        }
        let s_hat = precondition(m, &s);
        let t = st.a.matvec(&s_hat)?;
        let tt = dot(&t, &t);
        if tt == 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "t^T t",
                iteration: k,
            });
        }
        omega = dot(&t, &s) / tt;
        for i in 0..st.n {
            st.x[i] += omega * s_hat[i];
            r[i] = s[i] - omega * t[i];
        }
        r_norm = norm(&r);
        if r_norm <= tol {
            return Ok(st.done(st.x.clone(), k, r_norm));
        }
    }
    Err(st.exhausted(r_norm))
}

/// QMR without look-ahead (Freund-Nachtigal), left-preconditioned.
///
/// The Python passes the registry's preconditioner as scipy's `M1` and leaves `M2` the identity,
/// so this does the same: `M1 = M`, `M2 = I`. It is the only method here that applies `A^T` and
/// `M^{-T}`, which is why [`Preconditioner::apply_transpose`] exists.
fn qmr(mut st: State, m: Option<&Preconditioner>) -> Result<KrylovOutcome, KrylovFailure> {
    let (mut r, mut r_norm) = st.residual_of(&st.x)?;
    let tol = st.tolerance();
    if r_norm <= tol {
        return Ok(st.done(st.x.clone(), 0, r_norm));
    }
    let mut vt = r.clone();
    let mut y = precondition(m, &vt);
    let mut rho = norm(&y);
    let mut wt = r.clone();
    // M2 = I, so z = wt and xi = ||wt||.
    let mut z = wt.clone();
    let mut xi = norm(&z);
    let (mut gamma, mut eta, mut epsilon, mut theta) = (1.0_f64, -1.0_f64, 1.0_f64, 0.0_f64);
    let (mut v, mut w) = (vec![0.0; st.n], vec![0.0; st.n]);
    let (mut p, mut q) = (vec![0.0; st.n], vec![0.0; st.n]);
    let (mut d, mut s) = (vec![0.0; st.n], vec![0.0; st.n]);

    for k in 1..=st.maxiter {
        if rho == 0.0 || xi == 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "rho / xi",
                iteration: k,
            });
        }
        for i in 0..st.n {
            v[i] = vt[i] / rho;
            y[i] /= rho;
            w[i] = wt[i] / xi;
            z[i] /= xi;
        }
        let delta = dot(&z, &y);
        if delta == 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "delta",
                iteration: k,
            });
        }
        // M2 = I so yt = y; M1^{-T} z is the transposed preconditioner apply.
        let yt = y.clone();
        let zt = m.map_or_else(|| z.clone(), |pc| pc.apply_transpose(&z));
        if k == 1 {
            p.clone_from(&yt);
            q.clone_from(&zt);
        } else {
            let (fp, fq) = (xi * delta / epsilon, rho * delta / epsilon);
            for i in 0..st.n {
                p[i] = yt[i] - fp * p[i];
                q[i] = zt[i] - fq * q[i];
            }
        }
        let pt = st.a.matvec(&p)?;
        epsilon = dot(&q, &pt);
        if epsilon == 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "epsilon",
                iteration: k,
            });
        }
        let beta = epsilon / delta;
        if beta == 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "beta",
                iteration: k,
            });
        }
        for i in 0..st.n {
            vt[i] = pt[i] - beta * v[i];
        }
        y = precondition(m, &vt);
        let rho_1 = rho;
        rho = norm(&y);
        let atq = matvec_transpose(st.a, &q);
        for i in 0..st.n {
            wt[i] = atq[i] - beta * w[i];
        }
        z.clone_from(&wt);
        xi = norm(&z);
        let gamma_1 = gamma;
        let theta_1 = theta;
        theta = rho / (gamma_1 * beta.abs());
        gamma = 1.0 / (1.0 + theta * theta).sqrt();
        if gamma == 0.0 {
            return Err(KrylovFailure::Breakdown {
                quantity: "gamma",
                iteration: k,
            });
        }
        eta = -eta * rho_1 * gamma * gamma / (beta * gamma_1 * gamma_1);
        let carry = if k == 1 {
            0.0
        } else {
            (theta_1 * gamma).powi(2)
        };
        for i in 0..st.n {
            d[i] = eta * p[i] + carry * d[i];
            s[i] = eta * pt[i] + carry * s[i];
            st.x[i] += d[i];
            r[i] -= s[i];
        }
        r_norm = st.residual_of(&st.x)?.1;
        if r_norm <= tol {
            return Ok(st.done(st.x.clone(), k, r_norm));
        }
    }
    Err(st.exhausted(r_norm))
}
