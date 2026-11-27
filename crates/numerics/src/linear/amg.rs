//! Aggregation algebraic multigrid for real SPD nodal operators -- the setup AND the V-cycle.
//!
//! The Rust owner of `python/numerics_lab/linear/amg.py`, which only validates and marshals.
//! Coarse spaces come from strong matrix couplings and a constant tentative near-null vector; no
//! mesh coordinates are needed.
//!
//! # The three choices, each an option (`keep-valuable-options`)
//!
//! - [`Coarsening`]: `Pairwise` (greedy strongest-neighbor pairs, each level shrinks ~2x) or
//!   `SmoothedAggregation` (Vanek-Mandel-Brezina root-node aggregates, ~3x per level in 2-D).
//! - `theta`: the strength threshold. Only `|a_ij| >= theta sqrt(a_ii a_jj)` may join an aggregate;
//!   an anisotropic operator needs `theta > 0` so its weak direction is not aggregated.
//! - [`Smoothing`] of the tentative prolongation: `None` (plain aggregation), `Filtered` (one Jacobi
//!   step on the operator with weak couplings lumped into the diagonal, so `P` follows the strong
//!   graph) or `Full` (one Jacobi step on the full operator).
//!
//! # numerics-lab #33: why `Full` smoothing of a pairwise hierarchy fills it
//!
//! A smoothed `P` reaches one stencil further than `T`, so `P^T A P` reaches two stencils further
//! than the aggregate graph. At a 3-4x reduction the coarse graph absorbs that; at pairwise's 2x it
//! does not, and every level's stencil widens -- measured on the 3-D anisotropic Robin operator,
//! operator complexity 10 / 19 / 30 / 43 at n = 8 / 12 / 16 / 20, setup ~N^2. Plain pairwise
//! aggregation keeps `T^T A T` on the aggregate graph: complexity ~2 at every size.
//! [`Smoothing::default_for`] therefore pairs `Pairwise` with `None` and `SmoothedAggregation` with
//! `Filtered`; `Full` stays selectable and is the planted control of the tests.

use crate::sparse::Csr;

/// How a level's aggregates are formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coarsening {
    Pairwise,
    SmoothedAggregation,
}

/// How the tentative prolongation is smoothed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Smoothing {
    None,
    Filtered,
    Full,
}

impl Coarsening {
    /// The wire spellings, in one place so the parser and its error cannot drift.
    pub const KNOWN: [&'static str; 2] = ["pairwise", "smoothed-aggregation"];

    /// Parse a wire spelling.
    ///
    /// # Errors
    /// [`AmgError::Unknown`] for any other name.
    pub fn parse(name: &str) -> Result<Self, AmgError> {
        match name {
            "pairwise" => Ok(Self::Pairwise),
            "smoothed-aggregation" => Ok(Self::SmoothedAggregation),
            _ => Err(AmgError::Unknown {
                what: "coarsening",
                name: name.to_owned(),
                known: &Self::KNOWN,
            }),
        }
    }
}

impl Smoothing {
    pub const KNOWN: [&'static str; 3] = ["none", "filtered", "full"];

    /// Parse a wire spelling.
    ///
    /// # Errors
    /// [`AmgError::Unknown`] for any other name.
    pub fn parse(name: &str) -> Result<Self, AmgError> {
        match name {
            "none" => Ok(Self::None),
            "filtered" => Ok(Self::Filtered),
            "full" => Ok(Self::Full),
            _ => Err(AmgError::Unknown {
                what: "smoothing",
                name: name.to_owned(),
                known: &Self::KNOWN,
            }),
        }
    }

    /// The smoothing that keeps the coarsening's hierarchy sparse (module docs, #33).
    #[must_use]
    pub const fn default_for(coarsening: Coarsening) -> Self {
        match coarsening {
            Coarsening::Pairwise => Self::None,
            Coarsening::SmoothedAggregation => Self::Filtered,
        }
    }
}

/// Every way an AMG setup can refuse.
#[derive(Debug, Clone, PartialEq)]
pub enum AmgError {
    Unknown {
        what: &'static str,
        name: String,
        known: &'static [&'static str],
    },
    Theta(f64),
    NotSquare {
        shape: (usize, usize),
    },
    NonFinite,
    NonPositiveDiagonal {
        level: usize,
        row: usize,
    },
    CoarseNotPositiveDefinite,
    RhsLength {
        got: usize,
        want: usize,
    },
}

impl std::fmt::Display for AmgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { what, name, known } => {
                write!(f, "unknown AMG {what} {name:?}; known: {known:?}")
            }
            Self::Theta(t) => write!(
                f,
                "AMG strength threshold theta must lie in [0, 1), got {t}"
            ),
            Self::NotSquare { shape } => {
                write!(
                    f,
                    "AMG requires a finite real nonempty square SPD operator, got shape {shape:?}"
                )
            }
            Self::NonFinite => write!(f, "AMG requires a finite real nonempty square SPD operator"),
            Self::NonPositiveDiagonal { level, row } => {
                write!(
                    f,
                    "AMG requires positive diagonals on every level (level {level}, row {row})"
                )
            }
            Self::CoarseNotPositiveDefinite => {
                write!(f, "AMG coarse operator is not positive definite")
            }
            Self::RhsLength { got, want } => {
                write!(f, "AMG rhs has {got} entries, the operator {want} rows")
            }
        }
    }
}

impl std::error::Error for AmgError {}

/// The coarsest size solved by a dense Cholesky factor.
pub const COARSEST: usize = 32;

/// A CSR matrix with `usize` offsets -- the setup's working format (products, transposes).
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub n_rows: usize,
    pub n_cols: usize,
    pub ptr: Vec<usize>,
    pub idx: Vec<usize>,
    pub val: Vec<f64>,
}

impl Mat {
    fn from_csr(a: &Csr) -> Self {
        let (n_rows, n_cols) = a.shape();
        Self {
            n_rows,
            n_cols,
            ptr: a.indptr().iter().map(|&p| p as usize).collect(),
            idx: a.indices().iter().map(|&i| i as usize).collect(),
            val: a.values().to_vec(),
        }
    }

    fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let (s, e) = (self.ptr[i], self.ptr[i + 1]);
        (&self.idx[s..e], &self.val[s..e])
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }

    fn diagonal(&self) -> Vec<f64> {
        (0..self.n_rows)
            .map(|i| {
                let (cols, vals) = self.row(i);
                cols.iter().position(|&c| c == i).map_or(0.0, |k| vals[k])
            })
            .collect()
    }

    fn matvec_into(&self, x: &[f64], y: &mut [f64]) {
        for (i, out) in y.iter_mut().enumerate() {
            let (cols, vals) = self.row(i);
            *out = cols.iter().zip(vals).map(|(&c, &v)| v * x[c]).sum();
        }
    }

    fn transpose(&self) -> Self {
        let mut count = vec![0usize; self.n_cols + 1];
        for &c in &self.idx {
            count[c + 1] += 1;
        }
        for k in 0..self.n_cols {
            count[k + 1] += count[k];
        }
        let ptr = count.clone();
        let mut next = count;
        let mut idx = vec![0; self.nnz()];
        let mut val = vec![0.0; self.nnz()];
        for i in 0..self.n_rows {
            let (cols, vals) = self.row(i);
            for (&c, &v) in cols.iter().zip(vals) {
                idx[next[c]] = i;
                val[next[c]] = v;
                next[c] += 1;
            }
        }
        Self {
            n_rows: self.n_cols,
            n_cols: self.n_rows,
            ptr,
            idx,
            val,
        }
    }

    /// `self @ other` (Gustavson), columns ascending; exact zeros are kept as the pattern says.
    fn matmul(&self, other: &Self) -> Self {
        let mut ptr = Vec::with_capacity(self.n_rows + 1);
        ptr.push(0);
        let (mut idx, mut val) = (Vec::new(), Vec::new());
        let mut acc = vec![0.0; other.n_cols];
        let mut seen = vec![usize::MAX; other.n_cols];
        let mut touched = Vec::new();
        for i in 0..self.n_rows {
            touched.clear();
            let (cols, vals) = self.row(i);
            for (&k, &a) in cols.iter().zip(vals) {
                let (ocols, ovals) = other.row(k);
                for (&j, &b) in ocols.iter().zip(ovals) {
                    if seen[j] != i {
                        seen[j] = i;
                        acc[j] = 0.0;
                        touched.push(j);
                    }
                    acc[j] += a * b;
                }
            }
            touched.sort_unstable();
            for &j in &touched {
                idx.push(j);
                val.push(acc[j]);
            }
            ptr.push(idx.len());
        }
        Self {
            n_rows: self.n_rows,
            n_cols: other.n_cols,
            ptr,
            idx,
            val,
        }
    }

    /// `max_i sum_j |a_ij| / d_i` -- the Gershgorin bound on the spectrum of `D^-1 A`.
    fn gershgorin(&self, diagonal: &[f64]) -> f64 {
        (0..self.n_rows)
            .map(|i| self.row(i).1.iter().map(|v| v.abs()).sum::<f64>() / diagonal[i])
            .fold(0.0, f64::max)
    }
}

/// The strong off-diagonal couplings `|a_ij| >= theta sqrt(a_ii a_jj)`; `theta = 0` keeps every nonzero one.
#[must_use]
pub fn strong_connections(a: &Mat, theta: f64) -> Mat {
    let diagonal: Vec<f64> = a.diagonal().iter().map(|d| d.abs()).collect();
    let mut ptr = vec![0];
    let (mut idx, mut val) = (Vec::new(), Vec::new());
    for i in 0..a.n_rows {
        let (cols, vals) = a.row(i);
        for (&j, &v) in cols.iter().zip(vals) {
            if j != i && v != 0.0 && v.abs() >= theta * (diagonal[i] * diagonal[j]).sqrt() {
                idx.push(j);
                val.push(v);
            }
        }
        ptr.push(idx.len());
    }
    Mat {
        n_rows: a.n_rows,
        n_cols: a.n_cols,
        ptr,
        idx,
        val,
    }
}

/// Greedy strongest-neighbor pairs over the strong graph; unmatched rows are singletons.
fn pairwise_labels(strong: &Mat) -> (Vec<usize>, usize) {
    let mut labels = vec![usize::MAX; strong.n_rows];
    let mut count = 0;
    for row in 0..strong.n_rows {
        if labels[row] != usize::MAX {
            continue;
        }
        labels[row] = count;
        let (cols, vals) = strong.row(row);
        // The FIRST strongest free neighbor, as numpy's argmax picks it.
        let mut best: Option<(usize, f64)> = None;
        for (&c, &v) in cols.iter().zip(vals) {
            if labels[c] == usize::MAX && best.is_none_or(|(_, b)| v.abs() > b) {
                best = Some((c, v.abs()));
            }
        }
        if let Some((neighbor, _)) = best {
            labels[neighbor] = count;
        }
        count += 1;
    }
    (labels, count)
}

/// Root-node aggregation: a free node whose strong neighbors are all free roots them; leftovers join
/// their strongest aggregated neighbor; what is still free roots new aggregates with its free neighbors.
fn root_node_labels(strong: &Mat) -> (Vec<usize>, usize) {
    const FREE: usize = usize::MAX;
    let mut labels = vec![FREE; strong.n_rows];
    let mut count = 0;
    for row in 0..strong.n_rows {
        let cols = strong.row(row).0;
        if labels[row] == FREE && !cols.is_empty() && cols.iter().all(|&c| labels[c] == FREE) {
            labels[row] = count;
            for &c in cols {
                labels[c] = count;
            }
            count += 1;
        }
    }
    let mut attached = labels.clone();
    for row in (0..strong.n_rows).filter(|&r| labels[r] == FREE) {
        let (cols, vals) = strong.row(row);
        let mut best: Option<(usize, f64)> = None;
        for (&c, &v) in cols.iter().zip(vals) {
            if labels[c] != FREE && best.is_none_or(|(_, b)| v.abs() > b) {
                best = Some((labels[c], v.abs()));
            }
        }
        if let Some((aggregate, _)) = best {
            attached[row] = aggregate;
        }
    }
    labels = attached;
    for row in 0..strong.n_rows {
        if labels[row] != FREE {
            continue;
        }
        labels[row] = count;
        for &c in strong.row(row).0 {
            if labels[c] == FREE {
                labels[c] = count;
            }
        }
        count += 1;
    }
    (labels, count)
}

/// One aggregate label per row, and the aggregate count, of one level.
#[must_use]
pub fn aggregate_labels(a: &Mat, coarsening: Coarsening, theta: f64) -> (Vec<usize>, usize) {
    let strong = strong_connections(a, theta);
    match coarsening {
        Coarsening::Pairwise => pairwise_labels(&strong),
        Coarsening::SmoothedAggregation => root_node_labels(&strong),
    }
}

fn tentative(labels: &[usize], count: usize) -> Mat {
    Mat {
        n_rows: labels.len(),
        n_cols: count,
        ptr: (0..=labels.len()).collect(),
        idx: labels.to_vec(),
        val: vec![1.0; labels.len()],
    }
}

/// `(1 - w/rho D^-1 F) T` -- one damped Jacobi step of `F` on the tentative prolongation.
fn smooth(operator: &Mat, t: &Mat, weight: f64) -> Mat {
    let diagonal = operator.diagonal();
    let upper = operator.gershgorin(&diagonal);
    let mut ft = operator.matmul(t);
    for i in 0..ft.n_rows {
        let scale = weight / (upper * diagonal[i]);
        for v in &mut ft.val[ft.ptr[i]..ft.ptr[i + 1]] {
            *v *= -scale;
        }
    }
    // t has one entry per row: add it into row i of -(D^-1 F T).
    for i in 0..ft.n_rows {
        let column = t.idx[i];
        let (s, e) = (ft.ptr[i], ft.ptr[i + 1]);
        match ft.idx[s..e].binary_search(&column) {
            Ok(k) => ft.val[s + k] += 1.0,
            Err(_) => unreachable!("F T holds T's own pattern: F has a diagonal"),
        }
    }
    ft
}

/// The filtered operator: strong couplings kept, `a_ii` plus the row's weak couplings on the diagonal.
fn filtered(a: &Mat, theta: f64) -> Mat {
    let strong = strong_connections(a, theta);
    let mut out = Mat {
        n_rows: a.n_rows,
        n_cols: a.n_cols,
        ptr: vec![0],
        idx: Vec::new(),
        val: Vec::new(),
    };
    for i in 0..a.n_rows {
        let (cols, vals) = a.row(i);
        let strong_cols = strong.row(i).0;
        let mut lumped = 0.0;
        for (&j, &v) in cols.iter().zip(vals) {
            if j == i || strong_cols.binary_search(&j).is_err() {
                lumped += v;
            }
        }
        let mut placed = false;
        for (&j, &v) in strong.row(i).0.iter().zip(strong.row(i).1) {
            if !placed && j > i {
                out.idx.push(i);
                out.val.push(lumped);
                placed = true;
            }
            out.idx.push(j);
            out.val.push(v);
        }
        if !placed {
            out.idx.push(i);
            out.val.push(lumped);
        }
        out.ptr.push(out.idx.len());
    }
    out
}

/// One level's prolongation, or `None` when the coarsening cannot shrink it.
fn prolongation(a: &Mat, coarsening: Coarsening, theta: f64, smoothing: Smoothing) -> Option<Mat> {
    let (labels, count) = aggregate_labels(a, coarsening, theta);
    if count == a.n_rows {
        return None;
    }
    let t = tentative(&labels, count);
    Some(match smoothing {
        Smoothing::None => t,
        // Strict damping keeps I - B A nonsingular at the spectral bound, so no tentative column is erased.
        Smoothing::Full => smooth(a, &t, 2.0 / 3.0),
        // 4/3 over a Gershgorin bound of the filtered operator stays below 2 / rho.
        Smoothing::Filtered => smooth(&filtered(a, theta), &t, 4.0 / 3.0),
    })
}

struct Level {
    a: Mat,
    inverse: Vec<f64>,
    p: Mat,
    pt: Mat,
}

enum Coarse {
    /// Lower Cholesky factor, row-major `n x n`.
    Cholesky { n: usize, l: Vec<f64> },
    /// No strong coupling left to aggregate and too large to factor: one Jacobi sweep.
    Jacobi(Vec<f64>),
}

/// A built hierarchy: apply it as `M ~ A^-1` with [`Amg::apply`].
pub struct Amg {
    levels: Vec<Level>,
    coarse: Coarse,
    level_sizes: Vec<usize>,
    level_nnz: Vec<usize>,
}

fn cholesky(a: &Mat) -> Option<Vec<f64>> {
    let n = a.n_rows;
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        let (cols, vals) = a.row(i);
        for (&j, &v) in cols.iter().zip(vals) {
            l[i * n + j] = v;
        }
    }
    for j in 0..n {
        let mut d = l[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if !(d > 0.0 && d.is_finite()) {
            return None;
        }
        let d = d.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut s = l[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
        for i in 0..j {
            l[i * n + j] = 0.0;
        }
    }
    Some(l)
}

impl Amg {
    /// Build the hierarchy of a symmetric positive-definite `a` (columns ascending and unique).
    ///
    /// # Errors
    /// [`AmgError`] when `a` is not square/finite, `theta` lies outside `[0, 1)`, a level has a
    /// nonpositive diagonal, or the coarsest operator is not positive definite.
    pub fn new(
        a: &Csr,
        coarsening: Coarsening,
        theta: f64,
        smoothing: Smoothing,
    ) -> Result<Self, AmgError> {
        if !(0.0..1.0).contains(&theta) {
            return Err(AmgError::Theta(theta));
        }
        let (rows, cols) = a.shape();
        if rows != cols || rows == 0 {
            return Err(AmgError::NotSquare {
                shape: (rows, cols),
            });
        }
        if a.values().iter().any(|v| !v.is_finite()) {
            return Err(AmgError::NonFinite);
        }
        let mut matrix = Mat::from_csr(a);
        let mut levels = Vec::new();
        let mut level_sizes = Vec::new();
        let mut level_nnz = Vec::new();
        loop {
            level_sizes.push(matrix.n_rows);
            level_nnz.push(matrix.nnz());
            let diagonal = matrix.diagonal();
            if let Some(row) = diagonal.iter().position(|&d| d <= 0.0) {
                return Err(AmgError::NonPositiveDiagonal {
                    level: levels.len(),
                    row,
                });
            }
            if matrix.n_rows <= COARSEST {
                break;
            }
            let Some(p) = prolongation(&matrix, coarsening, theta, smoothing) else {
                break;
            };
            let upper = matrix.gershgorin(&diagonal);
            let inverse = diagonal.iter().map(|d| (2.0 / 3.0) / (upper * d)).collect();
            let pt = p.transpose();
            let coarse = pt.matmul(&matrix.matmul(&p));
            levels.push(Level {
                a: std::mem::replace(&mut matrix, coarse),
                inverse,
                p,
                pt,
            });
        }
        let coarse = if matrix.n_rows <= COARSEST {
            Coarse::Cholesky {
                n: matrix.n_rows,
                l: cholesky(&matrix).ok_or(AmgError::CoarseNotPositiveDefinite)?,
            }
        } else {
            Coarse::Jacobi(matrix.diagonal().iter().map(|d| 1.0 / d).collect())
        };
        Ok(Self {
            levels,
            coarse,
            level_sizes,
            level_nnz,
        })
    }

    /// The row count of every level, finest first.
    #[must_use]
    pub fn level_sizes(&self) -> &[usize] {
        &self.level_sizes
    }

    /// `sum(nnz_l) / nnz_0`: the hierarchy's storage and per-cycle work relative to the fine operator.
    #[must_use]
    pub fn operator_complexity(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let ratio = self.level_nnz.iter().sum::<usize>() as f64 / self.level_nnz[0] as f64;
        ratio
    }

    /// One symmetric V-cycle (pre and post damped Jacobi): `M^-1 rhs`.
    ///
    /// # Errors
    /// [`AmgError::RhsLength`] when `rhs` does not match the operator.
    pub fn apply(&self, rhs: &[f64]) -> Result<Vec<f64>, AmgError> {
        if rhs.len() != self.level_sizes[0] {
            return Err(AmgError::RhsLength {
                got: rhs.len(),
                want: self.level_sizes[0],
            });
        }
        Ok(self.cycle(0, rhs))
    }

    fn cycle(&self, level: usize, rhs: &[f64]) -> Vec<f64> {
        let Some(Level { a, inverse, p, pt }) = self.levels.get(level) else {
            return self.coarse_solve(rhs);
        };
        let mut estimate: Vec<f64> = inverse.iter().zip(rhs).map(|(w, r)| w * r).collect();
        let mut work = vec![0.0; rhs.len()];
        a.matvec_into(&estimate, &mut work);
        let residual: Vec<f64> = rhs.iter().zip(&work).map(|(r, w)| r - w).collect();
        let mut restricted = vec![0.0; pt.n_rows];
        pt.matvec_into(&residual, &mut restricted);
        let correction = self.cycle(level + 1, &restricted);
        p.matvec_into(&correction, &mut work);
        for (e, c) in estimate.iter_mut().zip(&work) {
            *e += c;
        }
        a.matvec_into(&estimate, &mut work);
        for i in 0..rhs.len() {
            estimate[i] += inverse[i] * (rhs[i] - work[i]);
        }
        estimate
    }

    fn coarse_solve(&self, rhs: &[f64]) -> Vec<f64> {
        match &self.coarse {
            Coarse::Jacobi(inverse) => inverse.iter().zip(rhs).map(|(w, r)| w * r).collect(),
            Coarse::Cholesky { n, l } => {
                let n = *n;
                let mut y = rhs.to_vec();
                for i in 0..n {
                    for k in 0..i {
                        y[i] -= l[i * n + k] * y[k];
                    }
                    y[i] /= l[i * n + i];
                }
                for i in (0..n).rev() {
                    for k in i + 1..n {
                        y[i] -= l[k * n + i] * y[k];
                    }
                    y[i] /= l[i * n + i];
                }
                y
            }
        }
    }
}

/// [`strong_connections`] of a [`Csr`], as a [`Mat`].
#[must_use]
pub fn strong_connections_of(a: &Csr, theta: f64) -> Mat {
    strong_connections(&Mat::from_csr(a), theta)
}

/// [`aggregate_labels`] of a [`Csr`].
#[must_use]
pub fn aggregate_labels_of(a: &Csr, coarsening: Coarsening, theta: f64) -> (Vec<usize>, usize) {
    aggregate_labels(&Mat::from_csr(a), coarsening, theta)
}
