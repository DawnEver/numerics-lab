//! The dense matrix exponential, transcribed from `scipy.linalg.expm` (SciPy 1.18.1, `scipy/linalg/src/
//! _matfuncs_expm.c`, `matrix_exponential_d`; Al-Mohy & Higham 2009) so that its result is `scipy.linalg.expm`'s BIT
//! FOR BIT on the reference box (user ruling 2026-11-17: held-march partitions match the Python march exactly).
//!
//! Bit identity needs more than the algorithm: SciPy calls OpenBLAS (`scipy_openblas`, SKYLAKEX kernels on the
//! reference Xeon W-2223), so each BLAS/LAPACK call below reproduces that kernel's summation order and fused
//! multiply-adds -- [`gemm`] (one FMA chain over `k`, then `fma(beta, C, acc)`: the small-matrix kernel), [`getrf`]
//! (`getrf_single`: [`getf2`] up to `2 * GEMM_UNROLL_N` blocking -- left-looking, `ddot` and `dgemv_n` orders,
//! reciprocal pivot -- else the recursive blocked LU) and [`getrs_t`] (the `trsm` LT/LN kernels' remainder blocking
//! with the inverted diagonal). Storage follows the C: column-major work arrays, `swap_cf` the transposing copy.
//! Sources: OpenBLAS 0.3.30's x86_64 `ddot`, `dgemv_n_4` with its SKYLAKEX microkernel, the generic
//! `trsm_kernel_LT` / `trsm_kernel_LN`, `getrf_single` and `getrs_single`.
//!
//! Bit identity holds up to [`MAX_N`], the largest MEASURED identical: past it OpenBLAS changes path again (the
//! small-matrix `dgemm` permit ends at `n^3 > 1e6`, the `trsm` driver blocks at `GEMM_P = 192`, from 400 SciPy
//! estimates its norms), so a larger `n` runs the same transcription and agrees with SciPy to round-off, not to the
//! bit. It is not refused: a march with more than sixteen states (an MMC phase reaches 64) has no other exponential.
//!
//! Row-major `n x n` slices in and out; dimensionless (the caller has already multiplied `A` by its time step).
//! `crates/numerics/tests/expm_is_scipy.rs` holds the identity against SciPy's own outputs (the fixture
//! motronics-studio's `tests/unit/laplace/control/test_the_native_expm_is_scipys_bit_for_bit.py` writes and re-checks).

/// The largest size measured bit-identical to SciPy (`tests/expm_is_scipy.rs`: every size 2..16). Sizes 20..32 of the
/// same fixture miss by 4..512 ulp (a second recursion level of the blocked LU and a full 16-row `trsm` block meet
/// there; not yet followed): they are computed to round-off, not to the bit.
pub const MAX_N: usize = 16;

/// Why an exponential could not be formed.
#[derive(Debug, Clone, PartialEq)]
pub enum ExpmError {
    /// The slice is not `n * n` long.
    Shape { len: usize, n: usize },
    /// An entry is NaN or infinite.
    NonFinite,
    /// The Pade denominator `V - U` is exactly singular (LAPACK `getrf` info > 0).
    Singular,
}

impl std::fmt::Display for ExpmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape { len, n } => {
                write!(f, "expm needs an n x n matrix: {len} entries for n = {n}")
            }
            Self::NonFinite => write!(f, "expm needs finite entries"),
            Self::Singular => write!(f, "expm: the Pade denominator is singular"),
        }
    }
}

impl std::error::Error for ExpmError {}

/// `THETA` of `pick_pade_structure_d`.
const THETA: [f64; 5] = [
    1.495585217958292e-002,
    2.539398330063230e-001,
    9.504178996162932e-001,
    2.097847961257068e+000,
    4.25,
];
/// `coeff` of `pick_pade_structure_d` (the `ell` bounds of Al-Mohy & Higham).
const COEFF: [f64; 5] = [
    1.1191048088221578e-11,
    1.1167770708198077e-06,
    4.9826124310493469e-01,
    6.5662862500000000e+05,
    1.2573361865339437e+19,
];

/// OpenBLAS SKYLAKEX `DGEMM_DEFAULT_UNROLL_M`: the `trsm` kernels' row blocking.
const UNROLL_M: usize = 16;

/// `swap_cf_d`: `dst[j + i n] = src[i + j n]` (the transposing copy; its recursion for `n >= 16` only reorders).
fn swap_cf(src: &[f64], dst: &mut [f64], n: usize) {
    for j in 0..n {
        for i in 0..n {
            dst[j + i * n] = src[i + j * n];
        }
    }
}

/// `dgemm("N", "N", n, n, n, 1.0, a, n, b, n, beta, c, n)`, column-major: `c = beta c + acc`, `acc` one FMA chain
/// over `k` from zero (OpenBLAS's packed kernel: the beta pass scales `c` first, the kernel adds `alpha acc`).
fn gemm(a: &[f64], b: &[f64], beta: f64, c: &mut [f64], n: usize) {
    for j in 0..n {
        for i in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc = a[i + k * n].mul_add(b[k + j * n], acc);
            }
            c[i + j * n] = if beta == 0.0 {
                acc
            } else {
                beta.mul_add(c[i + j * n], acc)
            };
        }
    }
}

/// `dgemv("T", n, n, 1.0, a, n, x, 1, 0.0, y, 1)`: `y[i] = column i . x`, one FMA chain.
fn gemv_t(a: &[f64], x: &[f64], y: &mut [f64], n: usize) {
    for i in 0..n {
        let mut acc = 0.0f64;
        for j in 0..n {
            acc = a[j + i * n].mul_add(x[j], acc);
        }
        y[i] = acc;
    }
}

/// `dnorm1`: the largest absolute column sum, column-major.
fn norm1(a: &[f64], n: usize) -> f64 {
    let mut norm = 0.0f64;
    for i in 0..n {
        let mut sum = 0.0f64;
        for j in 0..n {
            sum += a[i * n + j].abs();
        }
        if sum > norm {
            norm = sum;
        }
    }
    norm
}

fn max_of(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, &x| if x > m { x } else { m })
}

/// `lm = max(0, (int)ceil(log2(temp / normA / coeff) / div))`.
fn ell(temp: f64, norm_a: f64, coeff: f64, div: f64) -> i32 {
    let lm = ((temp / norm_a / coeff).log2() / div).ceil() as i32;
    lm.max(0)
}

/// The work arrays of `matrix_exponential_d`: five powers, `|A|`, the spin pair.
struct Work {
    n: usize,
    am: Vec<f64>,
}

impl Work {
    fn slot(&self, k: usize) -> std::ops::Range<usize> {
        k * self.n * self.n..(k + 1) * self.n * self.n
    }

    /// `c = a @ b` (or `beta c + a @ b`) between slots.
    fn gemm(&mut self, a: usize, b: usize, beta: f64, c: usize) {
        let (ra, rb, rc) = (self.slot(a), self.slot(b), self.slot(c));
        let (x, y) = (self.am[ra].to_vec(), self.am[rb].to_vec());
        gemm(&x, &y, beta, &mut self.am[rc], self.n);
    }

    /// One `|A|^T` spin from `from` into `to` (offsets into the vector pair after the six slots).
    fn spin(&mut self, from: usize, to: usize) {
        let n = self.n;
        let base = 6 * n * n;
        let abs = self.am[self.slot(5)].to_vec();
        let x = self.am[base + from * n..base + (from + 1) * n].to_vec();
        gemv_t(
            &abs,
            &x,
            &mut self.am[base + to * n..base + (to + 1) * n],
            n,
        );
    }

    fn spins(&mut self, pairs: usize) -> f64 {
        for _ in 0..pairs {
            self.spin(1, 0);
            self.spin(0, 1);
        }
        let n = self.n;
        max_of(&self.am[6 * n * n + n..6 * n * n + 2 * n])
    }
}

/// `pick_pade_structure_d`: `(m, s)`; scales slots 0..4 by `2^-s, 4^-s, 16^-s, 64^-s` when `m = 13`.
fn pick_pade_structure(w: &mut Work) -> (usize, i32) {
    let n = w.n;
    let base = 6 * n * n;
    for i in 0..n {
        w.am[base + i] = 1.0;
    }
    for k in 0..n * n {
        w.am[5 * n * n + k] = w.am[k].abs();
    }
    w.spin(0, 1);
    let mut norm_a = max_of(&w.am[base + n..base + 2 * n]);

    w.gemm(0, 0, 0.0, 1);
    w.gemm(1, 1, 0.0, 2);
    w.gemm(2, 1, 0.0, 3);
    let d4 = norm1(&w.am[w.slot(2)], n).powf(0.25);
    let d6 = norm1(&w.am[w.slot(3)], n).powf(1.0 / 6.0);
    let eta0 = d4.max(d6);
    let eta1 = eta0;

    let temp = w.spins(3);
    if eta0 < THETA[0] && ell(temp, norm_a, COEFF[0], 6.0) == 0 {
        return (3, 0);
    }
    let temp = w.spins(2);
    if eta1 < THETA[1] && ell(temp, norm_a, COEFF[1], 10.0) == 0 {
        return (5, 0);
    }
    w.gemm(2, 2, 0.0, 4);
    let d8 = norm1(&w.am[w.slot(4)], n).powf(0.125);
    let eta2 = d6.max(d8);
    let temp = w.spins(2);
    if eta2 < THETA[2] && ell(temp, norm_a, COEFF[2], 14.0) == 0 {
        return (7, 0);
    }
    let temp = w.spins(2);
    if eta2 < THETA[3] && ell(temp, norm_a, COEFF[3], 18.0) == 0 {
        return (9, 0);
    }
    w.gemm(3, 2, 0.0, 4);
    let d10 = norm1(&w.am[w.slot(4)], n).powf(0.1);
    let eta3 = d8.max(d10);
    let eta4 = eta2.min(eta3);
    let mut s = ((eta4 / THETA[4]).log2().ceil() as i32).max(0);
    if s != 0 {
        let two_pow_s = 2f64.powf(-f64::from(s));
        for k in 5 * n * n..6 * n * n {
            w.am[k] *= two_pow_s;
        }
        let spun = 2f64.powf(-19.0 * f64::from(s));
        for k in base + n..base + 2 * n {
            w.am[k] *= spun;
        }
        norm_a *= 2f64.powf(-f64::from(s));
    }
    let temp = w.spins(4);
    s += ell(temp, norm_a, COEFF[4], 26.0);
    if s != 0 {
        let scales = [2f64, 4.0, 16.0, 64.0].map(|b| b.powf(-f64::from(s)));
        for (k, scale) in scales.iter().enumerate() {
            for v in &mut w.am[k * n * n..(k + 1) * n * n] {
                *v *= scale;
            }
        }
    }
    (13, s)
}

/// `pade_UV_calc_d` up to the solve: `U` in slot 3, `V` in slot 1.
fn pade_uv(w: &mut Work, m: usize) {
    let n = w.n;
    let n2 = n * n;
    let am = &mut w.am;
    match m {
        3 => {
            let b = [120.0, 60.0, 12.0];
            am.copy_within(0..n2, 3 * n2);
            w.gemm(1, 0, b[1], 3);
            let am = &mut w.am;
            for v in &mut am[n2..2 * n2] {
                *v *= b[2];
            }
            for i in 0..n {
                am[n2 + n * i + i] += b[0];
            }
        }
        5 => {
            let b = [30240.0, 15120.0, 3360.0, 420.0, 30.0];
            for i in 0..n {
                for j in 0..n {
                    let k = n * i + j;
                    am[4 * n2 + k] = am[2 * n2 + k] + b[3] * am[n2 + k];
                    if i == j {
                        am[4 * n2 + k] += b[1];
                    }
                }
            }
            w.gemm(4, 0, 0.0, 3);
            let am = &mut w.am;
            for i in 0..n {
                for j in 0..n {
                    let k = n * i + j;
                    am[n2 + k] = b[4] * am[2 * n2 + k] + b[2] * am[n2 + k];
                    if i == j {
                        am[n2 + k] += b[0];
                    }
                }
            }
        }
        7 => {
            let b = [
                17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0,
            ];
            for i in 0..n {
                for j in 0..n {
                    let k = n * i + j;
                    am[4 * n2 + k] = am[3 * n2 + k] + b[5] * am[2 * n2 + k] + b[3] * am[n2 + k];
                    if i == j {
                        am[4 * n2 + k] += b[1];
                    }
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let k = n * i + j;
                    am[n2 + k] = b[6] * am[3 * n2 + k] + b[4] * am[2 * n2 + k] + b[2] * am[n2 + k];
                    if i == j {
                        am[n2 + k] += b[0];
                    }
                }
            }
            w.gemm(4, 0, 0.0, 3);
        }
        9 => {
            let b = [
                17643225600.0,
                8821612800.0,
                2075673600.0,
                302702400.0,
                30270240.0,
                2162160.0,
                110880.0,
                3960.0,
                90.0,
            ];
            for i in 0..n {
                for j in 0..n {
                    let k = n * i + j;
                    let (t1, t2) = (am[4 * n2 + k], am[n2 + k]);
                    am[4 * n2 + k] = t1 + b[7] * am[3 * n2 + k] + b[5] * am[2 * n2 + k] + b[3] * t2;
                    am[n2 + k] =
                        b[8] * t1 + b[6] * am[3 * n2 + k] + b[4] * am[2 * n2 + k] + b[2] * t2;
                    if i == j {
                        am[4 * n2 + k] += b[1];
                        am[n2 + k] += b[0];
                    }
                }
            }
            w.gemm(4, 0, 0.0, 3);
        }
        _ => {
            let b = [
                64764752532480000.0,
                32382376266240000.0,
                7771770303897600.0,
                1187353796428800.0,
                129060195264000.0,
                10559470521600.0,
                670442572800.0,
                33522128640.0,
                1323241920.0,
                40840800.0,
                960960.0,
                16380.0,
                182.0,
            ];
            for i in 0..n {
                for j in 0..n {
                    let k = n * i + j;
                    let (t1, t2, t3) = (am[n2 + k], am[2 * n2 + k], am[3 * n2 + k]);
                    am[2 * n2 + k] = b[7] * t3 + b[5] * t2 + b[3] * t1;
                    am[4 * n2 + k] = b[12] * t3 + b[10] * t2 + b[8] * t1;
                    am[n2 + k] = b[6] * t3 + b[4] * t2 + b[2] * t1;
                    am[5 * n2 + k] = t3 + b[11] * t2 + b[9] * t1;
                    if i == j {
                        am[2 * n2 + k] += b[1];
                        am[n2 + k] += b[0];
                    }
                }
            }
            w.gemm(4, 3, 1.0, 1);
            w.gemm(5, 3, 1.0, 2);
            w.gemm(2, 0, 1.0, 3);
        }
    }
}

/// OpenBLAS `ddot` on a strided `x` (the `getf2` row of `L`): pairs of four into two partial sums, an FMA tail.
fn dot_strided(x: &[f64], inc_x: usize, y: &[f64], len: usize) -> f64 {
    let (mut t1, mut t2) = (0.0f64, 0.0f64);
    let mut i = 0;
    while i < len & !3 {
        let m3 = y[i + 2] * x[(i + 2) * inc_x];
        let m4 = y[i + 3] * x[(i + 3) * inc_x];
        t1 += y[i].mul_add(x[i * inc_x], m3);
        t2 += y[i + 1].mul_add(x[(i + 1) * inc_x], m4);
        i += 4;
    }
    while i < len {
        t1 = y[i].mul_add(x[i * inc_x], t1);
        i += 1;
    }
    t1 + t2
}

/// OpenBLAS `dgemv_n` with `alpha = -1` on rows `rows` of column-major `a` (`lda = n`), columns `0..cols`:
/// `y[r] -= (A x)[r]`. The leading `rows & !3` rows go through `dgemv_n_4.c`'s 4-row kernels -- column blocks of 4
/// (`dgemv_n_microk_skylakex-4.c`: `a0 x0 + a1 x1 + a2 x2 + a3 x3` as GCC contracts it), then 2 (`a0 x0 + a1 x1`),
/// then 1 (the asm `a (alpha x)`) -- the last `rows & 3` through the scalar remainder (one FMA chain).
fn gemv_n_minus(
    a: &[f64],
    n: usize,
    row0: usize,
    rows: usize,
    cols: usize,
    x: &[f64],
    y: &mut [f64],
) {
    if rows == 0 || cols == 0 {
        return;
    }
    let lead = rows & !3;
    let mut widths = vec![4usize; cols / 4];
    widths.extend([2usize, 1].into_iter().filter(|w| cols % 4 & w != 0));
    for r in row0..row0 + lead {
        let mut c = 0;
        for &width in &widths {
            let at = |q: usize| a[r + (c + q) * n];
            let t = match width {
                1 => at(0) * x[c],
                2 => at(0).mul_add(x[c], at(1) * x[c + 1]),
                _ => at(3).mul_add(
                    x[c + 3],
                    at(2).mul_add(x[c + 2], at(0).mul_add(x[c], at(1) * x[c + 1])),
                ),
            };
            y[r] = (-1.0f64).mul_add(t, y[r]);
            c += width;
        }
    }
    for r in row0 + lead..row0 + rows {
        let mut t = 0.0f64;
        for c in 0..cols {
            t = a[r + c * n].mul_add(x[c], t);
        }
        y[r] = (-1.0f64).mul_add(t, y[r]);
    }
}

/// OpenBLAS `getf2` (left-looking, reciprocal pivot) on a column-major `m x nc` panel (`lda = m`); `ipiv` zero-based
/// and relative to the panel.
fn getf2(a: &mut [f64], m: usize, nc: usize, ipiv: &mut [usize]) -> Result<(), ExpmError> {
    for j in 0..nc.min(m) {
        let col = j * m;
        for i in 0..j {
            let ip = ipiv[i];
            if ip != i {
                a.swap(col + i, col + ip);
            }
        }
        for i in 1..j {
            let b: Vec<f64> = a[col..col + i].to_vec();
            let d = dot_strided(&a[i..], m, &b, i);
            a[col + i] -= d;
        }
        let x: Vec<f64> = a[col..col + j].to_vec();
        let mut column: Vec<f64> = a[col..col + m].to_vec();
        gemv_n_minus(a, m, j, m - j, j, &x, &mut column);
        a[col..col + m].copy_from_slice(&column);
        let mut jp = j;
        let mut best = a[col + j].abs();
        for r in j + 1..m {
            if a[col + r].abs() > best {
                best = a[col + r].abs();
                jp = r;
            }
        }
        ipiv[j] = jp;
        let pivot = a[col + jp];
        if pivot == 0.0 {
            return Err(ExpmError::Singular);
        }
        if jp != j {
            for c in 0..=j {
                a.swap(j + c * m, jp + c * m);
            }
        }
        if pivot.abs() >= f64::MIN_POSITIVE {
            let inv = 1.0 / pivot;
            for r in j + 1..m {
                a[col + r] *= inv;
            }
        } else {
            for r in j + 1..m {
                a[col + r] /= pivot;
            }
        }
    }
    Ok(())
}

/// OpenBLAS SKYLAKEX `GEMM_UNROLL_N`: `getrf_single`'s blocking unit.
const UNROLL_N: usize = 2;

/// OpenBLAS `getrf_single` on the `m x nc` block of column-major `a` (`lda`) at diagonal offset `off`; `ipiv`
/// absolute. Up to `2 * UNROLL_N` blocking it is [`getf2`]; past it, a recursive panel LU, the row interchanges on
/// the trailing columns, a unit-lower `trsm` (LT kernel blocking) and a `GEMM_KERNEL` update (one FMA chain, then
/// `C - acc`), and last the later interchanges on the earlier columns.
pub(crate) fn getrf(
    a: &mut [f64],
    lda: usize,
    off: usize,
    m: usize,
    nc: usize,
    ipiv: &mut [usize],
) -> Result<(), ExpmError> {
    let mn = m.min(nc);
    let blocking = mn / 2 + UNROLL_N - 1;
    let blocking = blocking / UNROLL_N * UNROLL_N;
    if blocking <= 2 * UNROLL_N {
        let mut panel = vec![0.0; m * nc];
        for c in 0..nc {
            for r in 0..m {
                panel[r + c * m] = a[(off + r) + (off + c) * lda];
            }
        }
        let mut local = vec![0usize; mn];
        getf2(&mut panel, m, nc, &mut local)?;
        for c in 0..nc {
            for r in 0..m {
                a[(off + r) + (off + c) * lda] = panel[r + c * m];
            }
        }
        for (k, p) in local.iter().enumerate() {
            ipiv[off + k] = off + p;
        }
        return Ok(());
    }
    let at = |r: usize, c: usize| r + c * lda;
    let mut j = 0;
    while j < mn {
        let jb = (mn - j).min(blocking);
        let o = off + j;
        getrf(a, lda, o, m - j, jb, ipiv)?;
        if j + jb < nc {
            for c in o + jb..off + nc {
                for i in o..o + jb {
                    if ipiv[i] != i {
                        a.swap(at(i, c), at(ipiv[i], c));
                    }
                }
                for (start, size) in trsm_blocks(jb, true) {
                    for r in start..start + size {
                        if start > 0 {
                            let acc = (0..start).fold(0.0f64, |s, k| {
                                a[at(o + r, o + k)].mul_add(a[at(o + k, c)], s)
                            });
                            a[at(o + r, c)] = (-1.0f64).mul_add(acc, a[at(o + r, c)]);
                        }
                    }
                    for r in start..start + size {
                        let bb = a[at(o + r, c)];
                        for k in r + 1..start + size {
                            a[at(o + k, c)] = (-bb).mul_add(a[at(o + k, o + r)], a[at(o + k, c)]);
                        }
                    }
                }
            }
            for c in o + jb..off + nc {
                for r in o + jb..off + m {
                    let acc = (o..o + jb).fold(0.0f64, |s, k| a[at(r, k)].mul_add(a[at(k, c)], s));
                    a[at(r, c)] = (-1.0f64).mul_add(acc, a[at(r, c)]);
                }
            }
        }
        j += jb;
    }
    let mut j = 0;
    while j < mn {
        let jb = (mn - j).min(blocking);
        for c in off + j..off + j + jb {
            for i in off + j + jb..off + mn {
                if ipiv[i] != i {
                    a.swap(at(i, c), at(ipiv[i], c));
                }
            }
        }
        j += jb;
    }
    Ok(())
}

/// The `trsm` kernels' `GEMM_KERNEL` update sum over `ks`.
fn kernel_dot(a: impl Fn(usize) -> f64, c: &[f64], ks: std::ops::Range<usize>) -> f64 {
    ks.fold(0.0f64, |acc, k| a(k).mul_add(c[k], acc))
}

/// The row blocks the `trsm` kernels solve: full `UNROLL_M` blocks, then the remainder's powers of two.
/// `forward`: top down, remainder descending; else bottom up, remainder ascending first.
fn trsm_blocks(m: usize, forward: bool) -> Vec<(usize, usize)> {
    let full = m / UNROLL_M * UNROLL_M;
    let mut blocks = Vec::new();
    if forward {
        for b in (0..full).step_by(UNROLL_M) {
            blocks.push((b, UNROLL_M));
        }
        let mut start = full;
        let mut size = UNROLL_M >> 1;
        while size > 0 {
            if m & size != 0 {
                blocks.push((start, size));
                start += size;
            }
            size >>= 1;
        }
    } else {
        let mut size = 1;
        while size < UNROLL_M {
            if m & size != 0 {
                blocks.push(((m & !(size - 1)) - size, size));
            }
            size <<= 1;
        }
        for b in (0..full).step_by(UNROLL_M).rev() {
            blocks.push((b, UNROLL_M));
        }
    }
    blocks
}

/// `dgetrs("T", ...)`: `A^T X = B` with `A = P L U` from [`getf2`]; `U^T` forward (LT kernel, inverted diagonal),
/// `L^T` backward (LN kernel, unit), then the row interchanges in reverse.
fn getrs_t(lu: &[f64], n: usize, ipiv: &[usize], b: &mut [f64]) {
    // U^T(r, k) = lu[k + r n]; L^T(r, k) = lu[k + r n] for k > r.
    let inv: Vec<f64> = (0..n).map(|r| 1.0 / lu[r + r * n]).collect();
    for col in 0..n {
        let c = &mut b[col * n..(col + 1) * n];
        for (start, size) in trsm_blocks(n, true) {
            if start > 0 {
                for r in start..start + size {
                    let acc = kernel_dot(|k| lu[k + r * n], c, 0..start);
                    c[r] = (-1.0f64).mul_add(acc, c[r]);
                }
            }
            for r in start..start + size {
                let bb = c[r] * inv[r];
                c[r] = bb;
                for k in r + 1..start + size {
                    c[k] = (-bb).mul_add(lu[r + k * n], c[k]);
                }
            }
        }
        for (start, size) in trsm_blocks(n, false) {
            let end = start + size;
            if end < n {
                for r in start..end {
                    let acc = kernel_dot(|k| lu[k + r * n], c, end..n);
                    c[r] = (-1.0f64).mul_add(acc, c[r]);
                }
            }
            for r in (start..end).rev() {
                let bb = c[r];
                for k in start..r {
                    c[k] = (-bb).mul_add(lu[r + k * n], c[k]);
                }
            }
        }
    }
    for k in (0..n).rev() {
        if ipiv[k] != k {
            for col in 0..n {
                b.swap(k + col * n, ipiv[k] + col * n);
            }
        }
    }
}

/// `bandwidth_d` on the row-major copy: `(lower, upper)`.
fn bandwidth(a: &[f64], n: usize) -> (usize, usize) {
    let (mut lb, mut ub) = (0usize, 0usize);
    for r in (1..n).rev() {
        let limit = (r - lb).min(n);
        for c in 0..limit {
            if a[r * n + c] != 0.0 {
                lb = r - c;
                break;
            }
        }
        if r <= lb {
            break;
        }
    }
    for r in 0..n.saturating_sub(1) {
        let mut c = n - 1;
        while c > r + ub {
            if a[r * n + c] != 0.0 {
                ub = c - r;
                break;
            }
            c -= 1;
        }
        if r + ub + 1 > n {
            break;
        }
    }
    (lb, ub)
}

/// `exp(A)` of the row-major `n x n` matrix `a`: `scipy.linalg.expm(a)` bit for bit through [`MAX_N`], to
/// round-off past it.
pub fn expm(a: &[f64], n: usize) -> Result<Vec<f64>, ExpmError> {
    if a.len() != n * n {
        return Err(ExpmError::Shape { len: a.len(), n });
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(ExpmError::NonFinite);
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![a[0].exp()]);
    }
    let n2 = n * n;
    let (lband, uband) = bandwidth(a, n);
    if lband == 0 && uband == 0 {
        let mut out = vec![0.0; n2];
        for i in 0..n {
            out[i * n + i] = a[i * n + i].exp();
        }
        return Ok(out);
    }
    let mut w = Work {
        n,
        am: vec![0.0; 6 * n2 + 2 * n],
    };
    swap_cf(a, &mut w.am[..n2], n);
    let is_lower = uband == 0;
    let triangular = lband == 0 || uband == 0;
    let (mut diag, mut sd) = (vec![0.0; n], vec![0.0; n]);
    if triangular {
        let am = &w.am;
        if is_lower {
            for i in 0..n - 1 {
                diag[i] = am[i * n + i];
                sd[i] = am[i * n + i + 1];
            }
            diag[n - 1] = am[(n - 1) * n + n - 1];
        } else {
            diag[0] = am[0];
            for i in 0..n - 1 {
                sd[i] = am[(i + 1) * n + i];
                diag[i + 1] = am[(i + 1) * n + i + 1];
            }
        }
    }
    let (m, s) = pick_pade_structure(&mut w);
    pade_uv(&mut w, m);
    // V - U; X solves (V - U)^T X = U^T in slot 2; 2 X + I; back to slot 0.
    let am = &mut w.am;
    for k in 0..n2 {
        am[n2 + k] = (-1.0f64).mul_add(am[3 * n2 + k], am[n2 + k]);
    }
    let u = am[3 * n2..4 * n2].to_vec();
    swap_cf(&u, &mut am[2 * n2..3 * n2], n);
    let mut lu = am[n2..2 * n2].to_vec();
    let mut ipiv = vec![0usize; n];
    getrf(&mut lu, n, 0, n, n, &mut ipiv)?;
    let mut x = am[2 * n2..3 * n2].to_vec();
    getrs_t(&lu, n, &ipiv, &mut x);
    for v in &mut x {
        *v *= 2.0;
    }
    for i in 0..n {
        x[n * i + i] += 1.0;
    }
    let mut r = vec![0.0; n2];
    swap_cf(&x, &mut r, n);
    for iter in (0..s).rev() {
        let mut next = vec![0.0; n2];
        gemm(&r, &r, 0.0, &mut next, n);
        r = next;
        if triangular {
            let scale = (-f64::from(iter)).exp2();
            if is_lower {
                for i in 0..n - 1 {
                    let (di, dj) = (diag[i] * scale, diag[i + 1] * scale);
                    let diff = dj - di;
                    let sinch = if diff == 0.0 {
                        di.exp()
                    } else {
                        (dj.exp() - di.exp()) / diff
                    };
                    r[i * n + i] = di.exp();
                    r[i * n + i + 1] = sinch * sd[i] * scale;
                }
                r[(n - 1) * n + n - 1] = (diag[n - 1] * scale).exp();
            } else {
                r[0] = (diag[0] * scale).exp();
                for i in 0..n - 1 {
                    let (di, dj) = (diag[i] * scale, diag[i + 1] * scale);
                    let diff = dj - di;
                    let sinch = if diff == 0.0 {
                        di.exp()
                    } else {
                        (dj.exp() - di.exp()) / diff
                    };
                    r[(i + 1) * n + i] = sinch * sd[i] * scale;
                    r[(i + 1) * n + i + 1] = dj.exp();
                }
            }
        }
    }
    let mut out = vec![0.0; n2];
    swap_cf(&r, &mut out, n);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64], rel: f64) -> bool {
        let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= rel * scale)
    }

    #[test]
    fn a_diagonal_matrix_exponentiates_entrywise() {
        for d in [1e-3, 0.1, 0.5, 1.5, 4.0, 40.0] {
            let e = expm(&[d, 0.0, 0.0, -2.0 * d], 2).unwrap();
            assert_eq!(e, vec![d.exp(), 0.0, 0.0, (-2.0 * d).exp()]);
        }
    }

    #[test]
    fn a_rotation_generator_exponentiates_to_its_rotation_at_every_degree() {
        for w in [0.001, 0.1, 0.6, 1.5, 3.0, 30.0] {
            let e = expm(&[0.0, -w, w, 0.0], 2).unwrap();
            // squaring compounds round-off: 1e-12 of the unit entries covers w = 30 (s = 3)
            assert!(
                close(&e, &[w.cos(), -w.sin(), w.sin(), w.cos()], 1e-12),
                "{w}: {e:?}"
            );
        }
    }

    #[test]
    fn a_dense_three_by_three_matches_its_eigen_closed_form() {
        // A = S diag(-1, -2, -3) S^-1 with S = [[1,1,0],[0,1,1],[0,0,1]], S^-1 = [[1,-1,1],[0,1,-1],[0,0,1]]
        let a = [-1.0, -1.0, 1.0, 0.0, -2.0, -1.0, 0.0, 0.0, -3.0];
        for t in [0.01, 0.5, 2.0, 9.0] {
            let at: Vec<f64> = a.iter().map(|v| v * t).collect();
            let e = expm(&at, 3).unwrap();
            let (e1, e2, e3) = ((-t).exp(), (-2.0 * t).exp(), (-3.0 * t).exp());
            let want = [e1, e2 - e1, e1 - e2, 0.0, e2, e3 - e2, 0.0, 0.0, e3];
            assert!(close(&e, &want, 1e-13), "{t}: {e:?}");
        }
    }

    #[test]
    fn a_nilpotent_block_exponentiates_to_its_finite_series() {
        assert!(close(
            &expm(&[0.0, 7.0, 0.0, 0.0], 2).unwrap(),
            &[1.0, 7.0, 0.0, 1.0],
            1e-15
        ));
        assert!(close(
            &expm(&[0.0, 0.0, 7.0, 0.0], 2).unwrap(),
            &[1.0, 0.0, 7.0, 1.0],
            1e-15
        ));
    }

    #[test]
    fn planted_a_perturbed_lu_solve_misses_the_closed_form() {
        // the blocked solve must see every row: drop the forward update and a 3 x 3 dense case misses
        let lu = [2.0, 0.5, 0.25, 1.0, 3.0, 0.5, 1.0, 1.0, 4.0];
        let mut x = vec![1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        getrs_t(&lu, 3, &[0, 1, 2], &mut x);
        // (P L U)^T x = b with P = I: check A^T x = b
        let l = [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.25, 0.5, 1.0]];
        let u = [[2.0, 1.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 4.0]];
        let mut a = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = (0..3).map(|k| l[i][k] * u[k][j]).sum();
            }
        }
        let b = [1.0, 2.0, 3.0];
        let atx: Vec<f64> = (0..3)
            .map(|j| (0..3).map(|i| a[i][j] * x[i]).sum())
            .collect();
        assert!(close(&atx, &b, 1e-14), "{atx:?}");
        assert!(!close(&atx, &[1.0, 2.0, 3.5], 1e-3));
    }

    #[test]
    fn the_trsm_blocks_cover_every_row_once() {
        for m in 1..40 {
            for forward in [true, false] {
                let mut rows: Vec<usize> = trsm_blocks(m, forward)
                    .iter()
                    .flat_map(|&(s, k)| s..s + k)
                    .collect();
                rows.sort_unstable();
                assert_eq!(rows, (0..m).collect::<Vec<_>>(), "{m} {forward}");
            }
        }
    }

    #[test]
    fn a_non_finite_or_misshapen_matrix_is_refused_and_a_large_one_is_not() {
        assert_eq!(expm(&[f64::NAN], 1), Err(ExpmError::NonFinite));
        assert_eq!(expm(&[1.0, 2.0], 2), Err(ExpmError::Shape { len: 2, n: 2 }));
        let n = 64;
        let mut shift = vec![0.0; n * n];
        for i in 0..n - 1 {
            shift[i * n + i + 1] = 1.0;
        }
        // exp of the nilpotent shift: entry (0, k) is 1/k!
        let e = expm(&shift, n).expect("a 64-state exponential");
        assert!(
            (e[0] - 1.0).abs() < 1e-15 && (e[3] - 1.0 / 6.0).abs() < 1e-15,
            "{} {}",
            e[0],
            e[3]
        );
    }
}
