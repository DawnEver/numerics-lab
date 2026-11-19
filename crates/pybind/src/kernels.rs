//! The generic kernels moved from motronics-studio (2026-11-19): dense `expm` / `solve` / `cond`, the
//! symbolic fill of a direct factor, and block static condensation. Thin glue over [`numerics`].

use numerics::dense;
use numerics::linear::condense::{block_matvec, CondensedCells};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Arr<'py> = Bound<'py, PyArray1<f64>>;

/// A square row-major matrix's entries and order, refusing anything not square.
fn square(a: &PyReadonlyArray2<'_, f64>) -> PyResult<(Vec<f64>, usize)> {
    let v = a.as_array();
    let (r, c) = v.dim();
    if r != c {
        return Err(PyValueError::new_err(format!(
            "a square matrix is needed, got {r} x {c}"
        )));
    }
    Ok((v.iter().copied().collect(), r))
}

fn nonneg(a: &PyReadonlyArray1<'_, i64>, what: &str) -> PyResult<Vec<usize>> {
    a.as_array()
        .iter()
        .map(|&v| {
            usize::try_from(v)
                .map_err(|_| PyValueError::new_err(format!("{what} holds a negative entry {v}")))
        })
        .collect()
}

/// `expm(a)` -> row-major entries of `scipy.linalg.expm(a)`, bit for bit up to
/// `EXPM_BIT_IDENTICAL_MAX_N`, to round-off past it.
#[pyfunction]
#[pyo3(name = "expm")]
pub fn py_expm<'py>(py: Python<'py>, a: PyReadonlyArray2<'py, f64>) -> PyResult<Arr<'py>> {
    let (data, n) = square(&a)?;
    let e = dense::expm(&data, n).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(e.into_pyarray(py))
}

/// `dense_solve(a, b)` -> `np.linalg.solve(a, b)` bit for bit, or `None` on an exactly zero pivot.
#[pyfunction]
#[pyo3(name = "dense_solve")]
pub fn py_dense_solve<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<Option<Arr<'py>>> {
    let (data, n) = square(&a)?;
    let b = b.as_array().to_vec();
    if b.len() != n {
        return Err(PyValueError::new_err(format!(
            "b has {} entries for n = {n}",
            b.len()
        )));
    }
    Ok(dense::solve(&data, n, &b).map(|x| x.into_pyarray(py)))
}

/// `dense_cond(a)` -> `np.linalg.cond(a)` bit for bit (`n <= COND_MAX_N`); `None` where numpy's SVD raises.
#[pyfunction]
#[pyo3(name = "dense_cond")]
pub fn py_dense_cond(a: PyReadonlyArray2<'_, f64>) -> PyResult<Option<f64>> {
    let (data, n) = square(&a)?;
    match dense::cond(&data, n) {
        Ok(c) => Ok(Some(c)),
        Err(dense::CondError::NotConverged) => Ok(None),
        Err(dense::CondError::Untranscribed { n }) => Err(PyValueError::new_err(format!(
            "cond is transcribed for n <= {}, got {n}",
            dense::COND_MAX_N
        ))),
    }
}

/// `cholesky_factor_len(n, indptr, indices)` -> `nnz(L)` of the AMD-ordered Cholesky factor of a
/// symmetric pattern, from the symbolic analysis alone: read it BEFORE a direct factor allocates.
#[pyfunction]
#[pyo3(name = "cholesky_factor_len")]
pub fn py_cholesky_factor_len(
    n: usize,
    indptr: PyReadonlyArray1<'_, i64>,
    indices: PyReadonlyArray1<'_, i64>,
) -> PyResult<usize> {
    numerics::linear::cholesky_factor_len(
        n,
        nonneg(&indptr, "indptr")?,
        nonneg(&indices, "indices")?,
    )
    .map_err(PyValueError::new_err)
}

/// Every block's condensed factors, held for the step's elimination and recovery.
#[pyclass(name = "CondensedCells")]
pub struct PyCondensedCells {
    inner: CondensedCells,
}

fn check_len(what: &str, got: usize, want: usize) -> PyResult<()> {
    if got == want {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{what} has {got} entries, {want} expected"
        )))
    }
}

#[pymethods]
impl PyCondensedCells {
    /// `(y, J_nb y)`: `y = J_bb^-1 rhs_b` per block (`(sum nb,)`) and its boundary correction (`(sum k,)`).
    fn eliminate<'py>(
        &self,
        py: Python<'py>,
        rhs_b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Arr<'py>, Arr<'py>)> {
        check_len("rhs_b", rhs_b.len()?, *self.inner.off_nb.last().unwrap())?;
        let rhs = rhs_b.as_array().to_vec();
        let (y, corr) = py.detach(|| self.inner.eliminate(&rhs));
        Ok((y.into_pyarray(py), corr.into_pyarray(py)))
    }

    /// `(sum nb,)`: every block's interior step `y - Z dv` from its boundary step `dv` (`(sum k,)`).
    fn recover<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        dv: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Arr<'py>> {
        check_len("y", y.len()?, *self.inner.off_nb.last().unwrap())?;
        check_len("dv", dv.len()?, *self.inner.off_k.last().unwrap())?;
        let (y, dv) = (y.as_array().to_vec(), dv.as_array().to_vec());
        Ok(py.detach(|| self.inner.recover(&y, &dv)).into_pyarray(py))
    }
}

/// `condense(j, sizes)` -> `(CondensedCells, schur)`: every row-major `(k + nb)^2` block of `j`
/// (concatenated; `sizes (E, 2)` rows `(k, nb)`) condensed onto its boundary; `schur` the concatenated
/// row-major `(k, k)` Schur complements.
#[pyfunction]
#[pyo3(name = "condense")]
pub fn py_condense<'py>(
    py: Python<'py>,
    j: PyReadonlyArray1<'py, f64>,
    sizes: PyReadonlyArray2<'py, i64>,
) -> PyResult<(PyCondensedCells, Arr<'py>)> {
    let s = sizes.as_array();
    if s.shape()[1] != 2 || s.iter().any(|v| *v < 0) {
        return Err(PyValueError::new_err(format!(
            "sizes must be (E, 2) nonnegative, got {:?}",
            s.shape()
        )));
    }
    let sizes: Vec<(usize, usize)> = s
        .rows()
        .into_iter()
        .map(|r| (r[0] as usize, r[1] as usize))
        .collect();
    let j = j.as_slice()?; // the largest array of a tangent: borrowed, never copied
    let (inner, schur) = py
        .detach(|| CondensedCells::new(j, &sizes))
        .map_err(PyValueError::new_err)?;
    Ok((PyCondensedCells { inner }, schur.into_pyarray(py)))
}

/// `block_matvec(k, sizes, u)` -> `K_e u_e` per row-major square block (`sizes (E,)` the block orders).
#[pyfunction]
#[pyo3(name = "block_matvec")]
pub fn py_block_matvec<'py>(
    py: Python<'py>,
    k: PyReadonlyArray1<'py, f64>,
    sizes: PyReadonlyArray1<'py, i64>,
    u: PyReadonlyArray1<'py, f64>,
) -> PyResult<Arr<'py>> {
    let sizes = nonneg(&sizes, "sizes")?;
    let (k, u) = (k.as_slice()?, u.as_slice()?);
    let out = py
        .detach(|| block_matvec(k, &sizes, u))
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

/// Register this file's functions and class on `_native`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_expm, m)?)?;
    m.add_function(wrap_pyfunction!(py_dense_solve, m)?)?;
    m.add_function(wrap_pyfunction!(py_dense_cond, m)?)?;
    m.add_function(wrap_pyfunction!(py_cholesky_factor_len, m)?)?;
    m.add_function(wrap_pyfunction!(py_condense, m)?)?;
    m.add_function(wrap_pyfunction!(py_block_matvec, m)?)?;
    m.add_class::<PyCondensedCells>()?;
    m.add("EXPM_BIT_IDENTICAL_MAX_N", dense::EXPM_BIT_IDENTICAL_MAX_N)?;
    m.add("COND_MAX_N", dense::COND_MAX_N)?;
    Ok(())
}

use numerics::lti::first_crossing as cross;
use numerics::lti::guard_scan::{self as scan_mod, AffineGuards, AffineScanError, History};

fn scan_error(e: AffineScanError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A guard's `(amplitude, sign, after_hit, inherited)`.
type HistoryRow = (f64, i8, bool, i8);

/// `guard_scan(system, state, guards, start, grid, histories, thresholds, evidence, tie, stride)
/// -> (time, crossings, sides, histories)`: the first tied arrivals of `G expm(M (t - start)) y0` on `grid`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(name = "guard_scan", signature = (system, state, guards, start, grid, histories, thresholds, evidence, tie, stride))]
pub fn py_guard_scan<'py>(
    system: PyReadonlyArray2<'py, f64>,
    state: PyReadonlyArray1<'py, f64>,
    guards: PyReadonlyArray2<'py, f64>,
    start: f64,
    grid: PyReadonlyArray1<'py, f64>,
    histories: Vec<HistoryRow>,
    thresholds: Vec<f64>,
    evidence: Vec<Option<f64>>,
    tie: f64,
    stride: usize,
) -> PyResult<(f64, Vec<usize>, Vec<i8>, Vec<HistoryRow>)> {
    let system: Vec<f64> = system.as_array().iter().copied().collect();
    let guards: Vec<f64> = guards.as_array().iter().copied().collect();
    let mut read = AffineGuards::new(system, state.as_array().to_vec(), guards, evidence, start)
        .map_err(scan_error)?;
    let histories: Vec<History> = histories
        .into_iter()
        .map(|(a, s, h, i)| History::checked(a, s, h, i))
        .collect::<Result<_, _>>()
        .map_err(scan_error)?;
    let grid = grid.as_array().to_vec();
    let found = scan_mod::scan_crossings(
        &mut read,
        &scan_mod::Scan {
            grid: &grid,
            histories: &histories,
            thresholds: &thresholds,
            tie,
            stride,
        },
    )
    .map_err(scan_error)?;
    let rows = found
        .histories
        .iter()
        .map(|h| (h.amplitude, h.sign, h.after_hit, h.inherited))
        .collect();
    Ok((found.time, found.crossings, found.sides, rows))
}

/// `first_crossing(system, spectrum, rows, offsets, band, za, span) -> (grid, found)`: the earliest downward zero of
/// `rows @ z + offsets` (past `-band`) on `z(s) = expm(system s) za`, `s` in `[0, span]`; `spectrum` is
/// `(|lambda|, Re lambda)` of `system`; `found` is `(s, rows)` or `None`.
#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(name = "first_crossing")]
pub fn py_first_crossing<'py>(
    system: PyReadonlyArray2<'py, f64>,
    spectrum: (Vec<f64>, Vec<f64>),
    rows: PyReadonlyArray2<'py, f64>,
    offsets: Vec<f64>,
    band: Vec<f64>,
    za: Vec<f64>,
    span: f64,
) -> PyResult<(Vec<f64>, Option<(f64, Vec<usize>)>)> {
    let as_dense = |a: &PyReadonlyArray2<'py, f64>| {
        let v = a.as_array();
        cross::Dense::new(v.nrows(), v.ncols(), v.iter().copied().collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    };
    let (system, rows) = (as_dense(&system)?, as_dense(&rows)?);
    let grid = cross::lattice(&spectrum.0, &spectrum.1, span);
    let found = cross::first_crossing(&cross::Segment {
        system: &system,
        grid: &grid,
        za: &za,
        span,
        rows: &rows,
        offsets: &offsets,
        band: &band,
        propagators: &cross::Propagators::default(),
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((grid, found))
}

/// `span_key(span)`: `span` to `SPAN_KEY_DIGITS` significant digits, the propagator cache's key.
#[pyfunction]
#[pyo3(name = "span_key")]
pub fn py_span_key(span: f64) -> f64 {
    cross::span_key(span)
}

/// Register the exact-trajectory event kernels on `_native`.
pub fn register_lti(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_guard_scan, m)?)?;
    m.add_function(wrap_pyfunction!(py_first_crossing, m)?)?;
    m.add_function(wrap_pyfunction!(py_span_key, m)?)?;
    m.add("SPAN_KEY_DIGITS", cross::SPAN_KEY_DIGITS)?;
    Ok(())
}
