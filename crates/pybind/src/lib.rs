//! `numerics_lab._native` — the Rust engine's Python surface.
//!
//! Thin glue only: NumPy arrays in, NumPy arrays out, calling [`numerics`]. No numerics live
//! here. The sparse operator arrives as COO triplets because that is the shape
//! `scipy.sparse.coo_matrix` hands out on the Python side, and `numerics::sparse` owns the
//! COO -> CSR conversion.

use numerics::sparse::{Coo, Csr};
use numerics::linear::direct::DirectFactorization;
use numerics::linear::{check_direct_residual, requires_symmetric, solve_linear_system};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Narrow one `i64` index to the `u32` [`Coo`] wants, REFUSING anything that would not survive.
///
/// `as u32` is a wrapping cast, and that is the whole reason this function exists. Measured
/// 2026-09-05 against the shadowed build: a row index of `2**32 + 1` handed to
/// [`py_solve_linear_system`] on a declared 3x3 wrapped to row 1 and the solve RETURNED
/// `[1.0, 0.5714285714285714, 3.0]` -- the exact solution of a DIFFERENT matrix, with the stray
/// value summed into `A[1,1]`. `scipy.sparse.coo_matrix` refuses the same triplet outright
/// ("axis 0 index 4294967297 exceeds matrix dimension 3"). A plausible-looking wrong answer
/// returned as a success is what `.claude/rules/invariant.md` forbids, so the narrowing is
/// checked here at the marshalling boundary rather than trusted.
///
/// Negative ids wrapped too, and the resulting `4294967295` is what the downstream bound check
/// reported -- so its message named a number the caller never passed. Refusing here keeps the
/// CALLER's number in the message.
fn narrow_index(value: i64, bound: usize, axis: &str) -> PyResult<u32> {
    if value < 0 || value as u128 >= bound as u128 {
        return Err(PyValueError::new_err(format!(
            "{axis} index {value} is outside [0, {bound}); a sparse index is narrowed to u32 on \
             the wire, so an out-of-range value would WRAP into a valid-looking index and solve \
             a different matrix"
        )));
    }
    Ok(value as u32)
}

/// `solve_linear_system(n_rows, n_cols, rows, cols, values, rhs, solver=None, preconditioner=None, x0=None)`
///
/// Solve `A @ x = rhs` where `A` is given as COO triplets. `solver` is `None` (the default
/// `direct` sparse LU) or a Krylov name; `preconditioner` is `jacobi` / `ilu` / `ssor`;
/// `x0` is an optional Krylov warm start. Returns `x` as `f64[n_rows]`.
///
/// Errors from the guard (a non-finite solution, a direct-solve residual that does not
/// verify, an unknown backend name) come back as `ValueError` — never a wrong `x` returned
/// as a success.
#[pyfunction]
#[pyo3(
    name = "solve_linear_system",
    signature = (n_rows, n_cols, rows, cols, values, rhs, solver=None, preconditioner=None, x0=None)
)]
#[allow(clippy::too_many_arguments)]
pub fn py_solve_linear_system<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
    rhs: PyReadonlyArray1<'py, f64>,
    solver: Option<&str>,
    preconditioner: Option<&str>,
    x0: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r: Vec<u32> = rows
        .as_array()
        .iter()
        .map(|&i| narrow_index(i, n_rows, "row"))
        .collect::<PyResult<_>>()?;
    let c: Vec<u32> = cols
        .as_array()
        .iter()
        .map(|&i| narrow_index(i, n_cols, "column"))
        .collect::<PyResult<_>>()?;
    let v: Vec<f64> = values.as_array().to_vec();
    let coo = Coo::from_triplets(n_rows, n_cols, r, c, v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let csr = coo
        .to_csr()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let b = rhs.as_array().to_vec();
    let warm = x0.map(|a| a.as_array().to_vec());
    let x = solve_linear_system(&csr, &b, solver, preconditioner, warm.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(x.into_pyarray(py))
}

/// `DirectFactorization(n_rows, n_cols, indptr, indices, values)` — the CACHED-symbolic direct LU.
///
/// [`py_solve_linear_system`] pays colamd and the symbolic factorisation on every call, which is
/// 34% / 46% of a faer solve at the two production sizes and is IDENTICAL between Newton
/// iterations. This class is that half held: the constructor analyses the pattern once and
/// [`PyDirectFactorization::solve`] pays only the numeric factorisation and the triangular solve.
///
/// It exists because the comparison that decides whether faer replaces SuperLU is only meaningful
/// in this shape — a wiring would hold the factorisation across the loop, so a benchmark that
/// re-analysed per call would be measuring an implementation nobody proposes.
///
/// The pattern is CHECKED, never assumed: the constructor refuses a non-canonical CSR (columns not
/// ascending / not unique, which `Coo::to_csr` would silently reorder into a different value
/// order), and `numerics`'s own `DirectFactorization::solve` refuses a changed pattern.
#[pyclass(name = "DirectFactorization")]
pub struct PyDirectFactorization {
    /// The analysed operator; only its VALUES move between solves.
    csr: Csr,
    factorization: DirectFactorization,
}

#[pymethods]
impl PyDirectFactorization {
    #[new]
    #[pyo3(signature = (n_rows, n_cols, indptr, indices, values))]
    fn new<'py>(
        n_rows: usize,
        n_cols: usize,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i64>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Self> {
        // `indptr` is validated BEFORE it is narrowed, because the row-expansion loop below
        // turns a bad offset straight into an allocation. Measured 2026-09-05: `indptr =
        // [0, -1, 2, 3]` passes both of the checks that used to be here (it is n_rows+1 long
        // and ends at nnz), then `-1 as usize` makes the range `0..18446744073709551615` and
        // the process ABORTS -- "memory allocation of 51539607552 bytes failed". That is not
        // an exception a caller can catch; it takes the interpreter with it.
        let raw_ptr = indptr.as_array();
        let mut ptr: Vec<usize> = Vec::with_capacity(raw_ptr.len());
        let mut previous = 0i64;
        for (row, &offset) in raw_ptr.iter().enumerate() {
            if offset < 0 || offset < previous {
                return Err(PyValueError::new_err(format!(
                    "indptr must be non-negative and non-descending; entry {row} is {offset} \
                     after {previous}. A descending or negative offset becomes an enormous \
                     unsigned range here and aborts the process rather than raising"
                )));
            }
            previous = offset;
            ptr.push(offset as usize);
        }
        let idx: Vec<u32> = indices
            .as_array()
            .iter()
            .map(|&i| narrow_index(i, n_cols, "column"))
            .collect::<PyResult<_>>()?;
        let val: Vec<f64> = values.as_array().to_vec();
        if ptr.len() != n_rows + 1 || ptr.last().copied().unwrap_or(0) != idx.len() {
            return Err(PyValueError::new_err(
                "indptr must be n_rows+1 long and end at nnz",
            ));
        }
        if val.len() != idx.len() {
            return Err(PyValueError::new_err(format!(
                "values and indices must agree; got {} values for {} indices",
                val.len(),
                idx.len()
            )));
        }
        let mut rows: Vec<u32> = Vec::with_capacity(idx.len());
        for r in 0..n_rows {
            for _ in ptr[r]..ptr[r + 1] {
                rows.push(r as u32);
            }
        }
        let coo = Coo::from_triplets(n_rows, n_cols, rows, idx.clone(), val)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let csr = coo
            .to_csr()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if csr.indices() != idx.as_slice() {
            return Err(PyValueError::new_err(
                "the CSR offered is not canonical (columns must be ascending and unique); \
                 call sort_indices()/sum_duplicates() before handing it over, because a \
                 reordering here would silently misplace every later value update",
            ));
        }
        let factorization =
            DirectFactorization::new(&csr).map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        Ok(Self { csr, factorization })
    }

    /// `solve(values, rhs)` — a new value vector on the SAME pattern, solved against the cache.
    ///
    /// The residual guard is applied here exactly as `solve_linear_system` applies it to the
    /// one-shot path: a finite-but-meaningless solution is refused rather than returned.
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
        rhs: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let new_values = values.as_array();
        let slot = self.csr.values_mut();
        if new_values.len() != slot.len() {
            return Err(PyValueError::new_err(
                "values must carry the analysed pattern's nnz",
            ));
        }
        for (dst, &src) in slot.iter_mut().zip(new_values.iter()) {
            *dst = src;
        }
        let b = rhs.as_array().to_vec();
        let x = self
            .factorization
            .solve(&self.csr, &b)
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        check_direct_residual(&self.csr, &b, &x)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(x.into_pyarray(py))
    }

    /// `refactor(values) -> "lu" | "cholesky"` — numerically factorise a new value vector on the
    /// SAME pattern into the held storage, and name the factorisation it took.
    ///
    /// The split from [`Self::back_substitute`] is what lets one factor answer several
    /// right-hand sides (the held-factor Newton, the bordered field<->circuit tangent). No
    /// residual guard runs here, because there is no right-hand side yet: the caller owes it,
    /// exactly as it owes it to SciPy's `splu`.
    fn refactor<'py>(&mut self, values: PyReadonlyArray1<'py, f64>) -> PyResult<&'static str> {
        let new_values = values.as_array();
        let slot = self.csr.values_mut();
        if new_values.len() != slot.len() {
            return Err(PyValueError::new_err(
                "values must carry the analysed pattern's nnz",
            ));
        }
        for (dst, &src) in slot.iter_mut().zip(new_values.iter()) {
            *dst = src;
        }
        let kind = self
            .factorization
            .refactor(&self.csr)
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        Ok(kind.as_str())
    }

    /// `back_substitute(rhs, nrhs)` — solve against the factor [`Self::refactor`] produced.
    ///
    /// `rhs` is `nrhs` right-hand sides flattened COLUMN-MAJOR (`b.ravel(order="F")`); the result
    /// has the same layout. Refused when no factor is held.
    fn back_substitute<'py>(
        &self,
        py: Python<'py>,
        rhs: PyReadonlyArray1<'py, f64>,
        nrhs: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut x = rhs.as_array().to_vec();
        self.factorization
            .solve_in_place(&mut x, nrhs)
            .map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
        Ok(x.into_pyarray(py))
    }

    /// The factorisation the last refactor took (`"lu"` / `"cholesky"`), or `None`.
    #[getter]
    fn kind(&self) -> Option<&'static str> {
        self.factorization.kind().map(|k| k.as_str())
    }

    /// The order of the analysed system.
    #[getter]
    fn n(&self) -> usize {
        self.factorization.n()
    }
}

/// `requires_symmetric(name)` — does the named backend assume a symmetric/SPD operator?
///
/// An unknown name is `False` rather than an error: this is a QUERY about a tag, and
/// `solve_linear_system` is where an unknown name is refused.
#[pyfunction]
#[pyo3(name = "requires_symmetric")]
pub fn py_requires_symmetric(name: &str) -> bool {
    requires_symmetric(name)
}

/// The extension module `numerics_lab._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_solve_linear_system, m)?)?;
    m.add_function(wrap_pyfunction!(py_requires_symmetric, m)?)?;
    m.add_class::<PyDirectFactorization>()?;
    Ok(())
}
