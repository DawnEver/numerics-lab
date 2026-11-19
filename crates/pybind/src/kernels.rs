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
