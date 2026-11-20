//! A small row-major dense matrix of any shape and its products, each summed in a fixed order so a transcribed
//! march reads the same bits every time. Moved out of `lti::first_crossing` on 2026-11-19.

/// Two shapes that do not compose, named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeError(pub String);

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ShapeError {}

/// A dense row-major matrix of any shape (an empty one keeps its column count).
#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Dense {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, ShapeError> {
        if data.len() != rows * cols {
            return Err(ShapeError(format!(
                "{rows} x {cols} block with {} entries",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[must_use]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// `self @ rhs`, accumulating in `i, k, j` order.
    /// # Errors
    /// The shapes do not compose.
    pub fn mul(&self, rhs: &Dense) -> Result<Dense, ShapeError> {
        if self.cols != rhs.rows {
            return Err(ShapeError(format!(
                "{} x {} @ {} x {}",
                self.rows, self.cols, rhs.rows, rhs.cols
            )));
        }
        let mut out = Dense::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[i * self.cols + k];
                for j in 0..rhs.cols {
                    out.data[i * rhs.cols + j] += a * rhs.data[k * rhs.cols + j];
                }
            }
        }
        Ok(out)
    }

    /// `self @ x`.
    #[must_use]
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        (0..self.rows).map(|i| dot(self.row(i), x)).collect()
    }
}

/// `a . b`, summed left to right.
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `v @ m` of a row vector.
#[must_use]
pub fn vecmat(v: &[f64], m: &Dense) -> Vec<f64> {
    let mut out = vec![0.0; m.cols];
    for (k, &a) in v.iter().enumerate() {
        for (j, o) in out.iter_mut().enumerate() {
            *o += a * m.data[k * m.cols + j];
        }
    }
    out
}
