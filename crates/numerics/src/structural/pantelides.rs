//! Pantelides structural index reduction.
//!
//! Input: the incidence of `n_eq` equations on `n_var` variables, and `derivative_of[j] = Some(k)` when
//! variable `k` is the time derivative of variable `j`. Pantelides (1988) differentiates the minimal
//! structurally singular subsets until every equation can be matched to a HIGHEST-derivative variable;
//! the equations it adds are the hidden constraints an index reduction (e.g. dummy derivatives) needs.

use super::graph::{Bipartite, StructureError};

/// The extended system Pantelides leaves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexReduction {
    /// Every equation's variables, ascending: the original equations first, each differentiated one appended.
    pub equations: Vec<Vec<usize>>,
    /// `equation_derivative[i] = Some(l)` when equation `l` is the time derivative of equation `i`.
    pub equation_derivative: Vec<Option<usize>>,
    /// `variable_derivative[j] = Some(k)` when variable `k` is the time derivative of variable `j`;
    /// the input map, extended by every derivative the reduction introduced.
    pub variable_derivative: Vec<Option<usize>>,
    /// The equation each highest-derivative variable is matched to (`None` for a variable not matched).
    pub assignment: Vec<Option<usize>>,
}

impl IndexReduction {
    /// How many times each of the first `n_original` equations was differentiated.
    #[must_use]
    pub fn differentiations(&self, n_original: usize) -> Vec<usize> {
        (0..n_original)
            .map(|mut i| {
                let mut count = 0;
                while let Some(l) = self.equation_derivative[i] {
                    i = l;
                    count += 1;
                }
                count
            })
            .collect()
    }
}

struct State {
    equations: Vec<Vec<usize>>,
    eq_der: Vec<Option<usize>>,
    var_der: Vec<Option<usize>>,
    assign: Vec<Option<usize>>,
    eq_color: Vec<bool>,
    var_color: Vec<bool>,
}

impl State {
    /// Pantelides' AUGMENTPATH from equation `i` over the undeleted (highest-derivative) variables,
    /// iterative. On failure the colored equations and variables are the structurally singular subset.
    fn augment(&mut self, i: usize) -> bool {
        // (equation, cursor, the variable that led into it)
        let mut stack: Vec<(usize, usize, usize)> = vec![(i, 0, usize::MAX)];
        self.eq_color[i] = true;
        let mut fresh = true;
        while let Some(&(e, k, _)) = stack.last() {
            if fresh {
                fresh = false;
                let free = self.equations[e]
                    .iter()
                    .copied()
                    .find(|&j| self.var_der[j].is_none() && self.assign[j].is_none());
                if let Some(j) = free {
                    self.assign[j] = Some(e);
                    for w in (1..stack.len()).rev() {
                        self.assign[stack[w].2] = Some(stack[w - 1].0);
                    }
                    return true;
                }
            }
            let Some(&j) = self.equations[e].get(k) else {
                stack.pop();
                continue;
            };
            stack.last_mut().expect("non-empty").1 += 1;
            if self.var_der[j].is_some() || self.var_color[j] {
                continue;
            }
            self.var_color[j] = true;
            let next = self.assign[j]
                .expect("an undeleted variable with no equation would have been taken");
            self.eq_color[next] = true;
            stack.push((next, 0, j));
            fresh = true;
        }
        false
    }
}

fn chains(derivative_of: &[Option<usize>]) -> Result<(), StructureError> {
    let n = derivative_of.len();
    let mut parent = vec![None; n];
    for (j, d) in derivative_of.iter().enumerate() {
        if let Some(k) = *d {
            if k >= n {
                return Err(StructureError::IndexOutOfRange {
                    axis: "derivative",
                    index: k,
                    bound: n,
                });
            }
            if k == j || parent[k].replace(j).is_some() {
                return Err(StructureError::DerivativeNotAChain { variable: k });
            }
        }
    }
    for start in 0..n {
        let (mut j, mut steps) = (start, 0);
        while let Some(k) = derivative_of[j] {
            j = k;
            steps += 1;
            if steps > n {
                return Err(StructureError::DerivativeNotAChain { variable: start });
            }
        }
    }
    Ok(())
}

/// Pantelides' algorithm on `system` (rows = equations, columns = variables).
///
/// An equation is differentiated at most `n_eq` times: a pattern that needs more has no matching under any
/// number of differentiations, and is refused rather than looped on.
///
/// # Errors
/// [`StructureError::LengthMismatch`] when `derivative_of` does not have one entry per variable,
/// [`StructureError::DerivativeNotAChain`] / [`StructureError::IndexOutOfRange`] for a bad derivative map,
/// [`StructureError::IndexBoundExceeded`] for a structurally singular DAE.
///
/// # Panics
/// Never: the `expect`s hold by construction of the augmenting-path search.
pub fn pantelides(
    system: &Bipartite,
    derivative_of: &[Option<usize>],
) -> Result<IndexReduction, StructureError> {
    let n_eq = system.n_rows();
    if derivative_of.len() != system.n_cols() {
        return Err(StructureError::LengthMismatch {
            left: derivative_of.len(),
            right: system.n_cols(),
        });
    }
    chains(derivative_of)?;
    let mut s = State {
        equations: (0..n_eq).map(|i| system.row(i).to_vec()).collect(),
        eq_der: vec![None; n_eq],
        var_der: derivative_of.to_vec(),
        assign: vec![None; system.n_cols()],
        eq_color: Vec::new(),
        var_color: Vec::new(),
    };
    for k in 0..n_eq {
        let mut i = k;
        let mut depth = 0;
        loop {
            s.eq_color = vec![false; s.equations.len()];
            s.var_color = vec![false; s.var_der.len()];
            if s.augment(i) {
                break;
            }
            depth += 1;
            if depth > n_eq {
                return Err(StructureError::IndexBoundExceeded {
                    equation: k,
                    bound: n_eq,
                });
            }
            let colored_vars: Vec<usize> = (0..s.var_color.len())
                .filter(|&j| s.var_color[j])
                .collect();
            let colored_eqs: Vec<usize> =
                (0..s.eq_color.len()).filter(|&l| s.eq_color[l]).collect();
            for &j in &colored_vars {
                s.var_der[j] = Some(s.var_der.len());
                s.var_der.push(None);
                s.assign.push(None);
            }
            for &l in &colored_eqs {
                let mut vars = Vec::with_capacity(2 * s.equations[l].len());
                for idx in 0..s.equations[l].len() {
                    let j = s.equations[l][idx];
                    // Every variable of a colored equation is colored or already differentiated.
                    let dj = s.var_der[j].expect("a colored equation's variable has a derivative");
                    vars.push(j);
                    vars.push(dj);
                }
                vars.sort_unstable();
                vars.dedup();
                s.eq_der[l] = Some(s.equations.len());
                s.equations.push(vars);
                s.eq_der.push(None);
            }
            for &j in &colored_vars {
                let e = s.assign[j].expect("a colored variable is matched");
                let dj = s.var_der[j].expect("just differentiated");
                s.assign[dj] = s.eq_der[e];
            }
            i = s.eq_der[i].expect("the failing equation is colored");
        }
    }
    Ok(IndexReduction {
        equations: s.equations,
        equation_derivative: s.eq_der,
        variable_derivative: s.var_der,
        assignment: s.assign,
    })
}
