//! The complex direct solve, its held factorization, the equilibrated variant and the complex pin.
//!
//! Judged against a dense complex reference (Gaussian elimination with partial pivoting, written
//! here so the reference shares no code with the sparse path) and by the residual it achieves.

use crate::linear::constrained::fixed_values::{apply_dirichlet_triplets, DirichletGroup};
use crate::linear::direct::{solve, DirectFactorization, DirectFailure, FactorKind};
use crate::linear::{solve_direct, solve_equilibrated, LinearSolveError};
use crate::sparse::{c64, Coo, Csr, Scalar};

/// Agreement with the dense reference: both are backward-stable factorizations of systems with
/// `kappa ~ 1e1`, so they agree to `~kappa * eps`; the bar sits two orders above that.
const AGREEMENT_RTOL: f64 = 1e-12;
/// The absolute leg of every comparison, in the unit of the compared entries: without it a
/// relative bar against a near-zero reference passes anything or nothing.
const AGREEMENT_ATOL: f64 = 1e-14;
/// `||A x - b|| <= RESIDUAL_RTOL * ||b|| + RESIDUAL_ATOL`.
const RESIDUAL_RTOL: f64 = 1e-12;
/// The absolute floor of the residual check, in the unit of `b`.
const RESIDUAL_ATOL: f64 = 1e-14;

fn z(re: f64, im: f64) -> c64 {
    c64::new(re, im)
}

/// Complex-symmetric (`A = A^T`, NOT Hermitian) tridiagonal, diagonally dominant.
fn complex_symmetric(n: usize) -> Csr<c64> {
    let mut a = Coo::new(n, n);
    for i in 0..n {
        let k = i as f64;
        a.push(
            i as u32,
            i as u32,
            z(4.0 + (k * 0.37) % 1.0, 1.5 + (k * 0.61) % 1.0),
        )
        .unwrap();
        if i + 1 < n {
            let off = z(-1.0, 0.5 + 0.1 * (k % 3.0));
            a.push(i as u32, i as u32 + 1, off).unwrap();
            a.push(i as u32 + 1, i as u32, off).unwrap();
        }
    }
    a.to_csr().unwrap()
}

/// General non-Hermitian, non-symmetric: different upper/lower couplings plus a far entry.
fn general(n: usize) -> Csr<c64> {
    let mut a = Coo::new(n, n);
    for i in 0..n {
        let k = i as f64;
        a.push(
            i as u32,
            i as u32,
            z(3.0 + (k % 5.0) / 4.0, -2.0 + (k % 3.0) / 2.0),
        )
        .unwrap();
        if i + 1 < n {
            a.push(i as u32, i as u32 + 1, z(-0.5, 0.25)).unwrap();
            a.push(i as u32 + 1, i as u32, z(-1.0, -0.75)).unwrap();
        }
    }
    a.push(0, (n - 1) as u32, z(0.3, 0.9)).unwrap();
    a.to_csr().unwrap()
}

fn rhs(n: usize) -> Vec<c64> {
    (0..n)
        .map(|i| z(((i % 7) as f64 - 3.0) / 2.0, 1.0 + (i % 3) as f64))
        .collect()
}

fn to_dense(a: &Csr<c64>) -> Vec<Vec<c64>> {
    let (n, _) = a.shape();
    let mut d = vec![vec![c64::ZERO; n]; n];
    for (r, row) in d.iter_mut().enumerate() {
        for k in a.indptr()[r] as usize..a.indptr()[r + 1] as usize {
            row[a.indices()[k] as usize] += a.values()[k];
        }
    }
    d
}

/// Dense Gaussian elimination with partial pivoting -- the reference.
fn dense_solve(a: &Csr<c64>, b: &[c64]) -> Vec<c64> {
    let mut m = to_dense(a);
    let mut x = b.to_vec();
    let n = x.len();
    for col in 0..n {
        let p = (col..n)
            .max_by(|&i, &j| m[i][col].modulus().total_cmp(&m[j][col].modulus()))
            .unwrap();
        m.swap(col, p);
        x.swap(col, p);
        for r in col + 1..n {
            let f = m[r][col] / m[col][col];
            for c in col..n {
                let v = m[col][c];
                m[r][c] -= f * v;
            }
            let v = x[col];
            x[r] -= f * v;
        }
    }
    for r in (0..n).rev() {
        let mut acc = x[r];
        for c in r + 1..n {
            acc -= m[r][c] * x[c];
        }
        x[r] = acc / m[r][r];
    }
    x
}

fn norm(v: &[c64]) -> f64 {
    v.iter().map(|z| z.modulus().powi(2)).sum::<f64>().sqrt()
}

fn assert_agrees(got: &[c64], want: &[c64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (*g - *w).modulus();
        assert!(
            d <= AGREEMENT_RTOL * w.modulus() + AGREEMENT_ATOL,
            "{what}[{i}]: {g:?} vs reference {w:?} (|d| = {d:e})"
        );
    }
}

fn assert_satisfies(a: &Csr<c64>, b: &[c64], x: &[c64], what: &str) {
    let ax = a.matvec(x).unwrap();
    let r: Vec<c64> = ax.iter().zip(b).map(|(p, q)| *p - *q).collect();
    assert!(
        norm(&r) <= RESIDUAL_RTOL * norm(b) + RESIDUAL_ATOL,
        "{what}: ||Ax-b|| = {:e} against ||b|| = {:e}",
        norm(&r),
        norm(b)
    );
}

#[test]
fn the_fixtures_are_what_they_claim_to_be() {
    // A complex-symmetric fixture that were Hermitian, or a general one that were symmetric,
    // would make the two tests below one test.
    let s = complex_symmetric(12);
    assert!(s.is_symmetric(0.0));
    assert!(
        to_dense(&s)[0][1] != to_dense(&s)[1][0].conj(),
        "symmetric, not Hermitian"
    );
    assert!(!general(12).is_symmetric(1e-12));
}

#[test]
fn a_complex_symmetric_solve_agrees_with_the_dense_reference_and_satisfies_the_system() {
    let a = complex_symmetric(16);
    let b = rhs(16);
    let x = solve(&a, &b).unwrap();
    assert_agrees(&x, &dense_solve(&a, &b), "complex symmetric");
    assert_satisfies(&a, &b, &x, "complex symmetric");
}

#[test]
fn a_general_non_hermitian_solve_agrees_with_the_dense_reference_and_satisfies_the_system() {
    let a = general(16);
    let b = rhs(16);
    let x = solve(&a, &b).unwrap();
    assert_agrees(&x, &dense_solve(&a, &b), "general");
    assert_satisfies(&a, &b, &x, "general");
}

#[test]
fn a_complex_symmetric_operator_is_factorized_by_lu_never_by_an_adjoint_cholesky() {
    // Its values ARE exactly symmetric, which on f64 routes to L L^T. Over c64 that factorization
    // would be L L^H -- the factor of a DIFFERENT matrix -- so the kind must stay LU.
    let a = complex_symmetric(10);
    let mut f = DirectFactorization::new(&a).unwrap();
    assert_eq!(f.refactor(&a).unwrap(), FactorKind::Lu);
}

#[test]
fn a_held_factorization_answers_every_revalued_system_and_several_right_hand_sides() {
    let a = general(20);
    let mut f = DirectFactorization::new(&a).unwrap();
    for s in [1.0, 0.5, 3.0] {
        let mut next = a.clone();
        for (k, v) in next.values_mut().iter_mut().enumerate() {
            *v *= z(s, 0.1 * (k % 4) as f64);
        }
        f.refactor(&next).unwrap();
        let (b1, b2) = (
            rhs(20),
            rhs(20).iter().map(|v| *v * z(0.0, 1.0)).collect::<Vec<_>>(),
        );
        let mut both = [b1.clone(), b2.clone()].concat();
        f.solve_in_place(&mut both, 2).unwrap();
        assert_agrees(&both[..20], &dense_solve(&next, &b1), "held, column 0");
        assert_agrees(&both[20..], &dense_solve(&next, &b2), "held, column 1");
        assert_agrees(&f.solve(&next, &b1).unwrap(), &both[..20], "held solve");
    }
}

#[test]
fn a_held_factorization_refuses_a_changed_pattern_and_a_solve_before_a_factor() {
    let a = general(8);
    let mut f = DirectFactorization::new(&a).unwrap();
    let mut b = rhs(8);
    assert_eq!(f.solve_in_place(&mut b, 1), Err(DirectFailure::NotFactored));
    assert_eq!(
        f.refactor(&complex_symmetric(8)).unwrap_err(),
        DirectFailure::PatternChanged
    );
}

#[test]
fn a_dense_border_row_is_eliminated_over_the_complex_field_too() {
    // One row/column coupled to every index takes the Schur-border path.
    let n = 400;
    let mut coo = Coo::new(n, n);
    for i in 0..n {
        coo.push(i as u32, i as u32, z(4.0, 1.0 + (i % 3) as f64))
            .unwrap();
        if i > 0 {
            coo.push(0, i as u32, z(0.01, -0.02)).unwrap();
            coo.push(i as u32, 0, z(-0.03, 0.01)).unwrap();
        }
    }
    let a = coo.to_csr().unwrap();
    let f = DirectFactorization::new(&a).unwrap();
    assert_eq!(f.border_size(), 1, "the fixture must exercise the border");
    let b = rhs(n);
    let x = solve(&a, &b).unwrap();
    assert_agrees(&x, &dense_solve(&a, &b), "bordered");
}

#[test]
fn a_structurally_singular_complex_matrix_is_refused_by_name() {
    let a = Coo::from_triplets(2, 2, vec![0], vec![0], vec![z(1.0, 1.0)])
        .unwrap()
        .to_csr()
        .unwrap();
    let b = vec![z(1.0, 0.0); 2];
    assert!(matches!(
        solve(&a, &b).unwrap_err(),
        DirectFailure::Singular { .. }
    ));
    assert!(matches!(
        solve_direct(&a, &b).unwrap_err(),
        LinearSolveError::Direct(DirectFailure::Singular { .. })
    ));
}

#[test]
fn a_numerically_singular_complex_matrix_is_refused_by_the_guard() {
    // Two identical rows: the factorization may or may not refuse, the guard must.
    let r = [z(2.0, 1.0), z(1.0, -1.0), z(1.0, 0.5)];
    let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let vals = vec![
        r[0],
        r[1],
        r[2],
        r[0],
        r[1],
        r[2],
        z(1.0, 0.0),
        z(1.0, 1.0),
        z(3.0, 0.0),
    ];
    let a = Coo::from_triplets(3, 3, rows, cols, vals)
        .unwrap()
        .to_csr()
        .unwrap();
    let b = vec![z(1.0, 0.0), z(2.0, 0.0), z(1.0, 0.0)];
    let err = solve_direct(&a, &b).unwrap_err();
    assert!(
        matches!(
            err,
            LinearSolveError::Direct(DirectFailure::Singular { .. })
                | LinearSolveError::DirectResidual { .. }
                | LinearSolveError::NonFinite { .. }
        ),
        "{err:?}"
    );
}

/// `D0 M D0 x = b` with `D0` spanning `1e-153..1e153` and `M` well-conditioned: every entry of the
/// operator is finite, but `||b||^2` overflows, so no residual measured in the operator's own units
/// can be computed. `(x, b)` are built from `y = D0 x` so the exact `x` is known.
fn badly_scaled(n: usize) -> (Csr<c64>, Vec<c64>, Vec<c64>) {
    let m = complex_symmetric(n);
    let d: Vec<f64> = (0..n)
        .map(|i| 10f64.powi(153 * ((i % 3) as i32 - 1)))
        .collect();
    let mut a = m.clone();
    for r in 0..n {
        for k in a.indptr()[r] as usize..a.indptr()[r + 1] as usize {
            let c = a.indices()[k] as usize;
            a.values_mut()[k] = m.values()[k].scale(d[r]).scale(d[c]);
        }
    }
    let y: Vec<c64> = rhs(n).iter().map(|v| v.scale(1e3)).collect();
    let x: Vec<c64> = y.iter().zip(&d).map(|(v, s)| v.scale(1.0 / s)).collect();
    let my = m.matvec(&y).unwrap();
    let b: Vec<c64> = my.iter().zip(&d).map(|(v, s)| v.scale(*s)).collect();
    (a, b, x)
}

#[test]
fn the_equilibrated_solve_recovers_a_badly_scaled_system() {
    let (a, b, x) = badly_scaled(9);
    assert!(a.values().iter().all(|v| Scalar::is_finite(*v)));
    let got = solve_equilibrated(&a, &b).unwrap();
    for (i, (g, w)) in got.iter().zip(&x).enumerate() {
        let d = (*g - *w).modulus();
        assert!(d <= AGREEMENT_RTOL * w.modulus(), "x[{i}]: {g:?} vs {w:?}");
    }
}

#[test]
fn planted_control_the_plain_guard_refuses_the_badly_scaled_system() {
    // The control the equilibration is measured against: on the SAME system the unscaled guard
    // cannot measure a residual (`inf / inf`), and an unmeasurable residual is a refusal, never a
    // pass. If this ever succeeds, the fixture stopped being badly scaled and the test above
    // stopped proving anything.
    let (a, b, _) = badly_scaled(9);
    let err = solve_direct(&a, &b).unwrap_err();
    assert!(
        matches!(err, LinearSolveError::DirectResidual { residual, .. } if residual.is_nan()),
        "{err:?}"
    );
}

#[test]
fn the_equilibrated_solve_passes_a_row_without_a_diagonal_through_at_unit_scale() {
    // [[0, 1], [1, 0]] x = [2, 3] -> x = [3, 2]: no diagonal to take out, no division by zero.
    let a = Coo::from_triplets(2, 2, vec![0, 1], vec![1, 0], vec![z(0.0, 1.0), z(0.0, 1.0)])
        .unwrap()
        .to_csr()
        .unwrap();
    let got = solve_equilibrated(&a, &[z(0.0, 2.0), z(0.0, 3.0)]).unwrap();
    assert_agrees(&got, &[z(3.0, 0.0), z(2.0, 0.0)], "off-diagonal only");
}

#[test]
fn the_equilibrated_solve_refuses_what_the_guard_refuses() {
    let a = Coo::from_triplets(2, 2, vec![0], vec![0], vec![z(1.0, 1.0)])
        .unwrap()
        .to_csr()
        .unwrap();
    assert!(solve_equilibrated(&a, &[z(1.0, 0.0); 2]).is_err());
}

#[test]
fn a_complex_pin_keeps_the_imaginary_part_and_the_pinned_system_solves() {
    let a = general(6);
    let (mut rows, mut cols, mut vals) = (vec![], vec![], vec![]);
    for r in 0..6 {
        for k in a.indptr()[r] as usize..a.indptr()[r + 1] as usize {
            rows.push(r as u32);
            cols.push(a.indices()[k]);
            vals.push(a.values()[k]);
        }
    }
    let mut b = rhs(6);
    let (r, c, v) = apply_dirichlet_triplets(
        rows,
        cols,
        vals,
        &mut b,
        &[
            DirichletGroup {
                value: z(1.5, -2.0),
                indices: vec![0],
            },
            DirichletGroup {
                value: z(0.0, 0.5),
                indices: vec![5, 5],
            },
        ],
        z(0.25, 0.0),
    );
    assert_eq!(b[0], z(1.75, -2.0), "the pinned value lost a component");
    assert_eq!(b[5], z(0.25, 0.5));
    let pinned = Coo::from_triplets(6, 6, r, c, v).unwrap().to_csr().unwrap();
    let d = to_dense(&pinned);
    assert_eq!(d[0], [vec![c64::ONE], vec![c64::ZERO; 5]].concat());
    let x = solve_direct(&pinned, &b).unwrap();
    assert_eq!(x[0], z(1.75, -2.0));
    assert_eq!(x[5], z(0.25, 0.5));
    assert_agrees(&x, &dense_solve(&pinned, &b), "pinned");
}
