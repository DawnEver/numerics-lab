//! The cached-symbolic direct factorisation: it must agree with the one-shot solve, and it must
//! REFUSE a pattern it did not analyse.
//!
//! There is no Python counterpart to mirror here. `scipy.sparse.linalg.spsolve` has no
//! symbolic/numeric split to expose, which is precisely the lever this type exists to take:
//! measured on the production Newton systems, the symbolic half is 34% (`case_i`) and 46%
//! (`case_n`) of a faer direct solve, and it is IDENTICAL between Newton iterations.

use super::{laplacian_2d, nonsymmetric_system, spd_system};
use crate::linear::direct::{solve, DirectFactorization, DirectFailure, FactorKind};
use crate::sparse::Coo;

/// The same pattern with different values -- what a Newton iteration hands the solver.
fn revalued(base: &crate::sparse::Csr, scale: f64) -> crate::sparse::Csr {
    let mut next = base.clone();
    for (k, v) in next.values_mut().iter_mut().enumerate() {
        *v *= scale + (k % 3) as f64 * 0.01;
    }
    next
}

#[test]
fn a_cached_symbolic_reproduces_the_one_shot_solve_on_every_revalued_system() {
    let (a, b) = spd_system(64);
    let mut cached = DirectFactorization::new(&a).unwrap();
    // Sweep the VALUES across a range a nonlinear iteration would actually traverse; the pattern
    // is what stays fixed. A single re-solve would not distinguish a cached symbolic from a stale
    // NUMERIC one, so the check is that every member agrees, not that one does.
    for scale in [1.0, 0.5, 2.0, 7.5] {
        let system = revalued(&a, scale);
        let expected = solve(&system, &b).unwrap();
        let got = cached.solve(&system, &b).unwrap();
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            assert!(
                (g - e).abs() <= 1e-12 * e.abs().max(1.0),
                "cached symbolic diverged from the one-shot solve at scale {scale}: {g} vs {e}"
            );
        }
    }
}

#[test]
fn the_analysis_survives_a_value_of_zero_in_a_stored_entry() {
    // A structurally present entry going numerically zero must not be mistaken for a pattern
    // change: the CSR still stores it, and re-analysing on every zero crossing would forfeit the
    // whole point of the cache.
    let (a, b) = spd_system(32);
    let mut cached = DirectFactorization::new(&a).unwrap();
    let mut zeroed = a.clone();
    for (k, v) in zeroed.values_mut().iter_mut().enumerate() {
        if k % 5 == 0 && k % 3 != 0 {
            *v = 0.0;
        }
    }
    let expected = solve(&zeroed, &b).unwrap();
    let got = cached.solve(&zeroed, &b).unwrap();
    for (g, e) in got.iter().zip(&expected) {
        assert!((g - e).abs() <= 1e-12 * e.abs().max(1.0));
    }
}

#[test]
fn a_different_sparsity_pattern_is_refused_rather_than_factorised() {
    // The declaration-that-lies case: a cached symbolic applied to a pattern it never analysed
    // returns a plausible vector for the WRONG structure. Planting the change and calling the
    // real guard is the only way to know the refusal fires.
    let (a, b) = spd_system(16);
    let mut cached = DirectFactorization::new(&a).unwrap();
    let mut coo = Coo::new(16, 16);
    for i in 0..16u32 {
        coo.push(i, i, 3.0).unwrap();
        if i + 2 < 16 {
            // An entry the analysed tridiagonal does not have -- the pattern genuinely moved.
            coo.push(i, i + 2, -0.5).unwrap();
        }
    }
    let other = coo.to_csr().unwrap();
    assert_eq!(cached.solve(&other, &b), Err(DirectFailure::PatternChanged));
}

#[test]
fn a_rhs_of_the_wrong_length_is_refused_by_shape() {
    let (a, _) = spd_system(8);
    let mut cached = DirectFactorization::new(&a).unwrap();
    assert_eq!(cached.n(), 8);
    assert!(matches!(
        cached.solve(&a, &[1.0, 2.0]),
        Err(DirectFailure::Shape(_))
    ));
}

#[test]
fn a_non_square_operator_is_refused_at_analysis_time() {
    let mut coo = Coo::new(3, 4);
    coo.push(0, 0, 1.0).unwrap();
    coo.push(1, 1, 1.0).unwrap();
    coo.push(2, 2, 1.0).unwrap();
    assert!(matches!(
        DirectFactorization::new(&coo.to_csr().unwrap()),
        Err(DirectFailure::Shape(_))
    ));
}

fn residual(a: &crate::sparse::Csr, x: &[f64], b: &[f64]) -> f64 {
    let ax = a.matvec(x).unwrap();
    let num: f64 = ax.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum();
    let den: f64 = b.iter().map(|q| q * q).sum();
    (num / den).sqrt()
}

#[test]
fn an_exactly_symmetric_positive_operator_is_factorised_by_cholesky() {
    // MEC's loop matrix is exactly SPD; the check that routes it to LLT is on pattern AND values,
    // so a symmetric system must take it and still solve.
    let (a, b) = laplacian_2d(12);
    let mut f = DirectFactorization::new(&a).unwrap();
    assert_eq!(f.refactor(&a).unwrap(), FactorKind::Cholesky);
    let mut x = b.clone();
    f.solve_in_place(&mut x, 1).unwrap();
    assert!(residual(&a, &x, &b) < 1e-13);
}

#[test]
fn a_nonsymmetric_operator_is_factorised_by_lu() {
    let (a, b) = nonsymmetric_system(40);
    let mut f = DirectFactorization::new(&a).unwrap();
    assert_eq!(f.refactor(&a).unwrap(), FactorKind::Lu);
    let x = f.solve(&a, &b).unwrap();
    assert!(residual(&a, &x, &b) < 1e-13);
}

#[test]
fn a_symmetric_pattern_with_one_unequal_value_is_not_taken_for_symmetric() {
    // Symmetry is VERIFIED per refactor, never inherited from the pattern: one entry moved by an
    // ulp is an unsymmetric operator and must go to LU.
    let (a, b) = spd_system(30);
    let mut f = DirectFactorization::new(&a).unwrap();
    assert_eq!(f.refactor(&a).unwrap(), FactorKind::Cholesky);
    let mut skew = a.clone();
    let k = skew.indptr()[5] as usize; // first stored entry of row 5, the off-diagonal (5,4)
    skew.values_mut()[k] = f64::from_bits(skew.values()[k].to_bits() + 1);
    assert_eq!(f.refactor(&skew).unwrap(), FactorKind::Lu);
    let x = f.solve(&skew, &b).unwrap();
    assert!(residual(&skew, &x, &b) < 1e-13);
}

#[test]
fn a_kernel_holds_one_kind_and_releases_the_one_it_leaves() {
    // A Newton seed's first tangent (A = 0) is exactly symmetric, every later one is not. Holding
    // both kinds was MEASURED at Cholesky + LU side by side (1.0 + 1.47 GiB, n = 810,000 proxy),
    // so the kind being left is released, whichever way the values move.
    let (a, b) = spd_system(30);
    let mut f = DirectFactorization::new(&a).unwrap();
    assert_eq!(f.refactor(&a).unwrap(), FactorKind::Cholesky);
    assert_eq!(f.held(), vec![FactorKind::Cholesky]);
    let mut skew = a.clone();
    let k = skew.indptr()[5] as usize;
    skew.values_mut()[k] *= 1.25;
    assert_eq!(f.refactor(&skew).unwrap(), FactorKind::Lu);
    assert_eq!(f.held(), vec![FactorKind::Lu]);
    let x = f.solve(&skew, &b).unwrap();
    assert!(residual(&skew, &x, &b) < 1e-13);
    // ... and back: the LU is released, the Cholesky re-analysed, and it still solves.
    assert_eq!(f.refactor(&a).unwrap(), FactorKind::Cholesky);
    assert_eq!(f.held(), vec![FactorKind::Cholesky]);
    let x = f.solve(&a, &b).unwrap();
    assert!(residual(&a, &x, &b) < 1e-13);
}

#[test]
fn an_allocation_failure_is_out_of_memory_never_singular() {
    use crate::linear::direct::{faer_refusal, lu_refusal};
    use faer::sparse::linalg::LuError;
    use faer::sparse::FaerError;
    let oom = DirectFailure::OutOfMemory {
        stage: "lu numeric".to_owned(),
    };
    assert_eq!(
        lu_refusal("lu numeric", LuError::Generic(FaerError::OutOfMemory)),
        oom
    );
    assert_eq!(faer_refusal("lu numeric", FaerError::OutOfMemory), oom);
    // The planted control: a genuinely singular structure stays Singular.
    assert!(matches!(
        lu_refusal("lu numeric", LuError::SymbolicSingular { index: 3 }),
        DirectFailure::Singular { .. }
    ));
}

#[test]
fn a_symmetric_indefinite_operator_is_refused_by_cholesky_and_solved_by_lu() {
    let (a, b) = spd_system(20);
    let mut indefinite = a.clone();
    for r in 0..20 {
        for k in indefinite.indptr()[r] as usize..indefinite.indptr()[r + 1] as usize {
            if indefinite.indices()[k] as usize == r && r % 2 == 1 {
                indefinite.values_mut()[k] = -indefinite.values()[k];
            }
        }
    }
    let mut f = DirectFactorization::new(&indefinite).unwrap();
    assert_eq!(f.refactor(&indefinite).unwrap(), FactorKind::Lu);
    let x = f.solve(&indefinite, &b).unwrap();
    assert!(residual(&indefinite, &x, &b) < 1e-12);
}

#[test]
fn one_factorisation_back_substitutes_several_right_hand_sides() {
    let (a, b) = nonsymmetric_system(25);
    let mut f = DirectFactorization::new(&a).unwrap();
    f.refactor(&a).unwrap();
    // Column-major: the first column is `b`, the second `2 b`.
    let mut two: Vec<f64> = b.iter().copied().chain(b.iter().map(|v| v * 2.0)).collect();
    f.solve_in_place(&mut two, 2).unwrap();
    let one = solve(&a, &b).unwrap();
    for i in 0..25 {
        assert!((two[i] - one[i]).abs() <= 1e-13 * one[i].abs().max(1.0));
        assert!((two[25 + i] - 2.0 * one[i]).abs() <= 1e-13 * one[i].abs().max(1.0));
    }
}

#[test]
fn a_badly_scaled_operator_is_solved_to_its_forward_accuracy_not_only_its_residual() {
    // The FEA shape: a stiffness block at ~1e7 with Dirichlet rows pinned to the identity. The
    // residual is small either way; what SuperLU's default equilibration buys is FORWARD accuracy,
    // and a finite-difference inductance downstream differentiates exactly that.
    let n = 200;
    let mut coo = Coo::new(n, n);
    for i in 0..n {
        let big = 1.0e7 * (1.0 + (i % 7) as f64);
        if i % 10 == 0 {
            coo.push(i as u32, i as u32, 1.0).unwrap();
        } else {
            coo.push(i as u32, i as u32, 2.0 * big + 1.0).unwrap();
            if i + 1 < n {
                coo.push(i as u32, (i + 1) as u32, -big).unwrap();
            }
            coo.push(i as u32, (i - 1) as u32, -big).unwrap();
        }
    }
    let a = coo.to_csr().unwrap();
    let truth: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64 * 0.37).sin()).collect();
    let b = a.matvec(&truth).unwrap();
    let x = solve(&a, &b).unwrap();
    let err = x
        .iter()
        .zip(&truth)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-12, "forward error {err:e}");
}

/// The field<->circuit shape: a sparse 2D Laplacian bordered by one DENSE row and column (a
/// winding coupled to every conductor dof) plus a small circuit block.
fn bordered_system(side: usize, dense_every: usize) -> (crate::sparse::Csr, Vec<f64>) {
    let (lap, _) = laplacian_2d(side);
    let n1 = side * side;
    let n = n1 + 2;
    let mut coo = Coo::new(n, n);
    for r in 0..n1 {
        for k in lap.indptr()[r] as usize..lap.indptr()[r + 1] as usize {
            coo.push(r as u32, lap.indices()[k], lap.values()[k])
                .unwrap();
        }
    }
    let (w, c) = (n1 as u32, (n1 + 1) as u32);
    for i in (0..n1).step_by(dense_every) {
        coo.push(i as u32, w, -0.25).unwrap();
        coo.push(w, i as u32, 0.5).unwrap();
    }
    coo.push(w, w, 3.0).unwrap();
    coo.push(w, c, 1.0).unwrap();
    coo.push(c, w, -1.0).unwrap();
    coo.push(c, c, 2.0).unwrap();
    let b = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    (coo.to_csr().unwrap(), b)
}

#[test]
fn a_dense_border_is_eliminated_by_its_schur_complement_not_factorised_as_a_clique() {
    let (a, b) = bordered_system(60, 2); // 3600 interior dofs, a border row of 1800 entries
    let mut f = DirectFactorization::new(&a).unwrap();
    assert_eq!(f.border_size(), 1);
    f.refactor(&a).unwrap();
    let mut x = b.clone();
    f.solve_in_place(&mut x, 1).unwrap();
    assert!(residual(&a, &x, &b) < 1e-13, "{:e}", residual(&a, &x, &b));
}

#[test]
fn an_ordinary_sparse_operator_has_no_border() {
    let (a, _) = laplacian_2d(30);
    assert_eq!(DirectFactorization::new(&a).unwrap().border_size(), 0);
}

#[test]
fn a_bordered_factor_back_substitutes_several_right_hand_sides() {
    let (a, b) = bordered_system(40, 3);
    let mut f = DirectFactorization::new(&a).unwrap();
    assert_eq!(f.border_size(), 1);
    f.refactor(&a).unwrap();
    let n = b.len();
    let mut two: Vec<f64> = b.iter().copied().chain(b.iter().map(|v| -v)).collect();
    f.solve_in_place(&mut two, 2).unwrap();
    assert!(residual(&a, &two[..n], &b) < 1e-13);
    let neg: Vec<f64> = b.iter().map(|v| -v).collect();
    assert!(residual(&a, &two[n..], &neg) < 1e-13);
}

#[test]
fn back_substituting_before_any_factorisation_is_refused() {
    let (a, b) = spd_system(8);
    let f = DirectFactorization::new(&a).unwrap();
    let mut x = b.clone();
    assert_eq!(f.solve_in_place(&mut x, 1), Err(DirectFailure::NotFactored));
}
