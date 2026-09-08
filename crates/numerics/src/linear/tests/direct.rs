//! The cached-symbolic direct factorisation: it must agree with the one-shot solve, and it must
//! REFUSE a pattern it did not analyse.
//!
//! There is no Python counterpart to mirror here. `scipy.sparse.linalg.spsolve` has no
//! symbolic/numeric split to expose, which is precisely the lever this type exists to take:
//! measured on the production Newton systems, the symbolic half is 34% (`lipo_inductor`) and 46%
//! (`pmasynrm_uv`) of a faer direct solve, and it is IDENTICAL between Newton iterations.

use super::spd_system;
use crate::linear::direct::{solve, DirectFactorization, DirectFailure};
use contracts::sparse::Coo;

/// The same pattern with different values -- what a Newton iteration hands the solver.
fn revalued(base: &contracts::sparse::Csr, scale: f64) -> contracts::sparse::Csr {
    let mut next = base.clone();
    for (k, v) in next.values_mut().iter_mut().enumerate() {
        *v *= scale + (k % 3) as f64 * 0.01;
    }
    next
}

#[test]
fn a_cached_symbolic_reproduces_the_one_shot_solve_on_every_revalued_system() {
    let (a, b) = spd_system(64);
    let cached = DirectFactorization::new(&a).unwrap();
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
    let cached = DirectFactorization::new(&a).unwrap();
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
    let cached = DirectFactorization::new(&a).unwrap();
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
    let cached = DirectFactorization::new(&a).unwrap();
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
