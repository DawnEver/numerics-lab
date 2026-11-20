//! Block static condensation's own identities, carried over from motronics-studio's
//! `rust/crates/hamilton_sdm/tests/coupled.rs` (a74923ef38): the condensed step is the dense solve
//! (which also pins the LU's row permutation), and `block_matvec` is each block's own product.

use numerics::linear::condense::{block_matvec, CondensedCells};

/// A tiny deterministic generator, so the test needs no crate.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }
}

#[test]
fn the_condensed_step_is_the_dense_solve() {
    let mut rng = Lcg(7);
    let sizes = [(3usize, 4usize), (5, 9), (2, 1)];
    let mut blocks = Vec::new();
    let mut dense = Vec::new();
    for &(k, nb) in &sizes {
        let s = k + nb;
        let m = rng.vec(s * s);
        // nonsymmetric, well conditioned, and its bubble block NEEDS pivoting (a small leading entry)
        let mut j: Vec<f64> = (0..s * s)
            .map(|x| m[x] + if x % (s + 1) == 0 { 3.0 } else { 0.0 })
            .collect();
        j[k * s + k] = 1e-3;
        blocks.extend_from_slice(&j);
        dense.push(j);
    }
    let (cells, schur) = CondensedCells::new(&blocks, &sizes).unwrap();
    let nb_total: usize = sizes.iter().map(|s| s.1).sum();
    let rhs_b = rng.vec(nb_total);
    let (y, corr) = cells.eliminate(&rhs_b);
    let dv = rng.vec(sizes.iter().map(|s| s.0).sum());
    let x_b = cells.recover(&y, &dv);
    let (mut at_k, mut at_b, mut at_s) = (0, 0, 0);
    for (e, &(k, nb)) in sizes.iter().enumerate() {
        let s = k + nb;
        let j = |r: usize, c: usize| dense[e][r * s + c];
        // J_bb x_b + J_bn dv = rhs_b, and the Schur / correction consistent with it
        for r in 0..nb {
            let mut lhs = 0.0;
            for c in 0..nb {
                lhs += j(k + r, k + c) * x_b[at_b + c];
            }
            for c in 0..k {
                lhs += j(k + r, c) * dv[at_k + c];
            }
            assert!(
                (lhs - rhs_b[at_b + r]).abs() < 1e-11,
                "cell {e} row {r}: {lhs} vs {}",
                rhs_b[at_b + r]
            );
        }
        for r in 0..k {
            let mut jy = 0.0;
            for c in 0..nb {
                jy += j(r, k + c) * y[at_b + c];
            }
            assert!((jy - corr[at_k + r]).abs() < 1e-11);
            // S dv = J_nn dv + J_nb x_b - J_nb y  (x_b = y - Z dv)
            let mut s_dv = 0.0;
            let mut want = 0.0;
            for c in 0..k {
                s_dv += schur[at_s + r * k + c] * dv[at_k + c];
                want += j(r, c) * dv[at_k + c];
            }
            for c in 0..nb {
                want += j(r, k + c) * (x_b[at_b + c] - y[at_b + c]);
            }
            assert!(
                (s_dv - want).abs() < 1e-11,
                "cell {e} schur row {r}: {s_dv} vs {want}"
            );
        }
        at_k += k;
        at_b += nb;
        at_s += k * k;
    }
}

#[test]
fn every_block_multiplies_only_its_own_slice() {
    let mut rng = Lcg(11);
    let sizes = [3usize, 1, 4];
    let blocks = rng.vec(sizes.iter().map(|m| m * m).sum());
    let u = rng.vec(sizes.iter().sum());
    let got = block_matvec(&blocks, &sizes, &u).unwrap();
    let (mut ob, mut ov) = (0, 0);
    for &m in &sizes {
        for r in 0..m {
            let want: f64 = (0..m).map(|c| blocks[ob + r * m + c] * u[ov + c]).sum();
            assert!((got[ov + r] - want).abs() < 1e-14, "row {r} of a {m}-block");
        }
        ob += m * m;
        ov += m;
    }
    // planted: a vector one entry short is refused, never read past
    assert!(block_matvec(&blocks, &sizes, &u[1..]).is_err());
}
