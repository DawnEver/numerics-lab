//! AMG on the numerics-lab #33 operator class: 3-D anisotropic conduction with a Robin film.
//!
//! 7-point finite volumes on `n^3` cells of the unit cube, 0.41 W/(m K) across vs 138 W/(m K) along
//! x, film 10 W/(m^2 K) on every face. The hierarchy must stay sparse (operator complexity bounded)
//! and preconditioned CG must converge in iterations bounded as N grows. The planted control is the
//! #33 mechanism itself, still selectable: pairwise aggregation smoothed on the FULL operator.

use numerics::linear::amg::{
    aggregate_labels_of, strong_connections_of, Amg, Coarsening, Smoothing,
};
use numerics::sparse::{Coo, Csr};

const COMPLEXITY_BAR: f64 = 2.5; // dimensionless sum(nnz_l) / nnz_0

fn conduction(n: usize) -> Csr {
    let (k_across, k_along, film) = (0.41, 138.0, 10.0);
    #[allow(clippy::cast_precision_loss)]
    let width = 1.0 / n as f64;
    let id = |i: usize, j: usize, k: usize| (i + n * (j + n * k)) as u32;
    let mut coo = Coo::new(n * n * n, n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let me = id(i, j, k);
                let mut faces = 0.0;
                for (axis, conductivity) in [(0, k_along), (1, k_across), (2, k_across)] {
                    let c = [i, j, k][axis];
                    if c == 0 || c == n - 1 {
                        faces += 1.0;
                    }
                    if c + 1 < n {
                        let mut other = [i, j, k];
                        other[axis] += 1;
                        let you = id(other[0], other[1], other[2]);
                        let g = conductivity * width;
                        coo.push(me, me, g).unwrap();
                        coo.push(you, you, g).unwrap();
                        coo.push(me, you, -g).unwrap();
                        coo.push(you, me, -g).unwrap();
                    }
                }
                coo.push(me, me, film * width * width * faces).unwrap();
            }
        }
    }
    coo.to_csr().unwrap()
}

/// Preconditioned CG from zero to `rtol`; the iteration count, or `None` past `max`.
fn pcg(a: &Csr, m: &Amg, rtol: f64, max: usize) -> Option<usize> {
    let n = a.shape().0;
    let b = vec![1.0; n]; // W
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
    let mut x = vec![0.0; n];
    let mut r = b.clone();
    let mut z = m.apply(&r).unwrap();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let stop = rtol * dot(&b, &b).sqrt();
    for iteration in 1..=max {
        let ap = a.matvec(&p).unwrap();
        let alpha = rz / dot(&p, &ap);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if dot(&r, &r).sqrt() <= stop {
            let true_residual: Vec<f64> = a
                .matvec(&x)
                .unwrap()
                .iter()
                .zip(&b)
                .map(|(ax, bi)| bi - ax)
                .collect();
            assert!(dot(&true_residual, &true_residual).sqrt() <= 10.0 * stop);
            return Some(iteration);
        }
        z = m.apply(&r).unwrap();
        let rz_next = dot(&r, &z);
        for i in 0..n {
            p[i] = z[i] + rz_next / rz * p[i];
        }
        rz = rz_next;
    }
    None
}

fn grid_independent(coarsening: Coarsening) {
    let mut iterations = Vec::new();
    for n in [8, 12, 16] {
        let a = conduction(n);
        let m = Amg::new(&a, coarsening, 0.08, Smoothing::default_for(coarsening)).unwrap();
        assert!(
            m.operator_complexity() <= COMPLEXITY_BAR,
            "{coarsening:?} n={n}: {}",
            m.operator_complexity()
        );
        iterations.push(pcg(&a, &m, 1e-8, 300).expect("PCG converges"));
    }
    assert!(iterations.iter().all(|&i| i <= 30), "{iterations:?}");
    assert!(iterations[2] <= iterations[0] + 10, "{iterations:?}");
}

#[test]
fn smoothed_aggregation_is_sparse_and_grid_independent() {
    grid_independent(Coarsening::SmoothedAggregation);
}

#[test]
fn pairwise_is_sparse_and_grid_independent() {
    grid_independent(Coarsening::Pairwise);
}

#[test]
fn planted_control_full_smoothing_fills_the_pairwise_hierarchy() {
    let complexities: Vec<f64> = [8, 12, 16]
        .iter()
        .map(|&n| {
            Amg::new(&conduction(n), Coarsening::Pairwise, 0.0, Smoothing::Full)
                .unwrap()
                .operator_complexity()
        })
        .collect();
    assert!(
        complexities.windows(2).all(|w| w[1] > w[0]),
        "{complexities:?}"
    );
    assert!(complexities[2] > 4.0 * COMPLEXITY_BAR, "{complexities:?}");
}

#[test]
fn planted_control_the_threshold_drops_the_weak_couplings() {
    let a = conduction(4);
    let strong = strong_connections_of(&a, 0.08);
    for i in 0..strong.n_rows {
        for &j in &strong.idx[strong.ptr[i]..strong.ptr[i + 1]] {
            assert_eq!(i.abs_diff(j), 1, "only x neighbors are strong");
        }
    }
    assert_eq!(strong_connections_of(&a, 0.0).nnz(), a.nnz() - a.shape().0);
    let (labels, _) = aggregate_labels_of(&a, Coarsening::SmoothedAggregation, 0.08);
    assert!(labels.iter().enumerate().all(|(i, &l)| labels
        .iter()
        .enumerate()
        .all(|(j, &m)| l != m || i / 4 == j / 4)));
    let (labels, _) = aggregate_labels_of(&a, Coarsening::SmoothedAggregation, 0.0);
    assert!(!labels.iter().enumerate().all(|(i, &l)| labels
        .iter()
        .enumerate()
        .all(|(j, &m)| l != m || i / 4 == j / 4)));
}

#[test]
fn a_bad_threshold_or_name_is_refused() {
    assert!(Amg::new(&conduction(2), Coarsening::Pairwise, 1.0, Smoothing::None).is_err());
    assert!(Coarsening::parse("ruge-stueben").is_err());
    assert!(Smoothing::parse("twice").is_err());
}
