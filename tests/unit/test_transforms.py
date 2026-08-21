"""Coordinate transforms (Park/Clarke) -- the physics-free math home (round-2 T1).

Migrated from hamilton (postprocessing/dq_inductance, coupling/foc) when the transforms became
the shared home both hamilton and laplace import: the round-trip identities, the amplitude
invariance, the N-phase generalization, and the balanced-set-to-DC property are the same
contract in the new home.
"""

from __future__ import annotations

import numpy as np
import pytest

from motronics.gauss.transforms import clarke_forward, park_forward, park_inverse, phase_L_to_dq_L


def test_park_roundtrip_three_phase_identity() -> None:
    """Inverse then forward Park recovers the original phase quantities (3-phase).

    For 3 phases the (d, q, zero) triple spans the whole phase space, so any 3-phase set
    round-trips exactly (inverse o forward == identity).
    """
    rng = np.random.default_rng(0)
    n_theta = 64
    theta_e = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    x_abc = rng.standard_normal((3, n_theta))
    d, q, zero = park_forward(x_abc, theta_e)
    x_back = park_inverse(d, q, zero, theta_e, n_phase=3)
    np.testing.assert_allclose(x_back, x_abc, atol=1e-12)


def test_park_forward_inverse_identity_five_phase() -> None:
    """Park generalizes to N phases: forward o inverse == identity on the (d, q, zero) subspace."""
    rng = np.random.default_rng(1)
    n_phase, n_theta = 5, 40
    theta_e = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    d0, q0, z0 = rng.standard_normal((3, n_theta))
    x = park_inverse(d0, q0, z0, theta_e, n_phase=n_phase)
    d, q, zero = park_forward(x, theta_e)
    np.testing.assert_allclose(d, d0, atol=1e-12)
    np.testing.assert_allclose(q, q0, atol=1e-12)
    np.testing.assert_allclose(zero, z0, atol=1e-12)


def test_park_balanced_positive_sequence_is_dc() -> None:
    """A balanced positive-sequence set aligned with the rotor maps to a constant d, zero q."""
    n_theta = 120
    theta_e = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    amp = 2.5
    x_abc = np.array([amp * np.cos(theta_e - k * 2 * np.pi / 3) for k in range(3)])
    d, q, zero = park_forward(x_abc, theta_e)
    np.testing.assert_allclose(d, amp, atol=1e-12)
    np.testing.assert_allclose(q, 0.0, atol=1e-12)
    np.testing.assert_allclose(zero, 0.0, atol=1e-12)


def test_clarke_amplitude_invariant() -> None:
    """Clarke maps a balanced 3-phase set to an (alpha, beta) vector of the phase amplitude."""
    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    peak = 5.0
    a = peak * np.cos(theta)
    b = peak * np.cos(theta - 2.0 * np.pi / 3.0)
    c = peak * np.cos(theta - 4.0 * np.pi / 3.0)
    alpha, beta = clarke_forward(np.array([a, b, c]))
    assert np.allclose(np.hypot(alpha, beta), peak, atol=1e-12)


def test_park_inverse_vector_commands_with_scalar_angle_broadcast_to_samples() -> None:
    """SR-024: vector (d, q, zero) at ONE angle is n_theta SAMPLES, not n_phase phases --
    the promised 'theta_e broadcastable to d' shape is ``(n_phase, n)``, never a ``(3,)``
    array that conflates samples with phases (and never a broadcast error).
    """
    rng = np.random.default_rng(3)
    n = 7
    d, q, zero = rng.standard_normal((3, n))
    theta = 0.7
    out = park_inverse(d, q, zero, theta_e=theta)
    assert out.shape == (3, n)
    axes = 2.0 * np.pi * np.arange(3, dtype=float) / 3
    expect = d * np.cos(theta - axes[:, None]) - q * np.sin(theta - axes[:, None]) + zero
    np.testing.assert_allclose(out, expect, atol=1e-12)
    # And the forward map recovers the commands: the (3, n) output is per-sample phases.
    d_back, q_back, zero_back = park_forward(out, np.array(theta))
    np.testing.assert_allclose(d_back, d, atol=1e-12)
    np.testing.assert_allclose(q_back, q, atol=1e-12)
    np.testing.assert_allclose(zero_back, zero, atol=1e-12)


def test_park_inverse_scalar_commands_with_vector_angle_broadcast() -> None:
    """Scalar (d, q, zero) over a vector theta_e: one constant command modulated at n angles
    -- the FOC signal-chain shape (constant commands at a rotating theta_e).
    """
    n_theta = 16
    theta_e = np.linspace(0.0, 2.0 * np.pi, n_theta)
    d, q, zero = 1.5, -2.0, 0.25
    out = park_inverse(d, q, zero, theta_e)
    assert out.shape == (3, n_theta)
    axes = 2.0 * np.pi * np.arange(3, dtype=float) / 3
    expect = d * np.cos(theta_e[None, :] - axes[:, None]) - q * np.sin(theta_e[None, :] - axes[:, None]) + zero
    np.testing.assert_allclose(out, expect, atol=1e-12)


def test_park_inverse_refuses_incompatible_command_shapes() -> None:
    """Commands that cannot broadcast to one sample shape are refused -- never silently
    conflated (a (3,) d with a (4,) q is not three phases by accident).
    """
    with pytest.raises(ValueError, match='broadcast'):
        park_inverse(np.zeros(3), np.zeros(4), 0.0, np.linspace(0.0, 1.0, 5))


def test_park_inverse_refuses_a_2d_theta_grid() -> None:
    """SR-031: a 2-D theta_e must be refused loudly -- the sample axis is 1-D by contract,
    and an (n_phase, n) phase grid is the n_phase parameter's job, never an input grid. The
    dangerous case: a (3, n) angle array whose leading axis EQUALS n_phase would otherwise be
    silently consumed as the phase axis (wrong numbers, no error); other 2-D shapes would
    surface as a raw numpy broadcast failure.
    """
    with pytest.raises(ValueError, match='theta_e'):
        park_inverse(1.0, 0.0, 0.0, np.zeros((3, 4)))
    with pytest.raises(ValueError, match='theta_e'):
        park_inverse(np.zeros((3, 4)), np.zeros((3, 4)), np.zeros((3, 4)), np.zeros((3, 4)))
    with pytest.raises(ValueError, match='theta_e'):
        park_inverse(np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 4)))


def test_park_and_clarke_agree_on_a_balanced_set_at_theta_zero() -> None:
    """The two transforms are one convention: at theta_e = 0 the d-axis equals alpha."""
    rng = np.random.default_rng(2)
    x_abc = rng.standard_normal(3)
    d, q, _zero = park_forward(x_abc, np.array(0.0))
    alpha, beta = clarke_forward(x_abc)
    assert d == pytest.approx(alpha, abs=1e-12)
    assert q == pytest.approx(beta, abs=1e-12)  # both use the q = -sum(x sin(angle)) convention


# -- phase_L_to_dq_L: the Park bridge (plan stage 2) ---------------------------------
#
# The phase-basis L matrix and the dq-basis L matrix are the SAME inductance tensor in two
# bases. phase_L_to_dq_L applies the amplitude-invariant Park rotation: L_dq = (2/n) M L M^T
# where M = [cos(theta_e - axis_k); -sin(theta_e - axis_k)] is the 2 x n forward-Park matrix.
# The (2/n) factor is the amplitude-invariant scaling -- the plan's "T L T^T" notation with the
# forward matrix would double-apply it (measured: T L T^T on a diagonal L_phase gives (2/n)
# too-small dq entries), so the bridge uses the UN-normalised M and the single 2/n factor.


def test_phase_L_to_dq_L_identity_maps_a_balanced_phase_matrix() -> None:
    """A phase matrix with equal diagonal and equal mutuals is rotation-INVARIANT: any theta
    gives the same diagonal dq inductance (a round stator has no preferred axis).
    """
    L_phase = np.array([
        [1.0, -0.2, -0.2],
        [-0.2, 1.0, -0.2],
        [-0.2, -0.2, 1.0],
    ])
    L_dq = phase_L_to_dq_L(L_phase, np.array(0.0))
    assert L_dq.shape == (2, 2)
    # The direct-injection route pins the VALUE: inject [I, 0] at theta=0, solve lambda = L i,
    # forward-Park lambda, read Ld = lambda_d / I. Measured Ld = Lq = 1.2 (0.8 from T L T^T).
    np.testing.assert_allclose(L_dq[0, 0], 1.2, atol=1e-12)
    np.testing.assert_allclose(L_dq[1, 1], 1.2, atol=1e-12)
    # Round stator: no off-diagonal dq coupling and no theta dependence.
    np.testing.assert_allclose(L_dq[0, 1], 0.0, atol=1e-12)
    L_dq_45 = phase_L_to_dq_L(L_phase, np.array(np.pi / 4))
    np.testing.assert_allclose(L_dq_45, L_dq, atol=1e-12)


def test_phase_L_to_dq_L_matches_the_direct_dq_injection_route() -> None:
    """The bridge agrees with injecting a dq current set and Park-forwarding the linkage.

    The direct route is the physical measurement: i_abc = park_inverse([I,0], theta), lambda =
    L_phase @ i_abc, Ld = park_forward(lambda)[0] / I. The bridge must reproduce it for a
    SALIENT (asymmetric) L_phase at an arbitrary theta -- the balanced case alone cannot tell a
    bridge from a constant.
    """
    np.random.default_rng(3)
    theta = np.array(0.37)
    L_phase = np.array([
        [1.4, -0.11, -0.09],
        [-0.11, 1.1, -0.13],
        [-0.09, -0.13, 1.0],
    ])
    L_dq = phase_L_to_dq_L(L_phase, theta)

    I_test = 1.0
    i_abc_d = park_inverse(I_test, 0.0, 0.0, theta, n_phase=3)
    i_abc_q = park_inverse(0.0, I_test, 0.0, theta, n_phase=3)
    lam_d = park_forward(L_phase @ i_abc_d, theta)
    lam_q = park_forward(L_phase @ i_abc_q, theta)
    np.testing.assert_allclose(L_dq[0, 0], lam_d[0] / I_test, atol=1e-12)
    np.testing.assert_allclose(L_dq[1, 1], lam_q[1] / I_test, atol=1e-12)
    # Cross terms too: lambda_q under a d-axis drive is L_dq * I.
    np.testing.assert_allclose(L_dq[0, 1], lam_q[0] / I_test, atol=1e-12)
    np.testing.assert_allclose(L_dq[1, 0], lam_d[1] / I_test, atol=1e-12)


def test_phase_L_to_dq_L_symmetric_from_a_symmetric_phase_matrix() -> None:
    """A symmetric L_phase must map to a symmetric L_dq (reciprocity is basis-independent)."""
    rng = np.random.default_rng(4)
    A = rng.standard_normal((3, 3))
    L_phase = A @ A.T + 2.0 * np.eye(3)  # SPD, symmetric
    theta = np.array(1.1)
    L_dq = phase_L_to_dq_L(L_phase, theta)
    np.testing.assert_allclose(L_dq, L_dq.T, atol=1e-12)


def test_phase_L_to_dq_L_round_trips_a_given_dq_matrix() -> None:
    """Given a target L_dq, the inverse map produces a phase matrix whose bridge is the target.

    The inverse of the bridge (phase matrix from a dq matrix) exists because the forward-Park
    map is a rank-2 projector on the balanced subspace; the round trip on that subspace is the
    identity. This pins the bridge as the TWO-BASIS SAME-TENSOR relation, not a heuristic.
    """
    L_dq_target = np.array([[1.2, 0.05], [0.05, 0.7]])
    theta = np.array(0.5)
    n = 3
    axes = 2.0 * np.pi * np.arange(n) / n
    M = np.array([
        [np.cos(theta - ax) for ax in axes],
        [-np.sin(theta - ax) for ax in axes],
    ])
    # The inverse bridge is (2/n) M.T L_dq M: forward = (2/n) M L M.T, and M M.T = (n/2) I,
    # so the forward of (2/n) M.T L M is (2/n) * (n/2) L (n/2) = L on the balanced subspace.
    L_phase = (2.0 / n) * (M.T @ L_dq_target @ M)
    np.testing.assert_allclose(phase_L_to_dq_L(L_phase, theta), L_dq_target, atol=1e-12)
