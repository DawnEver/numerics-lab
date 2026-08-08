"""Coordinate transforms (Park/Clarke) -- the physics-free math home (round-2 T1).

Migrated from hamilton (postprocessing/dq_inductance, coupling/foc) when the transforms became
the shared home both hamilton and laplace import: the round-trip identities, the amplitude
invariance, the N-phase generalization, and the balanced-set-to-DC property are the same
contract in the new home.
"""

from __future__ import annotations

import numpy as np
import pytest

from motronics.gauss.transforms import clarke_forward, park_forward, park_inverse


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


def test_park_and_clarke_agree_on_a_balanced_set_at_theta_zero() -> None:
    """The two transforms are one convention: at theta_e = 0 the d-axis equals alpha."""
    rng = np.random.default_rng(2)
    x_abc = rng.standard_normal(3)
    d, q, _zero = park_forward(x_abc, np.array(0.0))
    alpha, beta = clarke_forward(x_abc)
    assert d == pytest.approx(alpha, abs=1e-12)
    assert q == pytest.approx(beta, abs=1e-12)  # both use the q = -sum(x sin(angle)) convention
