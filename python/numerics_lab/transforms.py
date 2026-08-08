"""Coordinate transforms -- the physics-free math both hamilton and laplace consume (round-2 T1).

The Park/Clarke family is ORTHOGONAL COORDINATE ROTATION, not physics: rotating a balanced
N-phase set into the rotor frame is the same linear algebra whoever asks for it (a field
post-processing route, a control law, a drive signal chain). This module is their single home
under the "physics-free numerical runtime" -- the dependency arrow
``workflow -> {hamilton, laplace} -> gauss -> contracts`` lets BOTH platforms import it, which
is exactly why the extraction happened: laplace needed ``park_inverse`` for the FOC signal chain
and could not import hamilton. The motor dq EQUATIONS (``PMSMParameters`` etc.) are physics and
stay where the platforms keep them.

All transforms are amplitude-invariant: a balanced set of peak amplitude ``I`` maps to a
(d, q) vector of magnitude ``I``. Imports only numpy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _phase_axes(n_phase: int) -> NDArray[np.float64]:
    """Electrical axis angle of each of ``n_phase`` symmetric phases: ``2*pi*k/n_phase``."""
    return 2.0 * np.pi * np.arange(n_phase, dtype=float) / n_phase


def park_forward(
    x_phase: NDArray[np.float64],
    theta_e: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Park transform: N-phase stator quantities -> rotor-frame ``(d, q, zero)``.

    Projects the symmetric N-phase set (phase k on electrical axis ``2*pi*k/N``) onto the
    rotor-aligned direct and quadrature axes. Amplitude-invariant (the ``2/N`` scaling), so a
    balanced set of peak amplitude ``I`` aligned with the rotor gives ``d = I, q = 0`` -- the
    d-q amplitudes equal the phase amplitude. The zero-sequence is the phase mean.

    Args:
        x_phase: ``(n_phase, n_theta)`` per-phase quantity (flux linkage or current) at each
            rotor position; or ``(n_phase,)`` for a single position.
        theta_e: ``(n_theta,)`` or scalar rotor ELECTRICAL angle (rad) at each position.

    Returns:
        ``(d, q, zero)`` each ``(n_theta,)`` (or scalar), the rotor-frame components.

    """
    x = np.asarray(x_phase, dtype=float)
    n_phase = x.shape[0]
    theta_e = np.asarray(theta_e, dtype=float)
    axes = _phase_axes(n_phase)[:, None] if x.ndim == 2 else _phase_axes(n_phase)
    angle = theta_e - axes  # (n_phase, n_theta) or (n_phase,)
    d = (2.0 / n_phase) * np.sum(x * np.cos(angle), axis=0)
    q = -(2.0 / n_phase) * np.sum(x * np.sin(angle), axis=0)
    zero = np.mean(x, axis=0)
    return d, q, zero


def park_inverse(
    d: NDArray[np.float64],
    q: NDArray[np.float64],
    zero: NDArray[np.float64],
    theta_e: NDArray[np.float64],
    n_phase: int = 3,
) -> NDArray[np.float64]:
    """Inverse Park: rotor-frame ``(d, q, zero)`` -> ``(n_phase, n_theta)`` phase quantities.

    The exact inverse of :func:`park_forward` for the amplitude-invariant convention: phase k is
    ``d*cos(theta_e - axis_k) - q*sin(theta_e - axis_k) + zero``. Round-trips
    ``park_inverse(park_forward(x)) == x`` to machine precision for any ``n_phase >= 3`` (the
    forward map has a d, q, and zero degree of freedom; the remaining ``n_phase - 3`` phase
    harmonics vanish for a symmetric set, which is the standard balanced-machine assumption).

    Args:
        d, q, zero: rotor-frame components, each ``(n_theta,)`` or scalar.
        theta_e: rotor electrical angle (rad), broadcastable to ``d``; must be 1-D or scalar
            (a 2-D grid is refused -- the phase axis is the ``n_phase`` parameter's, never an
            input axis).
        n_phase: number of phases to reconstruct (default 3).

    Returns:
        ``(n_phase, n_theta)`` (or ``(n_phase,)``) reconstructed phase quantities.

    """
    d = np.asarray(d, dtype=float)
    q = np.asarray(q, dtype=float)
    zero = np.asarray(zero, dtype=float)
    theta_e = np.asarray(theta_e, dtype=float)
    # A 2-D theta_e is REFUSED (SR-031): after the broadcast its leading axis would be
    # silently consumed as the phase axis -- a (3, n) angle grid whose leading axis equals
    # n_phase returns wrong numbers with NO error, and other 2-D shapes surface as a raw
    # numpy broadcast failure. The sample axis is 1-D by contract; an (n_phase, n) phase
    # grid is the n_phase parameter's job, never an input grid.
    if theta_e.ndim > 1:
        msg = (
            f'theta_e must be 1-D or scalar (the SAMPLE axis), got {theta_e.ndim} axes of '
            f'shape {theta_e.shape}: an (n_phase, n) phase grid is the n_phase parameter\'s '
            'job, never an input grid -- a 2-D theta_e whose leading axis equals n_phase '
            'would otherwise be silently consumed as the phase axis.'
        )
        raise ValueError(msg)
    # Broadcast the commands AND the angle to a common sample shape FIRST (the docstring's
    # "broadcastable to d" contract): the sample axis is established by the broadcast, never
    # inferred from theta_e.ndim alone. A (3,) command vector with a scalar angle is
    # n_theta = 3 SAMPLES at one angle, not three phases -- keying the phase axis on
    # theta_e.ndim conflated the two and returned a (3,) result (SR-024). The phase axis is
    # prepended AFTER the broadcast, so the sample axis can never collide with it.
    d, q, zero, theta_e = np.broadcast_arrays(d, q, zero, theta_e)
    axes = _phase_axes(n_phase)[:, None] if d.ndim else _phase_axes(n_phase)
    angle = theta_e - axes
    return d * np.cos(angle) - q * np.sin(angle) + zero


def clarke_forward(x_phase: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Amplitude-invariant Clarke transform: 3-phase ``(a, b, c)`` -> stationary ``(alpha, beta)``.

    ``alpha = (2/3)*(a - b/2 - c/2)``, ``beta = (2/3)*(sqrt(3)/2)*(b - c)``. For a balanced set of
    peak ``I`` the ``(alpha, beta)`` vector has magnitude ``I`` (amplitude-invariant), matching the
    Park convention in :func:`park_forward`. ``x_phase`` is ``(3,)`` or ``(3, n)``.
    """
    x = np.asarray(x_phase, dtype=float)
    alpha = (2.0 / 3.0) * (x[0] - 0.5 * x[1] - 0.5 * x[2])
    beta = (2.0 / 3.0) * (np.sqrt(3.0) / 2.0) * (x[1] - x[2])
    return alpha, beta


__all__ = ['clarke_forward', 'park_forward', 'park_inverse']
