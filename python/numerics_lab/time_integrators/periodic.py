"""Physics-free periodic convergence on accepted states, including nonuniform grids."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral, Real

import numpy as np

from motronics.contracts.spec.analysis_formulation import DerivativeWeights

__all__ = [
    'bdf2_boundary_moment',
    'canonical_eec_study',
    'explicit_error_correction',
    'nonlinear_error_correction',
    'periodic_converged',
    'validate_eec_restart',
]


def canonical_eec_study(study, restart):
    """Resolve a declared uniform tick to its one original-clock representation.

    Identity validation remains exact: only the unique original integer tick may
    replace the caller's algebraically equivalent rounded endpoint expression.
    """
    if restart is None or restart.eec_state is None:
        return study
    origin = restart.eec_state.origin_time
    phase = (study.t0 - origin) / study.dt
    if not np.isfinite(phase) or not np.isclose(phase, round(phase), rtol=0.0, atol=1e-9):
        msg = 'EEC restart time must align with the original uniform integration clock'
        raise ValueError(msg)
    time = origin + round(phase) * study.dt
    if time != restart.time:
        msg = 'EEC restart original-clock tick does not match the checkpoint time'
        raise ValueError(msg)
    return replace(study, t0=time)


def validate_eec_restart(restart, period, sign, dt, nonlinear=False):
    """Reject missing or changed algorithm controls before restoring physical history."""
    if restart is None:
        return
    if (restart.eec_state is None) != (period is None):
        msg = 'EEC restart requires its saved cycle-origin history and active correction controls'
        raise ValueError(msg)
    if restart.eec_state is not None:
        restart.eec_state.validate_controls(period, sign, nonlinear)
        phase = (restart.time - restart.eec_state.origin_time) / dt
        if not np.isfinite(phase) or not np.isclose(phase, round(phase), rtol=0.0, atol=1e-9):
            msg = 'EEC restart time must align with the original uniform integration clock'
            raise ValueError(msg)


def bdf2_boundary_moment(history):
    """Uniform BDF2 telescopes to z_end-z_start, z=1.5u_n-0.5u_(n-1).

    With a backward-Euler first step, the initial boundary moment instead is
    0.5*(u_0+u_1); the driver captures it after that physical bootstrap solve.
    A constant coarse shift p of BOTH history states shifts z by p without
    destroying the BDF2 derivative offsets. This affine symmetric K/M coarse
    specialization is not a derivation of JMAG's proprietary implementation.
    """
    return 1.5 * history[-1] - 0.5 * history[-2]


def nonlinear_error_correction(problem, current, previous, time, integrated_tangent, *, solve):
    """Uniform BE coarse linearization using the actual cycle-integrated Jacobian.

    COMPUMAG 2019 PA-A5-9 II.B equation (5), single process/full period:
    ``dt sum J_g(u_i) p = M (u_end-u_start)``. This local coarse approximation
    has no global contraction guarantee and is not proprietary HTP-EEC.
    """
    zero = np.zeros_like(current)
    mass = problem.jacobian(current, zero, zero, time, DerivativeWeights(alpha0=0.0, alpha1=1.0))
    return current + np.asarray(solve(integrated_tangent, mass @ (current - previous)), dtype=float)


def explicit_error_correction(problem, current, previous, period, time, *, solve, sign=1):
    """First-order TP-EEC: remove the estimated cycle-mean transient error.

    For ``M u' + K u = f(t)``, integration over one supply period gives
    ``K mean(error) = -M (u_end-u_start)/T``. Adding ``K^-1 M delta/T``
    removes that mean error. This is EEC, not endpoint extrapolation or an
    exact periodic shooting solve; it is most effective for long time constants.
    The caller must supply a time-invariant affine first-order problem.

    Native half-period specialization (sign=-1) of the uniform backward-Euler
    coarse correction in Takahashi et al., COMPUMAG 2019, II.B, equations (3)-(5):
    https://www.compumag.org/Proceedings/2019_Paris/files/papers/PA-A5-9.pdf
    ``(n K + 2 M/dt) p = -(M/dt)(u_end+u_start)``. The correction is
    injected at the ENDPOINT'S advanced phase: ``u_end+p``, never ``-u_end+p``.
    This scholarly affine specialization is not proof of JMAG's proprietary
    implementation and implements neither reverse rotor correction nor HTP-EEC.
    """
    if isinstance(sign, bool) or not isinstance(sign, Integral) or sign not in (-1, 1):
        msg = 'EEC sign must be the explicit integer +1 or -1'
        raise ValueError(msg)
    if not isinstance(period, Real) or not np.isfinite(period) or period <= 0:
        msg = 'EEC correction interval must be a positive finite scalar'
        raise ValueError(msg)
    zero = np.zeros_like(current)
    stiffness = problem.jacobian(current, zero, zero, time, DerivativeWeights(alpha0=1.0))
    mass = problem.jacobian(current, zero, zero, time, DerivativeWeights(alpha0=0.0, alpha1=1.0))
    if sign == -1:
        correction = solve(stiffness + 2 * mass / period, -mass @ ((current + previous) / period))
        return current + np.asarray(correction, dtype=float)
    return current + np.asarray(solve(stiffness, mass @ ((current - previous) / period)), dtype=float)


def periodic_converged(times, states, period: float, atol: float, rtol: float) -> bool:
    """Compare every sample in the last cycle with interpolated previous-cycle states.

    Two complete cycles are mandatory. Comparing endpoints alone would mislabel a
    changing waveform whose endpoints happen to agree as steady.
    """
    if times[-1] - times[0] < 2 * period - 1e-12 * period:
        return False
    ts = np.asarray(times)
    us = np.asarray(states)
    recent = ts >= ts[-1] - period
    targets = ts[recent] - period
    previous = np.stack([np.interp(targets, ts, col) for col in us.reshape(len(ts), -1).T], axis=1)
    current = us[recent].reshape(len(targets), -1)
    return bool(np.all(np.abs(current - previous) <= atol + rtol * np.maximum(np.abs(current), np.abs(previous))))
