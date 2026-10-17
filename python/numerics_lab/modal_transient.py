"""Mass-normalized modal superposition for constant linear second-order operators."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh

from numerics_lab.modal_basis import semidefinite_modes
from numerics_lab.time_integrators import TransientSolution, get_integrator, integrate_transient

__all__ = ['ModalTransientResult', 'integrate_modal_transient']


@dataclass(frozen=True)
class ModalTransientResult:
    solution: TransientSolution
    eigenvalues: np.ndarray
    mode_shapes: np.ndarray
    initial_projection_error: float
    load_projection_error: float


@dataclass(frozen=True)
class _ReducedProblem:
    stiffness: np.ndarray
    damping: np.ndarray
    load: Callable[[float], np.ndarray]

    def residual(self, u, u_dot, u_ddot, t):
        return self.stiffness @ u + self.damping @ u_dot + u_ddot - self.load(t)

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return (
            weights.alpha0.real * self.stiffness
            + weights.alpha1.real * self.damping
            + weights.alpha2.real * np.eye(len(self.stiffness))
        )


def integrate_modal_transient(
    stiffness, mass, damping, load, window, *, n_modes, u0=None, v0=None, a0=None, damping_ratios=None
):
    """Project ``M u'' + C u' + K u = f`` onto the lowest retained eigenmodes.

    Operators are constant, symmetric, already restricted to unconstrained DOFs.
    Mass must be positive definite; stiffness may be semidefinite, retaining
    rigid modes with their inertia. Damping is fully projected, never discarded merely
    because modal coordinates make K diagonal. Optional per-mode damping ratios
    replace C rather than adding a second, ambiguous damping definition.
    ``initial_projection_error`` measures the worst relative mass-norm initial
    displacement/velocity omitted by
    truncation; zero integration residual is not evidence of truncation accuracy.
    ``load`` is a finite vector or a callable returning that vector at absolute
    integrator evaluation time. Load projection error is the worst visited value.
    """
    k, m, c = (csc_matrix(operator, dtype=float) for operator in (stiffness, mass, damping))
    size = k.shape[0]
    if (
        k.shape != (size, size)
        or m.shape != k.shape
        or c.shape != k.shape
        or isinstance(n_modes, bool)
        or not isinstance(n_modes, int)
        or not 1 <= n_modes <= size
        or not all(np.all(np.isfinite(operator.data)) for operator in (k, m, c))
    ):
        msg = 'modal operators must be finite matching square matrices and 1 <= n_modes <= size'
        raise ValueError(msg)
    for operator in (k, m, c):
        scale = float(np.max(np.abs(operator.data))) if operator.nnz else 0.0
        symmetry_floor = 64 * np.finfo(float).eps * scale
        if np.any(np.abs((operator - operator.T).data) > symmetry_floor):
            msg = 'modal superposition requires symmetric operators'
            raise ValueError(msg)
    # Validate the generalized pencil rather than trusting ARPACK's SPD-M assumption.
    # One smallest-algebraic mass mode detects an indefinite mass even when retained
    # vibration modes accidentally have positive generalized eigenvalues.
    minimum_mass = (
        float(m[0, 0])
        if size == 1
        else float(eigsh(m, k=1, which='SA', return_eigenvectors=False, v0=np.ones(size))[0])
    )
    if not np.isfinite(minimum_mass) or minimum_mass <= 0:
        msg = 'modal transient requires positive definite mass on all free DOFs'
        raise ValueError(msg)
    if not get_integrator(window.integrator).weights(window.dt).alpha2:
        msg = 'modal transient requires a second-order integrator'
        raise ValueError(msg)
    eigenvalues, shapes = semidefinite_modes(k, m, n_modes, minimum_mass=minimum_mass)
    modal_c = shapes.T @ (c @ shapes)
    if damping_ratios is not None:
        ratios = np.asarray(damping_ratios, dtype=float)
        if c.nnz and np.any(c.data):
            msg = 'modal damping ratios cannot combine with operator damping'
            raise ValueError(msg)
        if ratios.shape != (n_modes,) or not np.all(np.isfinite(ratios)) or np.any((ratios < 0) | (ratios > 1)):
            msg = 'provide one finite modal damping ratio in [0, 1] per retained mode'
            raise ValueError(msg)
        modal_c = np.diag(2 * ratios * np.sqrt(eigenvalues))
    initial = np.zeros(size) if u0 is None else np.asarray(u0, dtype=float)
    velocity = np.zeros(size) if v0 is None else np.asarray(v0, dtype=float)
    if (
        initial.shape != (size,)
        or velocity.shape != (size,)
        or not np.all(np.isfinite(initial))
        or not np.all(np.isfinite(velocity))
    ):
        msg = 'modal initial displacement and velocity must match the free operator dimension'
        raise ValueError(msg)
    q0, qv = shapes.T @ (m @ initial), shapes.T @ (m @ velocity)
    load_error = 0.0

    def projected_load(t):
        nonlocal load_error
        force = np.asarray(load(t) if callable(load) else load, dtype=float)
        if force.shape != (size,) or not np.all(np.isfinite(force)):
            msg = 'modal load must be a finite vector matching the free operator dimension'
            raise ValueError(msg)
        coordinates = shapes.T @ force
        missing = force - m @ (shapes @ coordinates)
        norm = float(np.linalg.norm(force))
        load_error = max(load_error, float(np.linalg.norm(missing)) / norm if norm else 0.0)
        return coordinates

    initial_load = projected_load(window.t0)
    reduced = _ReducedProblem(np.diag(eigenvalues), modal_c, projected_load)
    if a0 is not None and (np.asarray(a0).shape != (size,) or not np.all(np.isfinite(a0))):
        msg = 'modal initial acceleration must be finite and match the free operator dimension'
        raise ValueError(msg)
    # A missing acceleration is derived from equilibrium, not silently set to zero.
    qa = (
        initial_load - reduced.stiffness @ q0 - modal_c @ qv
        if a0 is None
        else shapes.T @ (m @ np.asarray(a0, dtype=float))
    )
    projected = integrate_transient(reduced, window, q0, v0=qv, a0=qa, on_nonconverged='label')
    physical = TransientSolution(
        times=projected.times,
        states=[shapes @ q for q in projected.states],
        worst_step_residual=projected.worst_step_residual,
        nonconverged_times=projected.nonconverged_times,
        floor_converged_times=projected.floor_converged_times,
    )
    errors = []
    for vector, coordinates in ((initial, q0), (velocity, qv)):
        omitted = vector - shapes @ coordinates
        initial_norm = np.sqrt(max(float(vector @ (m @ vector)), 0.0))
        omitted_norm = np.sqrt(max(float(omitted @ (m @ omitted)), 0.0))
        errors.append(omitted_norm / initial_norm if initial_norm else omitted_norm)
    return ModalTransientResult(physical, eigenvalues, shapes, max(errors), load_error)
