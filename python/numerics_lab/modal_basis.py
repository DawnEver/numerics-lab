"""Deterministic generalized symmetric eigenbases and numerical PSD admission."""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

__all__ = ['generalized_modes', 'semidefinite_modes']


def generalized_modes(stiffness, mass, count, *, shift):
    """Return ascending mass-normalized modes; a shift targets, never changes, eigenvalues."""
    size = stiffness.shape[0]
    if count == size or not stiffness.nnz or not np.any(stiffness.data):
        values, vectors = eigh(stiffness.toarray(), mass.toarray())
        return values[:count], vectors[:, :count]
    values, vectors = eigsh(stiffness, k=count, M=mass, sigma=shift, which='LM', v0=np.ones(size))
    order = np.argsort(values)
    return values[order], vectors[:, order]


def semidefinite_modes(stiffness, mass, count, *, minimum_mass):
    """Retain rigid modes, refusing negative stiffness beyond pencil roundoff.

    ``||K||_inf / lambda_min(M)`` bounds the mass-normalized stiffness norm.
    The zero classification and negative shift therefore scale with the actual
    pencil, not an invented stiffness or frequency threshold in physical units.
    """
    size = stiffness.shape[0]
    scale = float(np.max(np.asarray(abs(stiffness).sum(axis=1)), initial=0.0)) / minimum_mass
    if not np.isfinite(scale):
        msg = 'modal pencil stiffness-to-mass scale must be finite and representable'
        raise ValueError(msg)
    floor = 64 * np.finfo(float).eps * scale
    if count == size or scale == 0.0:
        values, vectors = generalized_modes(stiffness, mass, count, shift=0.0)
        minimum = values[0]
    else:
        minimum = float(eigsh(stiffness, k=1, M=mass, which='SA', return_eigenvectors=False, v0=np.ones(size))[0])
        if not np.isfinite(minimum) or minimum < -floor:
            msg = 'modal transient requires positive semidefinite stiffness; materially negative mode found'
            raise ValueError(msg)
        shift = -np.sqrt(np.finfo(float).eps) * scale
        if not np.isfinite(shift) or shift == 0.0:
            msg = 'modal pencil negative shift must be finite and representable'
            raise ValueError(msg)
        values, vectors = generalized_modes(stiffness, mass, count, shift=shift)
    if not np.all(np.isfinite(values)) or minimum < -floor:
        msg = 'modal transient requires positive semidefinite stiffness; materially negative mode found'
        raise ValueError(msg)
    return np.where(np.abs(values) <= floor, 0.0, values), vectors
