"""One material-update loop: the `MaterialUpdate` protocol and the composite every solver drives.

WHAT A MATERIAL UPDATE IS. Every nonlinear magnetic kernel in this repo is LINEAR in a small set of
per-point coefficients -- the reluctivity ``nu`` (and its tangent ``dnu/d(B^2)`` for Newton), the
remanence vector ``Br`` and the recoil permeability -- and nonlinear only through how those
coefficients depend on the field. So the whole "material loop" of a solve is one map

    (state, fields at the points, T)  ->  (state', coefficients at the points)

and a solver's assembly consumes the coefficients and nothing else. Steel B-H, magnet recoil and a
frozen-permeability freeze are three implementations of that one map, and they compose.

POINTS, NOT ELEMENTS. ``fields`` is per POINT: an element centroid, a MEC branch, a Gauss point or
an SDM interior quadrature point are all points to this module. A solver that wants per-element
means passes one point per element; a spectral subdomain passes its interior quadrature set. The
update never knows which.

STATE IS WHAT A FIELD CANNOT RECOMPUTE. A reversible law (steel) is a pure function of the field,
so its state is ``None`` and its :meth:`MaterialUpdate.change` is ``0`` -- the iterate test of
:func:`motronics.gauss.nonlinear._has_converged` already sees it settle. An IRREVERSIBLE law (a
one-way remanence clamp) carries history the field cannot reconstruct, and its change is what the
convergence rule's material term measures. The trial state from :meth:`evaluate` becomes the
COMMITTED state only when the owning system accepts the solve (:class:`MaterialLoop.commit`), so
an unconverged or damped-away iterate never ratchets a magnet.

Physics-free by layering: this module names coefficients the kernels are linear in; the laws that
produce them (a `ReluctivityLaw`, a demag knee) live in ``hamilton`` and implement the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    'COEFFICIENT_NAMES',
    'CompositeUpdate',
    'FrozenNu',
    'MaterialCoeffs',
    'MaterialFields',
    'MaterialLoop',
    'MaterialUpdate',
    'frozen_resolve',
    'relax_linear',
]


@dataclass(frozen=True)
class MaterialFields:
    """The field at ``n`` evaluation points, and which region group each point lies in.

    ``b`` is ``(n, 2)`` (a 2-D in-plane flux density) or ``(n,)`` when a consumer only has
    ``|B|``; :attr:`b_norm` answers both. ``group`` is ``(n,)`` integer region-group ids.
    """

    b: np.ndarray
    group: np.ndarray

    @property
    def n(self) -> int:
        return int(np.asarray(self.group).shape[0])

    @property
    def b_norm(self) -> np.ndarray:
        b = np.asarray(self.b, dtype=float)
        return np.abs(b) if b.ndim == 1 else np.hypot(b[:, 0], b[:, 1])

    @property
    def b_vec(self) -> np.ndarray:
        b = np.asarray(self.b, dtype=float)
        if b.ndim == 1:
            msg = 'this material update needs the flux-density VECTOR at each point; the solver passed only |B|'
            raise ValueError(msg)
        return b


@dataclass(frozen=True)
class MaterialCoeffs:
    """Exactly what the kernels are linear in, per point. ``None`` = this update does not set it."""

    nu: np.ndarray | None = None
    dnu_db2: np.ndarray | None = None
    br: np.ndarray | None = None
    mu_recoil: np.ndarray | None = None


#: The coefficient names, derived from the dataclass so a new one cannot be forgotten by the composite.
COEFFICIENT_NAMES: tuple[str, ...] = tuple(f.name for f in fields(MaterialCoeffs))


@runtime_checkable
class MaterialUpdate(Protocol):
    """One material law in the shape every solver's nonlinear loop drives."""

    #: Whether the state carries history the field cannot recompute (a one-way clamp).
    irreversible: bool

    def init_state(self, fields: MaterialFields, /) -> object: ...

    def evaluate(self, state: object, fields: MaterialFields, temperature=None, /) -> tuple[object, MaterialCoeffs]:
        """The TRIAL state at these fields, and the coefficients it implies. Never mutates ``state``."""
        ...

    def relax(self, previous: MaterialCoeffs, target: MaterialCoeffs, alpha: float, /) -> MaterialCoeffs:
        """Under-relax ``previous -> target`` by the GLOBAL scalar ``alpha`` (linear, so it commutes with assembly)."""
        ...

    def change(self, committed: object, trial: object, /) -> float:
        """Relative change of the irreversible STATE between two states; ``0`` for a reversible law."""
        ...


def _blend(prev: np.ndarray | None, target: np.ndarray | None, alpha: float) -> np.ndarray | None:
    if target is None or prev is None:
        return target
    return (1.0 - alpha) * np.asarray(prev, dtype=float) + alpha * np.asarray(target, dtype=float)


def relax_linear(previous: MaterialCoeffs, target: MaterialCoeffs, alpha: float) -> MaterialCoeffs:
    """The default :meth:`MaterialUpdate.relax`: every set coefficient blended by one global ``alpha``."""
    return MaterialCoeffs(**{k: _blend(getattr(previous, k), getattr(target, k), alpha) for k in COEFFICIENT_NAMES})


class CompositeUpdate:
    """Several updates acting on the same points, e.g. ``[SteelBH, MagnetRecoil]``.

    Each coefficient is OWNED by at most one member: two members setting the same one is a
    modelling contradiction (which ``nu`` is right?) and RAISES rather than letting order decide.
    The state is the tuple of member states; the change is the largest member change.
    """

    def __init__(self, members) -> None:
        self.members = tuple(members)
        if not self.members:
            msg = 'a CompositeUpdate needs at least one member'
            raise ValueError(msg)
        self.irreversible = any(m.irreversible for m in self.members)

    def init_state(self, fields: MaterialFields, /) -> tuple:
        return tuple(m.init_state(fields) for m in self.members)

    def evaluate(self, state, fields: MaterialFields, temperature=None, /):
        trial, merged = [], {}
        for member, s in zip(self.members, state, strict=True):
            s_new, coeffs = member.evaluate(s, fields, temperature)
            trial.append(s_new)
            for k in COEFFICIENT_NAMES:
                v = getattr(coeffs, k)
                if v is None:
                    continue
                if k in merged:
                    msg = f'two material updates both set {k!r}: {[type(m).__name__ for m in self.members]}'
                    raise ValueError(msg)
                merged[k] = v
        return tuple(trial), MaterialCoeffs(**merged)

    def relax(self, previous: MaterialCoeffs, target: MaterialCoeffs, alpha: float, /) -> MaterialCoeffs:
        return relax_linear(previous, target, alpha)

    def change(self, committed, trial, /) -> float:
        return max(
            (m.change(c, t) for m, c, t in zip(self.members, committed, trial, strict=True)),
            default=0.0,
        )


class MaterialLoop:
    """The committed/trial bookkeeping a SYSTEM holds for its update, so no strategy has to.

    Every :meth:`evaluate` computes a trial FROM THE COMMITTED STATE -- never from the previous
    trial -- so inside one solve an irreversible law is a pure function of the iterate: an
    unconverged, overshooting iterate cannot ratchet a one-way clamp. :attr:`last_change` is the
    change between CONSECUTIVE trials, the material term of the one convergence rule: it is zero
    exactly when the state the iterate implies has stopped moving. :meth:`commit` promotes the last
    trial and is called when the solve's result is ACCEPTED, i.e. converged -- the one place a
    magnet's history is written.
    """

    def __init__(self, update: MaterialUpdate, fields: MaterialFields) -> None:
        self.update = update
        self.committed = update.init_state(fields)
        self._trial = self.committed
        self.last_change = 0.0
        #: The coefficients of the last trial -- what the NEXT assembly consumes; ``None`` before
        #: the first evaluation (the assembly then uses the committed state it was built from).
        self.coeffs: MaterialCoeffs | None = None

    def evaluate(self, fields: MaterialFields, temperature=None) -> MaterialCoeffs:
        trial, coeffs = self.update.evaluate(self.committed, fields, temperature)
        self.last_change = float(self.update.change(self._trial, trial))
        self._trial = trial
        self.coeffs = coeffs
        return coeffs

    def commit(self) -> None:
        self.committed = self._trial


def frozen_resolve(system, x_frozen: np.ndarray, /) -> np.ndarray:
    """ONE linear solve of ``system`` with its material coefficients FROZEN at the iterate ``x_frozen``.

    The frozen-permeability re-solve, native and solver-agnostic. Every system in this repo
    evaluates its reversible coefficients (a steel ``nu``) from the field of the iterate it is
    assembled at and NOT from its source, so assembling ``system`` -- which may carry a DIFFERENT
    source than the solve that converged to ``x_frozen`` (magnets only, one current axis) -- at
    ``x_frozen`` builds exactly the operator of that converged material state; solving it with the
    system's own source is the satellite study. By linearity of that frozen operator the satellite
    fields superpose to the converged one, which is the whole method.
    """
    return system.solve_linear(system.assemble(np.asarray(x_frozen)))


class FrozenNu:
    """A material update whose ``nu`` is FROZEN per point: the coefficients of one converged evaluation.

    Built from any reversible update at a converged field (:meth:`from_update`), it ignores the
    field it is later evaluated at -- the frozen-permeability material state as a
    :class:`MaterialUpdate`, for a system that assembles from per-point coefficients.
    """

    irreversible = False

    def __init__(self, nu: np.ndarray) -> None:
        self.nu = np.asarray(nu, dtype=float).copy()

    @classmethod
    def from_update(cls, update: MaterialUpdate, fields: MaterialFields, temperature=None) -> FrozenNu:
        _state, coeffs = update.evaluate(update.init_state(fields), fields, temperature)
        if coeffs.nu is None:
            msg = f'{type(update).__name__} sets no nu, so there is nothing to freeze'
            raise ValueError(msg)
        return cls(coeffs.nu)

    def init_state(self, fields: MaterialFields, /) -> None:
        del fields

    def evaluate(self, state, fields: MaterialFields, temperature=None, /):
        del fields, temperature
        return state, MaterialCoeffs(nu=self.nu.copy())

    def relax(self, previous: MaterialCoeffs, target: MaterialCoeffs, alpha: float, /) -> MaterialCoeffs:
        return relax_linear(previous, target, alpha)

    def change(self, committed, trial, /) -> float:
        del committed, trial
        return 0.0
