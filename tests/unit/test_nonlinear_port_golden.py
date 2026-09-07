"""The nonlinear-strategy port's golden ITERATION TRAJECTORIES — one file, read by BOTH languages.

`.claude/memory/2026/09/03/plan-rust-migration-execution.md` ports
`src/motronics/gauss/nonlinear/` to `rust/crates/gauss/src/nonlinear/`. Two implementations of
one iteration scheme need a way to be compared, and the obvious one — call the Rust from Python
across the FFI — is not available on this workstation: a second native extension cannot be
installed into MAIN's venv from a lane (`.claude/rules/workflow.md`: "never mutate a shared
resource the other party learns about by accident").

So the comparison goes through a committed DATA file instead.
`rust/crates/gauss/tests/goldens/nonlinear.json` holds the fixtures and what the PYTHON does on
them. Two tests read it:

* this one, asserting the live Python still produces those trajectories;
* `rust/crates/gauss/tests/golden_nonlinear.rs` (Rust), asserting the port does too.

Together they are the equivalence proof. Separately, the first one IS the freeze protocol
(plan §3) made executable: prose cannot stop `nonlinear/` taking a behaviour change while the
port is in flight. This test can.

## Why the golden freezes TRAJECTORIES and not just the answer

Every strategy here converges to the SAME fixed point — that is what makes them peers. A golden
that pinned only `result.x` would therefore pass against a port that reached the right answer by
a completely different path: a broken alpha ratchet, a dead stagnation break, a Newton that
silently fell back to Picard. So what is frozen is the PER-ITERATION `(alpha, energy error,
increment)` triple that `on_iteration` reports, which is exactly the sequence those defects
change. The measured regressions this module carries — 557 of 605 iterations pinned at alpha
0.100, ~85 wasted iterations on a stationary increment — are all trajectory facts, invisible in a
final answer.

Two floors are asserted rather than assumed: every registered strategy must appear, and the
fixtures must include one that DAMPS (alpha leaves 1.0) and one that does not. Without the second,
the whole alpha-ratchet half of the port would be untested by a golden that still looked full.

Regenerate deliberately, never as a side effect of a run::

    MOTRONICS_REGEN_PORT_GOLDENS=1 .venv/Scripts/python.exe -m pytest \
        tests/unit/gauss/test_nonlinear_port_golden.py

A golden that rewrites itself when it disagrees is not a golden.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from tests.port_goldens import dump_golden, load_golden

from motronics.core.utils.config import repo_root
from motronics.gauss.nonlinear import ITERATION_REGISTRY, AssembledState, get_strategy

_GOLDEN = repo_root() / 'rust' / 'crates' / 'gauss' / 'tests' / 'goldens' / 'nonlinear.json'

#: `nu-damped` is driven on its own fixture: it needs the raw pre-fold pair (`raw_state` /
#: `fold_raw`) and REFUSES a system without it, so it cannot share the generic one. Named here
#: rather than filtered silently, because a registry-derived list that quietly stops covering
#: something is the failure this file's coverage test exists to catch.
_RAW_ONLY = 'nu-damped'
_GENERIC = sorted(set(ITERATION_REGISTRY) - {_RAW_ONLY})


class _AffineSystem:
    """`g(x)_i = slope_i * x_i + b_i * lam` — one map, enough physics-free structure for all twelve.

    The fixed point is `x*_i = lam * b_i / (1 - slope_i)`. A slope BELOW 1 contracts; a NEGATIVE
    slope overshoots every step, which is the period-2 oscillation the damped Picard's sign-flip
    ratchet exists for. Per-component slopes (rather than one scalar) matter for `anderson`: a map
    with a single slope makes every fixed-point residual collinear, and a multi-secant method then
    has a rank-1 history to fit — nothing to find, and a degenerate least-squares problem whose
    answer depends on the solver rather than on the physics.

    `total_energy = sum(x**2)` drives the energy-change convergence test; the tangent of
    `r(x) = x - g(x)` is the diagonal `1 - slope`, supplied as a PLAIN ARRAY on purpose: it is not
    a scipy sparse matrix, so `newton-krylov` takes its opaque-tangent branch and
    `modified-newton`'s `_lu_factorize` returns `None` — both routing through `solve_system`,
    which is the behaviour the port implements.
    """

    def __init__(
        self, slope, b, *, is_linear: bool = False, lam: float = 1.0, trace=None, energy_schedule=None
    ) -> None:
        self.slope = np.asarray(slope, float)
        self.b = np.asarray(b, float)
        self.is_linear = is_linear
        self.lam = float(lam)
        self.trace: list[dict] = [] if trace is None else trace
        #: An energy sequence INDEPENDENT of the iterate, held at its last value once exhausted.
        #: `sum(x**2)` on a contraction is monotone, so it can never drive alpha to its floor --
        #: and a golden that never reaches the floor cannot detect the floor moving (measured:
        #: perturbing `_ALPHA_MIN` by 1e-7 left every trajectory unchanged). A prescribed
        #: oscillation is how the ratchet, the cap and the recovery leg become observable.
        self.energy_schedule = None if energy_schedule is None else [float(e) for e in energy_schedule]
        self.assembled = 0

    def initial_guess(self):
        return np.zeros(self.b.size)

    def _g(self, x):
        if self.is_linear:
            return self.lam * self.b  # a genuinely linear system: the target, whatever x is
        return self.slope * np.asarray(x, float) + self.lam * self.b

    def assemble(self, x, /):
        x = np.asarray(x, float)
        if self.energy_schedule is None:
            energy = float(np.sum(x**2))
        else:
            energy = self.energy_schedule[min(self.assembled, len(self.energy_schedule) - 1)]
        self.assembled += 1
        return AssembledState(system_matrix=None, total_energy=energy, payload=x.copy())

    def solve_linear(self, state, /):
        return self._g(state.payload)

    def residual(self, x, state, /):
        x = np.asarray(x, float)
        return x - self._g(x)

    def residual_norm(self, x, state, /) -> float:
        # The Protocol's DEFAULT body, spelled out: a duck-typed system inherits nothing from a
        # `Protocol`, and `newton-linesearch` calls this on every backtrack.
        return float(np.linalg.norm(self.residual(x, state)))

    def tangent(self, x, state, /):
        # ALWAYS a tangent, including for the linear fixture, and the reason is a real gap rather
        # than a convenience. `newton-krylov` has two branches: a tangent it cannot apply routes
        # through `solve_system`, and NO tangent goes Jacobian-free through GMRES. The Rust port
        # implements the first and NOT the second (see `NewtonKrylovStrategy`'s docs: a Krylov
        # backend lives in `gauss::linear`, not in the strategy). Handing every fixture a tangent
        # keeps the golden a statement about what the port claims to do. Measured while building
        # this file: with `None` here the two sides disagreed by 1e-9 on the very first increment
        # -- GMRES's own `rtol=1e-8`, not a porting error, and exactly the branch that is missing.
        return 1.0 - self.slope

    def solve_system(self, matrix, rhs, /):
        return np.asarray(rhs, float) / np.asarray(matrix, float)

    def solve_secant(self, state, rhs, /):
        return np.asarray(rhs, float)  # H0 = I, the seed the quasi-Newton family is given

    def scale_source(self, lam, /):
        # A COPY sharing the trace, so continuation's sub-solves land in one ordered record.
        return _AffineSystem(
            self.slope,
            self.b,
            is_linear=self.is_linear,
            lam=lam,
            trace=self.trace,
            energy_schedule=self.energy_schedule,
        )

    def on_iteration(self, i, state, err, alpha, increment, /) -> None:
        self.trace.append({'alpha': _num(alpha), 'err': _num(err), 'increment': _num(increment)})


class _NuMixSystem:
    """The FEA `raw_state`/`fold_raw` contract, reduced to one scalar per call.

    Raw stiffness and source are LINEAR in a single `k` (standing in for the per-element
    reluctivity target), which is the identity `nu-damped` rests on. The `k` sequence forces a
    KNOWN alpha ratchet at a KNOWN iteration, so the mixed value is hand-computable: energies
    `k**2` give deltas +8 then -5, the sign flip ratchets alpha to 0.6, and the fourth mix is
    `0.4*2 + 0.6*5 = 3.8`.
    """

    is_linear = False

    def __init__(self, k_seq) -> None:
        self.k_seq = list(k_seq)
        self.calls = 0
        self.folded: list[list[float]] = []
        self.trace: list[dict] = []

    def initial_guess(self):
        return np.zeros(1)

    def raw_state(self, x):
        k = self.k_seq[min(self.calls, len(self.k_seq) - 1)]
        self.calls += 1
        return np.array([0]), np.array([0]), np.array([k]), np.array([2.0 * k]), float(k**2)

    def fold_raw(self, s_row, s_col, s_val, t_mat, total_energy=0.0):
        self.folded.append([float(s_val[0]), float(t_mat[0])])
        return AssembledState(system_matrix=s_val, total_energy=total_energy, payload={'T_mat': t_mat})

    def solve_linear(self, state, /):
        return np.array([1.0])  # irrelevant: `raw_state` ignores x and follows `k_seq`

    def on_iteration(self, i, state, err, alpha, increment, /) -> None:
        self.trace.append({'alpha': _num(alpha), 'err': _num(err), 'increment': _num(increment)})


def _num(value) -> float | None:
    """JSON has no infinity. The first pass has no energy predecessor, so its error IS infinite —
    recorded as `null` rather than as a large finite lie, because a reader must be able to tell
    "no predecessor" from "a huge change".
    """
    v = float(value)
    return v if math.isfinite(v) else None


#: The generic fixtures, chosen so the ratchet logic is exercised on BOTH sides of its decision.
#: `contraction` never disturbs the energy, so alpha must stay at 1.0 throughout; `oscillating`
#: has a negative slope and overshoots every step, so the sign-flip ratchet must pull alpha down.
#: A fixture set with only the first would leave the entire damping half of the port unmeasured
#: while still looking like full coverage — the floor `the_fixtures_exercise_both_damping_answers`
#: asserts.
_FIXTURES: dict[str, dict] = {
    'contraction': {'slope': [0.5, 0.3, 0.7], 'b': [2.0, 4.0, 1.0], 'is_linear': False, 'max_iter': 12},
    'oscillating': {'slope': [-1.5, -0.8], 'b': [1.0, 2.0], 'is_linear': False, 'max_iter': 12},
    'linear': {'slope': [0.0, 0.0], 'b': [3.0, -1.5], 'is_linear': True, 'max_iter': 6},
    # The alpha FLOOR, and then the way back off it. A period-2 energy oscillation at 5% amplitude
    # -- far above the 0.01% tolerance, so the `meaningful` guard admits it -- ratchets alpha
    # 1 -> 0.6 -> 0.36 -> 0.216 -> 0.1296 -> 0.1 (clamped), and the flat tail afterwards exercises
    # the calm counter, the cap and the cap's own release. Every measured defect this module
    # carries lives on this trajectory, and no fixture above reaches any of it: an energy of
    # `sum(x**2)` on a contraction is monotone and never disturbs alpha at all.
    'oscillate_then_settle': {
        'slope': [0.9],
        'b': [1.0],
        'is_linear': False,
        'max_iter': 40,
        'energy_schedule': [100.0 + (5.0 if k % 2 == 0 else -5.0) for k in range(10)] + [100.0],
        # EXCLUDED HERE ONLY, and only from this fixture, because the exclusion is about the
        # FIXTURE and not about the strategy. `continuation` is the one strategy that builds a new
        # system per ramp step (`scale_source` returns a COPY on the Python side, while the Rust
        # port scales IN PLACE -- a documented deviation), and a copy restarts the schedule's call
        # counter. Real physics reads its energy off the ITERATE, where the two are identical, and
        # `continuation` is compared in full on the other three fixtures. Excluding it on all of
        # them would be hiding the strategy; excluding it here names an artifact of the ruler.
        'exclude': ['continuation'],
    },
}

_RTOL_PCT = 0.01
_NU_K_SEQ = [1.0, 3.0, 2.0, 5.0]


def _run_generic(name: str, fixture: dict) -> dict:
    system = _AffineSystem(
        fixture['slope'],
        fixture['b'],
        is_linear=fixture['is_linear'],
        energy_schedule=fixture.get('energy_schedule'),
    )
    result = get_strategy(name).solve(system, max_iter=fixture['max_iter'], rtol_energy_pct=_RTOL_PCT)
    return {
        'trace': system.trace,
        'x': [_num(v) for v in np.asarray(result.x, float).ravel()],
        'iterations': int(result.iterations),
        'converged': bool(result.converged),
    }


def _build() -> dict:
    """Everything the two implementations must agree on, computed by the PYTHON."""
    fixtures = {}
    for fixture_name, spec in _FIXTURES.items():
        fixtures[fixture_name] = {
            'slope': spec['slope'],
            'b': spec['b'],
            'is_linear': spec['is_linear'],
            'max_iter': spec['max_iter'],
            'energy_schedule': spec.get('energy_schedule'),
            'strategies': {name: _run_generic(name, spec) for name in _GENERIC if name not in spec.get('exclude', ())},
        }

    nu = _NuMixSystem(_NU_K_SEQ)
    nu_result = get_strategy(_RAW_ONLY).solve(nu, max_iter=4, rtol_energy_pct=_RTOL_PCT)
    return {
        'note': (
            'Generated from src/motronics/gauss/nonlinear/. Read by BOTH '
            'tests/unit/gauss/test_nonlinear_port_golden.py (does the Python still agree?) and '
            'rust/crates/gauss/tests/golden_nonlinear.rs (does the port agree?). See the Python '
            "test's docstring; the per-iteration traces are the point, not the final x."
        ),
        'rtol_energy_pct': _RTOL_PCT,
        'fixtures': fixtures,
        'nu_damped': {
            'k_seq': _NU_K_SEQ,
            'max_iter': 4,
            'folded': nu.folded,
            'trace': nu.trace,
            'iterations': int(nu_result.iterations),
            'converged': bool(nu_result.converged),
        },
    }


def _load() -> dict:
    # Under regeneration the golden is being REWRITTEN by a sibling test, and xdist runs the two
    # in different workers -- a reader that raced the writer read a truncated file. A regeneration
    # run is not a verdict about anything, so every reader stands down for it.
    if os.environ.get('MOTRONICS_REGEN_PORT_GOLDENS'):
        pytest.skip('regenerating: the golden is being rewritten by a sibling test')
    if not _GOLDEN.is_file():
        pytest.fail(
            f'{_GOLDEN} is missing. It is a committed artifact, not a cache -- regenerate it '
            f'deliberately with MOTRONICS_REGEN_PORT_GOLDENS=1 and commit the result.'
        )
    return load_golden(_GOLDEN)


def _same(got, want, where: str) -> None:
    """Compare one recorded number, `None` (no predecessor) included, at `repr` round-trip."""
    if want is None or got is None:
        assert got == want, f'{where}: {got} vs frozen {want}'
        return
    assert got == pytest.approx(want, rel=1e-15, abs=0.0), f'{where}: {got} vs frozen {want}'


# THE XFAIL IS GONE, 2026-09-07, AND IT WENT THE WAY ITS OWN LAST SENTENCE ASKED: it said "the day
# it agrees again this row must be deleted rather than left as a waiver nothing uses", and it
# agrees again. Recorded here rather than in a commit message alone, because the NEGATIVE results
# below are the part a future reader needs and no diff carries them.
#
# WHAT IT CLAIMED. Measured 2026-09-06 on the `integrate/main` tree: `contraction/anderson`
# trace[3].err 13.218666157097903 vs frozen 13.21866615709789 (8.0 ULP) and trace[3].increment
# 0.0527117739799137 vs 0.052711773979914694 (143 ULP), on the reasoning that two entries of one
# trace drifting by 8 and 143 ULP is not noise, so the arithmetic path had changed.
#
# WHAT IS MEASURED NOW, on the merged tree, and it is stronger than the two entries the row cited:
# the WORST disagreement anywhere in the golden -- every fixture, every strategy, every trace
# field, every x -- is **0.0 ULP**. Bit-identical, including both cited values. The freeze is
# whole again, and `nonlinear.json` was NOT regenerated: it is byte-identical to the file that
# arrived with `integrate/main` (verified by hash across base / both parents / the merge), so
# `rust/crates/gauss/tests/golden_nonlinear.rs` is judged against exactly the same target it was.
# That is the thing the row was most careful to protect.
#
# WHAT WAS REFUTED, each by a direct file-swap experiment on this tree rather than by argument.
# Both of the row's named suspects are wrong, and so is the obvious third:
#
#   * `gauss/linear/__init__.py` -- the row's PRIME SUSPECT (the singular-system residual guard on
#     eighteen call sites). Swapped in wholesale from `integrate/main`: the golden STILL agrees.
#   * `gauss/nonlinear/` -- `__init__.py` and `_strategies_picard.py` together, which is where the
#     one control-flow difference between the parents lives (`if np.isfinite(prev_err)` on this
#     lane against `elif` upstream, a change that genuinely can reach the iterates). Swapped in:
#     STILL agrees. Note the merged tree carries UPSTREAM's `elif`, so this is not the merge having
#     quietly undone their change -- that was checked first and is not what happened.
#   * multithreaded BLAS nondeterminism, the classic explanation for a few-ULP spread in a
#     least-squares reduction. Driven at OMP/OPENBLAS/MKL_NUM_THREADS of 1, 4 and 8: all three
#     produce the frozen values exactly.
#   * the four remaining `gauss/` files that differ between the parents (`linear/constraints.py`,
#     `linear/gauge.py`, `time_integrators/__init__.py`, `time_integrators/factor_reuse.py`),
#     swapped in as a group: STILL agrees.
#
# SO THE CAUSE IS OUTSIDE `gauss/`, which is the one thing the bisect the row asked for would have
# had to establish and is as far as negative evidence can take it. The fixtures are built from
# `contracts` objects, and this lane and `integrate/main` differ there too; whatever moved the
# iterates on their tree, the merge resolved that file to a version whose arithmetic matches the
# freeze. It is not re-opened here because there is nothing left to fix: the target is unchanged,
# the Rust reads the same bytes, and the freeze is enforced STRICTLY again from this commit -- a
# future drift reds on this test, with no waiver in front of it.
@pytest.mark.unit
def test_the_live_python_still_produces_the_frozen_trajectories() -> None:
    """The freeze, enforced. A behaviour change in `nonlinear/` reds HERE, not in silence.

    Tolerance is `1e-15` relative: the golden is the same code on the same platform, so the only
    admissible difference is JSON's own round-trip, and `repr`-precision floats round-trip exactly.
    A real change to an alpha schedule or a convergence rule moves things far above this.
    """
    if os.environ.get('MOTRONICS_REGEN_PORT_GOLDENS'):
        dump_golden(_GOLDEN, _build())
        pytest.skip(f'regenerated {_GOLDEN} -- commit it, then run again without the env var')

    golden = _load()
    live = _build()
    assert sorted(golden['fixtures']) == sorted(live['fixtures']), 'the fixture set moved'

    for fixture_name, want_fixture in golden['fixtures'].items():
        got_fixture = live['fixtures'][fixture_name]
        assert sorted(want_fixture['strategies']) == sorted(got_fixture['strategies']), fixture_name
        assert want_fixture['strategies'], f'{fixture_name}: the golden drives no strategy at all'
        for name, want in want_fixture['strategies'].items():
            got = got_fixture['strategies'][name]
            where = f'{fixture_name}/{name}'
            assert got['iterations'] == want['iterations'], f'{where}.iterations'
            assert got['converged'] == want['converged'], f'{where}.converged'
            assert len(got['trace']) == len(want['trace']), f'{where}: trace length'
            for k, (g, w) in enumerate(zip(got['trace'], want['trace'], strict=True)):
                for field in ('alpha', 'err', 'increment'):
                    _same(g[field], w[field], f'{where}.trace[{k}].{field}')
            for k, (g, w) in enumerate(zip(got['x'], want['x'], strict=True)):
                _same(g, w, f'{where}.x[{k}]')

    want_nu, got_nu = golden['nu_damped'], live['nu_damped']
    assert len(got_nu['folded']) == len(want_nu['folded']), 'nu_damped: fold count'
    for k, (g_row, w_row) in enumerate(zip(got_nu['folded'], want_nu['folded'], strict=True)):
        for j, (g, w) in enumerate(zip(g_row, w_row, strict=True)):
            _same(g, w, f'nu_damped.folded[{k}][{j}]')
    assert got_nu['iterations'] == want_nu['iterations']
    assert got_nu['converged'] == want_nu['converged']


@pytest.mark.unit
def test_the_golden_covers_every_registered_strategy() -> None:
    """A golden narrower than the thing it freezes is a scan with no floor.

    `ITERATION_REGISTRY` is the runtime's own list. If a strategy is added and the golden is not
    regenerated, the equivalence proof silently stops covering it.
    """
    golden = _load()
    covered = set(golden['fixtures']['contraction']['strategies']) | {_RAW_ONLY}
    assert covered == set(ITERATION_REGISTRY), (
        f'the golden covers {sorted(covered)} but the runtime registers '
        f'{sorted(ITERATION_REGISTRY)}. Regenerate, and port the new strategy on the Rust side.'
    )


@pytest.mark.unit
def test_the_fixtures_exercise_both_damping_answers() -> None:
    """The alpha ratchet must be pinned on BOTH sides of its own decision.

    A fixture set where alpha never left 1.0 would let a port that ignored every ratchet — the
    `meaningful` guard, the cap, the recovery leg, all of it — pass a golden that still looked
    complete. The measured defects this module exists to prevent (alpha pinned at 0.100 for 557 of
    605 iterations; a noise-driven ratchet costing 10x the iterations) are ALL in this half.
    """
    golden = _load()
    damped = set()
    for fixture_name, fixture in golden['fixtures'].items():
        alphas = [
            step['alpha'] for step in fixture['strategies']['picard-damped']['trace'] if step['alpha'] is not None
        ]
        assert alphas, f'{fixture_name}: picard-damped recorded no alpha at all'
        damped.add(min(alphas) < 1.0)
    assert damped == {True, False}, (
        f'every fixture answers the same way on damping ({damped}), so the ratchet logic is '
        f'untested: one fixture must leave alpha at 1.0 and one must pull it down'
    )


@pytest.mark.unit
def test_the_nu_damped_mix_is_the_hand_computed_one() -> None:
    """The frozen `nu-damped` fold history is checked against arithmetic, not only against itself.

    `k_seq = [1, 3, 2, 5]` gives energy deltas +8 (no PRIOR sign to flip against) then -5 (a flip,
    so alpha ratchets to 0.6 for the NEXT mix). The fourth mixed reluctivity is therefore
    `0.4*2 + 0.6*5 = 3.8`, and the source is twice it. A golden that merely records what the code
    did could not catch the code doing the wrong thing consistently.
    """
    folded = _load()['nu_damped']['folded']
    assert folded[:3] == [[1.0, 2.0], [3.0, 6.0], [2.0, 4.0]]
    assert folded[3] == pytest.approx([3.8, 7.6])
