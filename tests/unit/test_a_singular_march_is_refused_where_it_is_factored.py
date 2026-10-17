"""A singular MNA is refused by its actual physical or numerical authority.

The CLI rejects underdetermined initial source currents before marching, so
that refusal must precede every FactorCache solve. An independent guard test
passes the same real four-unknown MNA weighted tangent directly to the cache: its
first miss must refuse a singular factor with order/residual/backend evidence.
The one-source sibling remains the well-posed floor through the real CLI.
"""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
from tests.bound_names import binds, bound

import motronics.__main__ as cli
import motronics.gauss.linear as gauss_linear
from motronics.gauss.formulation import DerivativeWeights
from motronics.gauss.linear import LinearSolveError
from motronics.gauss.time_integrators.factor_reuse import FactorCache
from motronics.laplace.circuit.cases.case_library import load_circuit_case

#: The one line the CLI prints that a user can act on. Read off stdout, because that is the surface.
_REPORT_PREFIX = 'report:'

_MANIFEST = """
    [node]
    kind = "network"
    form = "circuit"
    name = "{name}"
    description = "{description}"

    [payload]
    circuit = "netlist.toml"
"""

_GOLDEN = """
    [meta]
    case = "{name}"
    generator = "none"
    description = "{description}"

    [probe]
    kind = "node_voltage"
    target = "a"
"""

_STUDY_MANIFEST = """
    [node]
    kind = "study"
    name = "{name}"
    description = "Operating point for the {name} circuit node."

    [payload]
    study = "study.toml"
"""

_STUDY = """
    [network]
    members = [{{ name = "{name}", form = "circuit" }}]

    [scenario.nominal]
    kind = "tran"
    tstep = 1.0e-5
    tstop = 5.0e-3
"""

#: An IDEAL source across ``a``-``gnd``. Two of them make the MNA exactly rank-deficient: the two
#: branch-current unknowns enter every row identically, so no finite pair of currents is determined.
#: A resistor is present in both nets so the singularity is the SOURCES and not an orphan node.
_SOURCE = """
    [[component]]
    name = "{name}"
    type = "voltage_source"
    nodes = ["a", "gnd"]
    value = 1.0
"""

_NETLIST = """
    [meta]
    name = "{name}"
    description = "{description}"
    ground = "gnd"
{sources}
    [[component]]
    name = "R1"
    type = "resistor"
    nodes = ["a", "b"]
    value = 1000.0

    [[component]]
    name = "C1"
    type = "nonlinear_capacitor"
    nodes = ["b", "gnd"]

    [component.params]
    law = "linear"
    C = 1.0e-6
"""


def _plant(root: Path, name: str, n_sources: int, description: str) -> Path:
    """Write a runnable `network`/`circuit` node plus the study that binds it. Returns the node dir."""
    case = root / 'network' / 'circuit' / name
    study = root / 'study' / 'network' / 'circuit' / name
    case.mkdir(parents=True)
    study.mkdir(parents=True)
    fields = {'name': name, 'description': description}
    # The sources keep the template's indentation and the WHOLE netlist is dedented once, so the
    # planted TOML has one uniform margin rather than two.
    sources = ''.join(_SOURCE.format(name=f'V{i + 1}') for i in range(n_sources))
    (case / 'motronics.toml').write_text(textwrap.dedent(_MANIFEST).format(**fields), encoding='utf-8')
    (case / 'golden.toml').write_text(textwrap.dedent(_GOLDEN).format(**fields), encoding='utf-8')
    (case / 'netlist.toml').write_text(textwrap.dedent(_NETLIST.format(sources=sources, **fields)), encoding='utf-8')
    (study / 'motronics.toml').write_text(textwrap.dedent(_STUDY_MANIFEST).format(name=name), encoding='utf-8')
    (study / 'study.toml').write_text(textwrap.dedent(_STUDY).format(name=name), encoding='utf-8')
    return case


@pytest.fixture
def planted(tmp_path: Path) -> dict[str, Path]:
    """The singular node and its well-posed sibling, under ONE root so both resolve the same way.

    They differ in exactly one component. A fixture that planted only the failing half could be
    satisfied by a guard that refuses every circuit in the library.
    """
    return {
        'singular': _plant(
            tmp_path, 'two_ideal_sources', 2, 'Two ideal sources on one node pair: an exactly singular MNA.'
        ),
        'well_posed': _plant(
            tmp_path, 'one_ideal_source', 1, 'The same net with ONE source: well-posed, and it must still march.'
        ),
    }


@pytest.fixture
def solve_counts(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count both march entry points at once. Counting only one cannot tell a bypass from silence."""
    counts = {'factor_cache': 0, 'seam': 0}
    cache_solve = FactorCache.solve
    seam = gauss_linear.solve_linear_system

    def counted_cache(self, key, build_matrix, rhs):
        counts['factor_cache'] += 1
        return cache_solve(self, key, build_matrix, rhs)

    def counted_seam(*args, **kwargs):
        counts['seam'] += 1
        return seam(*args, **kwargs)

    monkeypatch.setattr(FactorCache, 'solve', counted_cache)
    # The name is rebound in every module that IMPORTED it: `from x import y` binds `y` into the
    # importer at import time, so patching only the definition site would count zero and read as
    # "nothing calls the seam" -- which happens to be the finding, and would then be unfalsifiable.
    for module_name, module in list(sys.modules.items()):
        if (
            isinstance(module, types.ModuleType)
            and binds(module_name, 'solve_linear_system')
            and bound(module_name, 'solve_linear_system') is seam
        ):
            monkeypatch.setattr(module, 'solve_linear_system', counted_seam)
    return counts


def _samples_from(out: str) -> int:
    """The ``n=`` the CLI printed -- an INDEPENDENT total for the solve count to be reconciled against."""
    for line in out.splitlines():
        for token in line.split():
            if token.startswith('n='):
                return int(token[2:])
    pytest.fail(f'the run printed no sample count to reconcile against:\n{out}')
    raise AssertionError  # pragma: no cover - pytest.fail does not return


@pytest.mark.unit
def test_a_singular_mna_is_refused_naming_its_initial_physical_authority(planted, capsys) -> None:
    """The real node lacks unique initial ideal-source current authority."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(['case', 'run', str(planted['singular'])])
    assert excinfo.value.code == 2, 'a node that cannot be marched must exit non-zero'

    # BOTH streams: the refusal goes to stderr and the success line to stdout, and a test that
    # read only one would call an unprinted message a printed one.
    captured = capsys.readouterr()
    message = captured.out + captured.err
    assert 'two_ideal_sources' in message, 'the refusal must name the node the user asked for'
    assert 'initial state/rate equations require additional physical authority' in message, message
    assert 'array must not contain infs or NaNs' not in message, (
        "that is scipy's `check_finite` firing one solve DOWNSTREAM of the cause, on a right-hand "
        'side the previous solve poisoned. It names no order, no statistic and no remedy.'
    )


@pytest.mark.unit
def test_initial_physical_refusal_precedes_every_march_linear_solve(planted, solve_counts) -> None:
    """Physical initial admission refuses before any numerical march factorization."""
    with pytest.raises(SystemExit):
        cli.main(['case', 'run', str(planted['singular'])])
    assert solve_counts['factor_cache'] == 0, 'an unadmitted physical initial pair must not reach the march cache'
    assert solve_counts['seam'] == 0, 'an unadmitted physical initial pair must not reach the march linear seam'


@pytest.mark.unit
def test_actual_singular_mna_tangent_is_refused_on_the_first_factor_cache_miss(planted):
    """Judge the cache independently of the earlier physical initial refusal."""
    dae = load_circuit_case(planted['singular'], point='nominal').assemble()
    zero = np.zeros(4)
    weights = DerivativeWeights(1, 1.0e5, 0)
    tangent = dae.jacobian(zero, zero, zero, 0.0, weights)
    matrix = tangent.toarray() if hasattr(tangent, 'toarray') else np.asarray(tangent)
    assert matrix.shape == (4, 4)
    rhs = -dae.residual(zero, zero, zero, 0.0)
    cache = FactorCache()
    key = dae.factor_reuse_key(zero, zero, zero, 0.0, weights)
    with pytest.raises(LinearSolveError) as refusal:
        cache.solve(key, lambda: matrix, rhs)
    message = str(refusal.value)
    assert 'n=4' in message
    assert 'residual =' in message
    assert 'factor_reuse' in message
    assert cache.solves == 1


@pytest.mark.unit
def test_the_well_posed_sibling_still_marches_and_stays_on_the_unguarded_entry(planted, solve_counts, capsys) -> None:
    """THE FLOOR, and the bypass measurement in one run.

    Two independent floors, because each alone is satisfiable by an accident:

    * the sibling MARCHES -- a guard that refused every circuit would pass the test above;
    * its FactorCache count is RECONCILED against the sample count the CLI printed. A march cannot
      advance N steps with fewer than N linear solves, so a probe that silently patched nothing
      (count 0) is refused by arithmetic rather than by a threshold somebody chose.
    """
    assert cli.main(['case', 'run', str(planted['well_posed'])]) == 0, 'the well-posed sibling must still solve'
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert _REPORT_PREFIX in out, 'a successful run must announce a report the user can open'

    samples = _samples_from(out)
    assert samples > 1, f'a one-sample march reconciles nothing; the CLI reported n={samples}'
    # `samples - 1` is the STEP count: the first sample is the initial condition and is not solved
    # for. Measured 500 solves against n=501, i.e. exactly one solve per step and no spare -- so
    # this is an equality dressed as a floor, and a probe that patched nothing (0) cannot reach it.
    assert solve_counts['factor_cache'] >= samples - 1, (
        f'{solve_counts["factor_cache"]} FactorCache solves for {samples - 1} marched steps. A march '
        'cannot advance a step without solving for it, so this probe is measuring the wrong '
        'thing -- most likely it patched a name the march does not call.'
    )
    assert solve_counts['seam'] == 0, (
        f'{solve_counts["seam"]} solves reached gauss.linear. That would be GOOD news, but it '
        f'contradicts the 2026-09-06 measurement this file is built on ({solve_counts["factor_cache"]} '
        'FactorCache solves against zero seam solves over seven library nodes). If the march was '
        'rerouted through the guarded seam, the FactorCache row in '
        'scripts/gate/linear_sweep.py::UNGUARDED_DIRECT_SITES is stale and must go with it.'
    )
