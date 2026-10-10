"""A singular MNA is refused BY NAME through `motronics case run`, at the seam that factored it.

# The gap this pins, and why nothing was red while it was open

`gauss.linear.solve_linear_system` is the guarded seam: it computes a relative residual, and above
the trigger it takes one step of iterative refinement and RAISES with the order, both statistics
and a remedy. :class:`~motronics.gauss.time_integrators.factor_reuse.FactorCache` is the march's
OTHER entry point, taken whenever the problem exposes ``factor_reuse_key`` -- and it returned
``self._apply(self._factor, b)`` unexamined. One march, two entry points, one of them guarded, and
which one a case gets turns on a capability the case never declares.

MEASURED 2026-09-06 on this tree, through `motronics.__main__.main`, over seven
`cases/network/circuit` nodes: **595,898 FactorCache solves and ZERO reaching `gauss.linear`**.
The two numbers are quoted together deliberately -- a zero beside a zero is a broken probe, a zero
beside 595,898 is a finding. So every shipped circuit march is on the unguarded side.

No shipped node trips the criterion (worst relative residual measured over a full
a `boost`'s `closed_pi` march: 5.8e-11 over 532,365 judged solves, nothing over the 1e-6 trigger). A
SINGULAR one does, and before this file the user saw:

    LinAlgWarning: Diagonal number 4 is exactly zero. Singular matrix.
    motronics: two_ideal_sources: the circuit chain cannot march this node -- array must not
    contain infs or NaNs

-- no order, no statistic, no remedy, and nothing naming the singularity. That text is quoted from
this exact fixture with the guard reverted, which is also this file's revert check. Worse, the
``ValueError`` behind it is raised on the SECOND solve, by `scipy`'s ``check_finite`` on a
right-hand side the FIRST solve had already poisoned: the failure was reported one step
downstream of its cause, which is why reading the message sent nobody to the singularity.

REVERTED-AND-RUN, 2026-09-06: with the guard removed the first two tests below go red (the second
on `2 FactorCache solves` instead of 1 -- the second solve being the poisoned one) and the third,
the FLOOR, stays green. That split is the point: a floor that went red with the guard would be
testing the guard twice.

# Where the guard went, and what it deliberately does NOT cover

The MISS path only -- the branch that calls ``build_matrix()``. That is the one place the matrix is
already in hand, so judging costs one matvec against a factorization the branch just paid
O(n^3) for.

A HIT is left unjudged ON PURPOSE, and the price of the alternative is measured rather than
guessed. Over the same seven nodes, 211,388 of the 595,898 solves (35%) were hits, and for
`class_e` 123,208 of 129,749 (95%) were. A hit does not have the matrix -- `FactorCache` exists
precisely so it never builds one -- so guarding it means calling ``build_matrix()`` on every hit,
i.e. reinstating exactly the work the cache removes, on a third of every circuit march in the
library and on 95% of one. That is not a guard, it is a deletion of the feature, and it is not
taken here. What makes the omission safe rather than convenient is the failure's own shape,
which this file asserts: the singular system is refused on solve #1, which is always a MISS.

# The floors

A guard that refuses everything would satisfy the refusal assertion. So the well-posed sibling --
the SAME netlist with one source instead of two -- is run in the same fixture and must still
march, and its FactorCache count is reconciled against the sample count the CLI itself prints.
"""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path

import pytest
from tests.bound_names import binds, bound

import motronics.__main__ as cli
import motronics.gauss.linear as gauss_linear
from motronics.gauss.time_integrators.factor_reuse import FactorCache

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
def test_a_singular_mna_is_refused_naming_its_order_statistic_and_seam(planted, capsys) -> None:
    """The failing half. The refusal must be about SINGULARITY, at the seam that actually factored."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(['case', 'run', str(planted['singular'])])
    assert excinfo.value.code == 2, 'a node that cannot be marched must exit non-zero'

    # BOTH streams: the refusal goes to stderr and the success line to stdout, and a test that
    # read only one would call an unprinted message a printed one.
    captured = capsys.readouterr()
    message = captured.out + captured.err
    assert 'two_ideal_sources' in message, 'the refusal must name the node the user asked for'
    assert 'singular' in message, f'the refusal never says the system is singular:\n{message}'
    assert 'n=4' in message, f'the refusal names no ORDER, so the reader cannot locate the system:\n{message}'
    assert 'residual =' in message, f'the refusal quotes no STATISTIC, so it cannot be checked:\n{message}'
    assert 'factor_reuse' in message, (
        'the refusal must name the seam that FACTORED this system. `factor_reuse` is the entry '
        "point the march took; naming gauss.linear's `direct` would send the reader to code that "
        f'never ran:\n{message}'
    )
    assert 'array must not contain infs or NaNs' not in message, (
        "that is scipy's `check_finite` firing one solve DOWNSTREAM of the cause, on a right-hand "
        'side the previous solve poisoned. It names no order, no statistic and no remedy.'
    )


@pytest.mark.unit
def test_the_refusal_lands_on_the_first_solve_which_is_a_factorisation(planted, solve_counts) -> None:
    """WHY THE HIT PATH NEED NOT BE GUARDED, asserted rather than asserted-in-prose.

    The guard sits on the MISS branch only, because that is the one holding the matrix. That is
    sound only if a singular system is caught before any hit can consume its output -- i.e. on the
    very first solve, which is always a miss (the cache starts empty). This pins that.
    """
    with pytest.raises(SystemExit):
        cli.main(['case', 'run', str(planted['singular'])])
    assert solve_counts['factor_cache'] == 1, (
        f'the singular march took {solve_counts["factor_cache"]} FactorCache solves. It must be '
        'refused on the FIRST one -- a later refusal means a hit already back-substituted against '
        'a singular factor, and the miss-only guard would then be leaving a real window open.'
    )


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
