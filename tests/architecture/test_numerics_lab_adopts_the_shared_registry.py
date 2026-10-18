"""numerics-lab ADOPTS the family rules registry: every rule is enforced here by a named mechanism, or absent.

``lab_commons.dev.rules`` holds the one canonical statement of each rule; ``assert_adopted`` resolves every
rule claimed here against a TRACKED mechanism in this tree and refuses a rule neither claimed nor declared
absent. It proves the accounting, not that the mechanisms pass -- that is the suite's business.

numerics-lab is a LIBRARY on the family branch flow: lanes merge into ``integrate/main``, and ``main`` moves
by ``integrate promote`` on a full forge PASS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.profile import RepoProfile
from lab_commons.dev.rules import Adoption, LintRule, TestPath, assert_adopted

_ROOT: Final = Path(__file__).resolve().parents[2]

_PROFILE: Final = RepoProfile(
    app_name='numerics_lab', package='numerics_lab', root=_ROOT, lint_config=_ROOT / 'ruff.toml'
)

_BRANCHES: Final = TestPath('tests/architecture/test_the_branches_are_the_declared_set.py')
_HOOKS: Final = TestPath('tests/architecture/test_the_hooks_are_the_family_base.py')

_MECHANISMS: Final = {
    'PROJECT-FILES-HAVE-ONE-SOURCE': (TestPath('tests/architecture/test_the_family_config_is_rendered.py'),),
    'RETIRED-NAMES-REGISTERED': (TestPath('tests/architecture/test_retired_names_stay_retired.py'),),
    'FORMATS-UPGRADE-ON-READ': (TestPath('tests/unit/test_an_older_checkpoint_file_upgrades_on_read.py'),),
    'NO-LAZY-IMPORT': (LintRule(code='PLC0415'),),
    'ONE-BRANCH-PER-SESSION': (_BRANCHES,),
    'MERGE-OR-DELETE': (_BRANCHES,),
    'MERGE-DEVIATIONS-NAMED': (_HOOKS,),
    'PUBLISH-IS-THE-LOCAL-REF': (_HOOKS,),
    'SESSION-HISTORY-ONLY-GROWS': (_HOOKS,),
    'INTEGRATION-IS-MERGES-ONLY': (_HOOKS,),
    'LATEST-DEPENDENCIES': (_HOOKS,),
}

#: Rules with no mechanism in this tree today, BY NAME -- deriving it from ``RULES`` would silently absorb a
#: rule added upstream instead of making this repo decide. A rule leaves by gaining a mechanism above.
_ABSENT: Final = frozenset({
    # Enforced by the kit's own commands on this repo's declaration, with no test of this tree's: forge,
    # verdicts, integration and promotion.
    'VERDICT-BAR-IS-THE-INCREMENT', 'NETWORK-RETRY-THEN-REPORT', 'SHARED-CHECKOUT', 'FORGE-THROUGH-THE-DOOR',
    'VERDICT-AS-STATUS', 'ISSUE-IS-INTENT', 'AUTO-MODE-RUNS-THE-DOORS', 'VERDICTS-FROM-THE-FORGE',
    'MAIN-SESSION-PUBLISHES', 'SUBAGENT-NO-PUSH', 'WORKTREES-STAY-INSIDE', 'ONE-BOX-ONE-LOCK',
    'DOWNSTREAM-SYNCS-INTEGRATION', 'PRS-TARGET-INTEGRATION', 'TRUNK-MOVES-BY-FAST-FORWARD',
    'ONE-WRITER-PER-SESSION-BRANCH',
    # The environment: shims are installed per box, and no agent guard is rendered here yet.
    'HOOKS-ARE-WIRED', 'INSTALL-DOOR-DELIVERS-THE-DECLARATION', 'ENV-MUTATION-THROUGH-THE-DOOR', 'AGENT-GUARD',
    # Domain and measurement rules with no subject in a pure numerics engine.
    'IMPLEMENT-EVERYTHING', 'BAR-IS-A-CONSTANT', 'UNITS-GO-THROUGH-PINT', 'TOLERANCE-CARRIES-A-UNIT',
    'REGISTRY-OWNS-THE-DECISION', 'PRODUCTION-ENTRY-POINT', 'UNSUPPORTED-RAISES', 'REFUSAL-NAMES-THE-REMEDY',
    'NO-REFLECTION', 'PUBLIC-SURFACE-DECLARED', 'FIX-THE-CAUSE',
    # Docs and memory: no rules pages, injected docs or dated-memory guard in this repo yet.
    'DOCS-SPLIT', 'MEMORY-SHAPE', 'INJECTED-DOC-WIDTH-CEILING', 'INJECTED-TEXT-IS-PROGRESSIVE',
    'NO-CJK-IN-TRACKED-SOURCE', 'REPORTS-LIVE-IN-MEMORY',
    # Code placement and scratch: one python/ and one tests/ tree, no scratch.
    'CODE-IN-CODE-ROOTS', 'SCRATCH-ARCHIVED-OR-PROMOTED',
    # Meta-rules about guards: carried by the family's shared bodies.
    'DECLARATION-LIES', 'PLANTED-CONTROL', 'FLOOR-ON-EVERY-SCAN', 'ESCAPE-HATCH-CEILING', 'RATCHET-TWO-SIDES',
    'NAMED-SETS-NOT-COUNTS', 'XFAIL-NOT-SKIP', 'MODULE-SIZE-ALARM',
})  # fmt: skip

#: The size of the absent set at adoption (2026-10-18); lowering it is the only legal edit.
_ABSENT_CEILING: Final = 47


def test_every_rule_is_enforced_or_declared_absent() -> None:
    assert_adopted(_PROFILE, Adoption(app_name='numerics_lab', mechanisms=_MECHANISMS, declared_absent=_ABSENT))


def test_the_absent_set_does_not_grow() -> None:
    assert len(_ABSENT) <= _ABSENT_CEILING, sorted(_ABSENT)


def test_no_rule_is_both_enforced_and_absent() -> None:
    assert not (_ABSENT & frozenset(_MECHANISMS))
