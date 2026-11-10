"""PUBLIC-ROOT-ONLY: the `public-root` pre-push row refuses private history on a GitHub push.

The guard is `scripts/public_root.py`; each test plants a repo with a public root and a private second root.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest
from dev_lab.dev.githooks.runner import hook_table

_ROOT: Final = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'scripts'))
from public_root import refusals  # noqa: E402

_GITHUB: Final = 'https://github.com/DawnEver/numerics-lab.git'
_GITEA: Final = 'https://gitea.mingyangbao.site/Tianjie-Zou-Team/numerics-lab.git'
_ZERO: Final = '0' * 40
_GIT: Final = shutil.which('git') or 'git'


def _git(repo: Path, *args: str) -> str:
    config = ['-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=']
    return subprocess.run([_GIT, *config, *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name, encoding='utf-8')
    _git(repo, 'add', name)
    _git(repo, 'commit', '-q', '-m', name)
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def planted(tmp_path: Path) -> dict[str, str]:
    """A public root and tip, a private orphan root, and a merge reaching both roots."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'public')
    public_root = _commit(repo, 'public-root')
    public_tip = _commit(repo, 'public-tip')
    _git(repo, 'checkout', '-q', '--orphan', 'private')
    _git(repo, 'rm', '-rq', '--cached', '.')
    private = _commit(repo, 'private-data')
    _git(repo, 'checkout', '-q', '-f', 'public')
    _git(repo, 'merge', '-q', '--allow-unrelated-histories', '-m', 'merge private', 'private')
    merged = _git(repo, 'rev-parse', 'HEAD')
    return {'repo': str(repo), 'root': public_root, 'tip': public_tip, 'private': private, 'merged': merged}


def _push(planted: dict[str, str], url: str | None, sha: str) -> list[str]:
    line = f'refs/heads/x {sha} refs/heads/x {_ZERO}\n'
    return refusals(Path(planted['repo']), url, line, public_root=planted['root'])


def test_a_commit_not_descending_from_the_public_root_is_refused_on_github(planted: dict[str, str]) -> None:
    [message] = _push(planted, _GITHUB, planted['private'])
    assert 'does not descend from the public root' in message


def test_a_commit_reaching_a_second_root_is_refused_on_github(planted: dict[str, str]) -> None:
    [message] = _push(planted, _GITHUB, planted['merged'])
    assert f'reaches a second root ({planted["private"]})' in message


def test_a_gitea_push_is_not_judged(planted: dict[str, str]) -> None:
    assert _push(planted, _GITEA, planted['private']) == []
    assert _push(planted, _GITEA, planted['merged']) == []


def test_an_unnamed_destination_is_judged(planted: dict[str, str]) -> None:
    """Fail closed: a runner that forwards no url still refuses private history."""
    assert _push(planted, None, planted['private'])


def test_a_descending_commit_passes_on_github(planted: dict[str, str]) -> None:
    """The control: the guard refuses BY its rule, not every push."""
    assert _push(planted, _GITHUB, planted['tip']) == []


def test_a_deleted_ref_publishes_nothing(planted: dict[str, str]) -> None:
    line = f'(delete) {_ZERO} refs/heads/x {planted["tip"]}\n'
    assert refusals(Path(planted['repo']), _GITHUB, line, public_root=planted['root']) == []


def test_the_guard_is_a_declared_pre_push_row() -> None:
    rows = {hook.id: hook for hook in hook_table(_ROOT)['pre-push']}
    assert 'public-root' in rows, sorted(rows)
    assert list(rows['public-root'].run)[-1] == 'scripts/public_root.py'
    assert rows['public-root'].always_run
