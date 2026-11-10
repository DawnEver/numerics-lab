"""PUBLIC-ROOT-ONLY: a push to GitHub publishes only history descending from the public root, and no second root.

numerics-lab's public history on GitHub starts at the parentless commit `PUBLIC_ROOT`; the Gitea history
before it holds private business data and must never be published (integrator guard D78/D86, 2026-10-29).
Declared as the `public-root` pre-push row in pyproject.toml, so it runs with the family hooks wherever
they are installed. Git's ref lines arrive on stdin; the remote url is git's second hook argument.

When no url is given (a runner that does not forward git's arguments) the push is JUDGED: the guard fails
closed, since an unnamed destination may be GitHub.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

__all__ = ['PUBLIC_ROOT', 'judged', 'main', 'refusals']

PUBLIC_ROOT: Final = '02cb9d2a3c0dbcf1b5128161508b8c70b8b09fdc'
_ZERO: Final = '0' * 40
_GIT: Final = shutil.which('git') or 'git'


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([_GIT, *args], cwd=root, capture_output=True, text=True, check=False)


def judged(url: str | None) -> bool:
    """Whether a push to *url* is judged: GitHub, or a destination nobody named."""
    return url is None or 'github.com' in url


def refusals(root: Path, url: str | None, ref_lines: str, public_root: str = PUBLIC_ROOT) -> list[str]:
    """Why each pushed ref is refused; empty when the push may proceed."""
    if not judged(url):
        return []
    found: list[str] = []
    for line in ref_lines.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        local_ref, local_sha = fields[0], fields[1]
        if local_sha == _ZERO:
            continue
        if _git(root, 'merge-base', '--is-ancestor', public_root, local_sha).returncode != 0:
            found.append(
                f'refused: {local_ref} ({local_sha}) does not descend from the public root {public_root} -- '
                'it carries the private Gitea history. Port the change onto GitHub integrate/main instead.'
            )
            continue
        roots = _git(root, 'rev-list', '--max-parents=0', local_sha).stdout.split()
        others = [sha for sha in roots if sha != public_root]
        if others:
            found.append(
                f'refused: {local_ref} reaches a second root ({" ".join(others)}), i.e. private history merged in.'
            )
    return found


def main(argv: Sequence[str] | None = None) -> int:
    """Pre-push entry: ``[remote [url]]`` as git passes them, ref lines on stdin."""
    args = list(sys.argv[1:] if argv is None else argv)
    url = args[1] if len(args) > 1 else None
    root = Path(_git(Path.cwd(), 'rev-parse', '--show-toplevel').stdout.strip() or '.')
    found = refusals(root, url, '' if sys.stdin.isatty() else sys.stdin.read())
    for message in found:
        sys.stderr.write(f'{message}\n')
    return 1 if found else 0


if __name__ == '__main__':
    raise SystemExit(main())
