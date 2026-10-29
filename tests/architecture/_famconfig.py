"""numerics-lab's delta against each family config base -- data, computed by nothing.

The bases live in ``dev_lab.dev._famconfig_rows``; ``python -m dev_lab.dev.famfiles --deltas
tests/architecture/_famconfig.py`` turns base plus delta into the file. What numerics-lab adds is the
Rust half: the cargo build tree and the compiled extension the maturin develop build drops in place.
"""

from __future__ import annotations

from typing import Final

from dev_lab.dev.famconfig import Delta

__all__ = ['DELTAS', 'REPO']

REPO: Final = 'numerics-lab'

_GITIGNORE: Final = Delta(
    repo=REPO,
    # The cargo target directory and the native extension an editable build writes beside its sources.
    added=('/target/', '*.pyd', '*.so'),
    dropped={},
    ceiling=3,
)

DELTAS: Final[dict[str, Delta]] = {
    '.gitignore': _GITIGNORE,
    # Every tracked text file is LF in the tree on every OS, as when this code lived in its original repo.
    '.gitattributes': Delta(repo=REPO, added=('* text=auto eol=lf',), dropped={}, ceiling=1),
    '.rgignore': Delta(repo=REPO, added=(), dropped={}, ceiling=0),
    'Makefile': Delta(repo=REPO, added=(), dropped={}, ceiling=0),
}
