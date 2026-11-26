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
    # The native extension an editable build writes beside its sources (the base carries the cargo target).
    added=('*.pyd', '*.so'),
    dropped={},
    ceiling=2,
)

DELTAS: Final[dict[str, Delta]] = {
    '.gitignore': _GITIGNORE,
    # The base now carries '* text=auto eol=lf' itself; this repo adds nothing.
    '.gitattributes': Delta(repo=REPO, added=(), dropped={}, ceiling=0),
}
