"""THE TEST SUITE'S ONE NAME -> OBJECT BOUNDARY, for a test whose SUBJECT is a name.

Reflection is banned in `tests/` (`tests/architecture/ratchets/test_getattr_debt_only_shrinks.py`).
Most tests that reached for it were reading a declared field and now read it directly. What is left
is a test whose subject IS a name held as DATA -- "every symbol the source imports from a library
exists", "this module no longer binds the retired name" -- and for those the name is resolved here,
through the stdlib's own `module:attr` resolver (the one console scripts load through), so no test
carries `getattr`, `hasattr` or `vars` of its own.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint

__all__ = ['binds', 'bound']


def bound(module: str, name: str) -> object:
    """What *module* binds to *name* (a dotted ``Class.attr`` path is allowed); ``AttributeError`` if nothing."""
    return EntryPoint(name=name, value=f'{module}:{name}', group='tests').load()


def binds(module: str, name: str) -> bool:
    """Whether *module* binds *name* at all."""
    try:
        bound(module, name)
    except AttributeError:
        return False
    return True
