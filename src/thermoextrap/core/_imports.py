from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # some weird stuff happens with sympy and pyright
    # this should stop those errors:
    from importlib import import_module

    sympy = import_module("sympy")
else:
    import sympy


@lru_cache
def module_available(module: str, minversion: str | None = None) -> bool:
    """Checks whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.

    Parameters
    ----------
    module : str
        Name of the module.
    minversion : str, optional
        Minimum version of the module

    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    from importlib.util import find_spec

    if find_spec(module) is None:
        return False

    if minversion is not None:
        import importlib.metadata

        from packaging.version import Version

        version = importlib.metadata.version(module)

        return Version(version) >= Version(minversion)

    return True


__all__ = ["module_available", "sympy"]
