"""Utilities for sympy."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, overload

# fix an issue with typing
from ._imports import sympy as sp

if TYPE_CHECKING:
    from sympy.core.symbol import Symbol  # pyright: ignore[reportMissingTypeStubs]
    from sympy.tensor.indexed import (  # pyright: ignore[reportMissingTypeStubs]
        IndexedBase,
    )


@overload
def get_default_symbol(args: str) -> Symbol: ...
@overload
def get_default_symbol(*args: str) -> tuple[Symbol, ...]: ...


@lru_cache(100)
def get_default_symbol(*args: str) -> Symbol | tuple[Symbol, ...]:
    """Helper to get sympy symbols."""
    out = sp.symbols(args)
    if len(out) == 1:
        return out[0]
    return out


@overload
def get_default_indexed(args: str) -> IndexedBase: ...
@overload
def get_default_indexed(*args: str) -> tuple[IndexedBase, ...]: ...


@lru_cache(100)
def get_default_indexed(*args: str) -> IndexedBase | tuple[IndexedBase, ...]:
    """Helper to get sympy IndexBase objects."""
    out = tuple(sp.IndexedBase(key) for key in args)
    if len(out) == 1:
        return out[0]
    return out
