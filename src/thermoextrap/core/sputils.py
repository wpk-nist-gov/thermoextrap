# pyright: reportMissingTypeStubs=false
"""Utilities for sympy."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, overload

# fix an issue with typing
from ._imports import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from sympy.core.expr import Expr
    from sympy.core.symbol import Symbol
    from sympy.tensor.indexed import (
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


def lambdify_with_defaults(
    args: Any,
    expr: Expr,
    modules: str | Sequence[str] | None = None,
    printer: Any = None,
    use_imps: bool = True,
    dummify: bool = True,
    cse: bool | Callable[..., Any] = True,
    docstring_limit: int | None = 10,
) -> Callable[..., Any]:
    """Interface to :func:`~sympy.utilities.lambdify.lambdify`"""
    return sp.lambdify(  # type: ignore[no-any-return]
        args=args,
        expr=expr,
        modules=modules,
        printer=printer,
        use_imps=use_imps,
        dummify=dummify,
        cse=cse,
        docstring_limit=docstring_limit,
    )
