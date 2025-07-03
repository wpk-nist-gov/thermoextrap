"""
Typing aliases (:mod:`thermoextrap.core.typing`)
================================================
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence  # noqa: F401
from typing import Any  # noqa: F401

import xarray as xr

from .typing_compat import TypeAlias, TypeVar

DataT = TypeVar("DataT", xr.DataArray, xr.Dataset, default=xr.DataArray)
DataT_ = TypeVar("DataT_", xr.DataArray, xr.Dataset, default=xr.DataArray)

XArrayObj: TypeAlias = "xr.DataArray | xr.Dataset"


MetaKws: TypeAlias = "Mapping[str, Any] | None"

SingleDim: TypeAlias = str
MultDims: TypeAlias = "str | Sequence[Hashable]"
