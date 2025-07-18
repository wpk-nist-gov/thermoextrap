# ruff: noqa: D102
"""
Typing aliases (:mod:`thermoextrap.core.typing`)
================================================
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence  # noqa: F401
from typing import TYPE_CHECKING, Any, Protocol, SupportsIndex, runtime_checkable

import xarray as xr

from .typing_compat import Self, TypeAlias, TypeVar

if TYPE_CHECKING:
    from cmomy.core.typing import Sampler

    # from thermoextrap.data import DataCallbackABC, DataSelector

DataT = TypeVar("DataT", xr.DataArray, xr.Dataset, default=xr.DataArray)
DataT_ = TypeVar("DataT_", xr.DataArray, xr.Dataset, default=xr.DataArray)

XArrayObj: TypeAlias = "xr.DataArray | xr.Dataset"


MetaKws: TypeAlias = "Mapping[str, Any] | None"

SingleDim: TypeAlias = str
MultDims: TypeAlias = "str | Sequence[Hashable]"

T_co = TypeVar("T_co", covariant=True)
# DerivsArgs: TypeAlias = "tuple[XArrayObj | DataSelector[xr.DataArray] | DataSelector[xr.Dataset], ...]"   # more specific, but not needed.
DerivsArgs: TypeAlias = "tuple[Any, ...]"


@runtime_checkable
class SupportsGetItem(Protocol[T_co]):
    """Protocol for thing that container that supports __getitem__"""

    def __getitem__(self, index: SupportsIndex, /) -> T_co: ...


@runtime_checkable
class SupportsDataProtocol(Protocol[T_co]):
    """Protocol for Data"""

    # @property
    # def meta(self) -> DataCallbackABC: ...
    # @property
    # def umom_dim(self) -> SingleDim: ...
    # @property
    # def deriv_dim(self) -> SingleDim | None: ...
    # @property
    # def x_is_u(self) -> bool: ...
    @property
    def order(self) -> int: ...
    # @property
    # def central(self) -> bool: ...
    @property
    def xalpha(self) -> bool: ...
    @property
    def derivs_args(self) -> DerivsArgs: ...

    def resample(self, sampler: Sampler) -> Self: ...


# TODO(wpk): remove when rework PerturbModel
@runtime_checkable
class SupportsDataPerturbModel(Protocol[T_co]):
    """Data protocol for PerturbModel"""

    @property
    def uv(self) -> T_co: ...
    @property
    def xv(self) -> T_co: ...
    @property
    def rec_dim(self) -> str: ...

    def resample(self, sampler: Sampler) -> Self: ...


@runtime_checkable
class SupportsModelProtocol(Protocol[T_co]):
    """Protocol for single model."""

    @property
    def alpha0(self) -> float: ...
    @property
    def order(self) -> int: ...

    # @property
    # def data(self) -> SupportsDataProtocol[T_co]: ...
    # @property
    # def derivatives(self) -> Derivatives: ...

    # def derivs(self, *args: Any, **kwargs: Any) -> T_co: ...
    # def coefs(self, *args: Any, **kwargs: Any) -> T_co: ...
    def predict(self, *args: Any, **kwargs: Any) -> T_co: ...
    def resample(self, sampler: Sampler, **kwargs: Any) -> Self: ...


@runtime_checkable
class SupportsModelProtocolDerivs(SupportsModelProtocol[T_co], Protocol[T_co]):
    """Protocol for single model with derivs"""

    def derivs(self, *args: Any, **kwargs: Any) -> T_co: ...


SupportsModelProtocolT = TypeVar(
    "SupportsModelProtocolT", bound=SupportsModelProtocol["xr.DataArray | xr.Dataset"]
)
SupportsModelProtocolDerivsT = TypeVar(
    "SupportsModelProtocolDerivsT",
    bound=SupportsModelProtocolDerivs["xr.DataArray | xr.Dataset"],
)
