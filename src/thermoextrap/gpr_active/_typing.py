from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from xarray import DataArray

    from thermoextrap.core.typing import NDArrayAny, SupportsModelDerivs


@runtime_checkable
class SupportsDataWrapper(Protocol):
    """Data wrapper protocol."""

    @property
    def beta(self) -> float: ...

    def get_data(
        self, *args: Any, **kwargs: Any
    ) -> tuple[DataArray, DataArray, NDArrayAny]: ...

    def build_state(
        self, *args: Any, **kwargs: Any
    ) -> SupportsModelDerivs[DataArray]: ...


@runtime_checkable
class SupportsSimulation(Protocol):
    """Simulation protocol."""

    def run_sim(self, *args: Any, **kwargs: Any) -> SupportsDataWrapper: ...
