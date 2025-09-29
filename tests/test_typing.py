"""Check/test typing support"""
# pyright: reportUnreachable=false

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, SupportsFloat, SupportsIndex, SupportsInt

import cmomy
import numpy as np
import xarray as xr

import thermoextrap as xtrap

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

if TYPE_CHECKING:
    import sympy as sp  # pyright: ignore[reportMissingTypeStubs]
    from cmomy.core.typing_compat import TypeVar

    from thermoextrap.core.typing import (
        SupportsModel,
        SupportsModelT,
    )
    from thermoextrap.models import StateCollection

    T = TypeVar("T")


# * Parameters
val_array = np.zeros((10,), dtype=np.float64)
val_dataarray = xr.DataArray(val_array, dims=("val",))
val_dataset = xr.Dataset({"values": val_dataarray})

data_array = np.zeros((10, 2, 3), dtype=np.float64)
data_dataarray = xr.DataArray(data_array, dims=("val", "xmom", "umom"))
data_dataset = xr.Dataset({"data": data_dataarray})

dxduave_dataarray = cmomy.wrap(data_dataarray, mom_ndim=2)
dxduave_dataset = cmomy.wrap(data_dataset, mom_ndim=2, mom_dims=("xmom", "umom"))


datacentralmoments_dataarray = xtrap.data.DataCentralMoments(dxduave_dataarray)
datacentralmoments_dataset = xtrap.data.DataCentralMoments(dxduave_dataset)


# * DataSelector --------------------------------------------------------------
def check_dataselector(
    actual: T,
    klass: type[Any],
    data_type: type[Any],
) -> T:
    assert isinstance(actual, klass)

    assert isinstance(actual.data, data_type)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    return actual


def test_thermoextrap_data_dataselector() -> None:
    dims = "umom"
    check_dataselector(
        assert_type(
            xtrap.data.DataSelector(data_dataarray, dims),
            xtrap.data.DataSelector[xr.DataArray],
        ),
        xtrap.data.DataSelector,
        xr.DataArray,
    )
    check_dataselector(
        assert_type(
            xtrap.data.DataSelector.from_defaults(data_dataarray, mom_dim=dims),
            xtrap.data.DataSelector[xr.DataArray],
        ),
        xtrap.data.DataSelector,
        xr.DataArray,
    )
    check_dataselector(
        assert_type(
            xtrap.data.DataSelector(data_dataset, dims),
            xtrap.data.DataSelector[xr.Dataset],
        ),
        xtrap.data.DataSelector,
        xr.Dataset,
    )
    check_dataselector(
        assert_type(
            xtrap.data.DataSelector[xr.Dataset].from_defaults(
                data_dataset, mom_dim=dims
            ),
            xtrap.data.DataSelector[xr.Dataset],
        ),
        xtrap.data.DataSelector,
        xr.Dataset,
    )


# * DataValues ----------------------------------------------------------------
def check_datavalues(
    actual: T,
    klass: type[Any],
    uv_type: type[Any],
    xv_type: type[Any],
) -> T:
    assert isinstance(actual, klass)

    assert isinstance(actual.uv, uv_type)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert isinstance(actual.xv, xv_type)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    return actual


def test_thermoextrap_data_datavalues() -> None:
    check_datavalues(
        assert_type(
            xtrap.data.DataValues(uv=val_dataarray, xv=val_dataarray, order=2),
            xtrap.DataValues[xr.DataArray],
        ),
        xtrap.DataValues,
        xr.DataArray,
        xr.DataArray,
    )
    check_datavalues(
        assert_type(
            xtrap.data.DataValues(uv=val_dataarray, xv=val_dataset, order=2),
            xtrap.DataValues[xr.Dataset],
        ),
        xtrap.DataValues,
        xr.DataArray,
        xr.Dataset,
    )


def test_thermoextrap_data_datavaluescentral() -> None:
    check_datavalues(
        assert_type(
            xtrap.data.DataValuesCentral(uv=val_dataarray, xv=val_dataarray, order=2),
            xtrap.DataValuesCentral[xr.DataArray],
        ),
        xtrap.DataValuesCentral,
        xr.DataArray,
        xr.DataArray,
    )
    check_datavalues(
        assert_type(
            xtrap.data.DataValuesCentral(uv=val_dataarray, xv=val_dataset, order=2),
            xtrap.DataValuesCentral[xr.Dataset],
        ),
        xtrap.DataValuesCentral,
        xr.DataArray,
        xr.Dataset,
    )


# * DataCentralMoments --------------------------------------------------------
def check_datacentralmoments(
    actual: T,
    klass: type[Any],
    obj_type: type[Any],
) -> T:
    assert isinstance(actual, klass)

    assert isinstance(actual.dxduave, cmomy.CentralMomentsData)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert isinstance(actual.dxduave.obj, obj_type)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    return actual


def test_thermoextrap_data_datacentralmoments_dataarray() -> None:
    dim = "val"
    # DataArray
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments(dxduave_dataarray),
            xtrap.data.DataCentralMoments[xr.DataArray],
        ),
        xtrap.data.DataCentralMoments,
        xr.DataArray,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_raw(data_dataarray),
            xtrap.data.DataCentralMoments[xr.DataArray],
        ),
        xtrap.data.DataCentralMoments,
        xr.DataArray,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_vals(
                val_dataarray, xv=val_dataarray, order=2, dim=dim
            ),
            xtrap.data.DataCentralMoments[xr.DataArray],
        ),
        xtrap.data.DataCentralMoments,
        xr.DataArray,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_data(data_dataarray),
            xtrap.data.DataCentralMoments[xr.DataArray],
        ),
        xtrap.data.DataCentralMoments,
        xr.DataArray,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_resample_vals(
                uv=val_dataarray, xv=val_dataarray, order=2, sampler=2, dim=dim
            ),
            xtrap.data.DataCentralMoments[xr.DataArray],
        ),
        xtrap.data.DataCentralMoments,
        xr.DataArray,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_ave_raw(
                u=val_dataarray, xu=val_dataarray, umom_dim="val"
            ),
            xtrap.data.DataCentralMoments[xr.DataArray],
        ),
        xtrap.data.DataCentralMoments,
        xr.DataArray,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_ave_central(
                du=val_dataarray, dxdu=val_dataarray, umom_dim="val"
            ),
            xtrap.data.DataCentralMoments[xr.DataArray],
        ),
        xtrap.data.DataCentralMoments,
        xr.DataArray,
    )


def test_thermoextrap_data_datacentralmoments_dataset() -> None:
    dim = "val"
    # Dataset
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments(dxduave_dataset),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments[xr.Dataset].from_raw(data_dataset),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments[xr.Dataset].from_vals(
                val_dataarray, xv=val_dataset, order=2, dim=dim
            ),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments[xr.Dataset].from_data(data_dataset),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments[xr.Dataset].from_resample_vals(
                uv=val_dataarray, xv=val_dataset, order=2, sampler=2, dim=dim
            ),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments[xr.Dataset].from_ave_raw(
                u=val_dataset, xu=val_dataset, umom_dim="val"
            ),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments[xr.Dataset].from_ave_central(
                du=val_dataset, dxdu=val_dataset, umom_dim="val"
            ),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )


if TYPE_CHECKING:
    from collections.abc import Callable

    from thermoextrap.core.typing import (
        DataT,
        SupportsData,
        SupportsDataXU,
    )
    from thermoextrap.models import ExtrapModel, InterpModel

    def func_data(x: SupportsData[DataT]) -> int:
        return x.order

    def tester_data(
        a: xtrap.data.DataValues,
        b: xtrap.data.DataValuesCentral,
        c: xtrap.data.DataCentralMoments,
        d: xtrap.data.DataCentralMomentsVals,
        e: xtrap.data.DataCentralMomentsBase,
    ) -> None:
        assert_type(func_data(a), int)
        assert_type(func_data(b), int)
        assert_type(func_data(c), int)
        assert_type(func_data(d), int)
        assert_type(func_data(e), int)

    def func_models(x: SupportsModel[DataT]) -> float:
        return x.alpha0

    def tester_protocol(a: xtrap.models.ExtrapModel) -> None:
        assert_type(func_models(a), float)

    def func_dataperturbmodel(x: SupportsDataXU[DataT]) -> DataT:
        return x.xv

    def tester_dataperturbmodel(
        a: xtrap.data.DataValues,
        b: xtrap.data.DataValuesCentral,
        c: xtrap.data.DataValues[xr.Dataset],
        d: xtrap.data.DataValuesCentral[xr.Dataset],
    ) -> None:
        assert_type(func_dataperturbmodel(a), xr.DataArray)
        assert_type(func_dataperturbmodel(b), xr.DataArray)

        assert_type(func_dataperturbmodel(c), xr.Dataset)
        assert_type(func_dataperturbmodel(d), xr.Dataset)

    def func_statecollection(
        # factory_state: Callable[..., SupportsModelDerivsT]
        state: SupportsModelT,
        state_collection: StateCollection[SupportsModelT, xr.DataArray],
        factory_state_collection: Callable[
            ..., StateCollection[SupportsModelT, xr.DataArray]
        ],
    ) -> None:
        pass

    def tester_statecollection(
        state: ExtrapModel[xr.DataArray],
        state_collection: InterpModel[ExtrapModel[xr.DataArray], xr.DataArray],
        factory_state_collection: type[
            InterpModel[ExtrapModel[xr.DataArray], xr.DataArray]
        ],
    ) -> None:
        func_statecollection(state, state_collection, factory_state_collection)

    def func_supports(
        index: SupportsIndex,
        int_: SupportsInt,
        float_: SupportsFloat,
    ) -> None:
        pass

    def tester_supports(
        x_int: int,
        x_np_int: np.int_,
        x_sp_int: sp.Integer,
        x_float: float,
        x_np_float: np.float_,
        x_sp_float: sp.Float,
    ) -> None:
        func_supports(x_int, x_int, x_int)
        func_supports(x_np_int, x_np_int, x_np_int)
        func_supports(x_sp_int, x_sp_int, x_sp_int)
        func_supports(x_int, x_float, x_float)
        func_supports(x_np_int, x_np_float, x_np_float)
        func_supports(x_sp_int, x_sp_float, x_sp_float)

    from collections.abc import Iterable, Sequence

    from thermoextrap.core.typing_compat import Concatenate

    def func_concat(
        f: Callable[Concatenate[Sequence[int], ...], int], **kws: Any
    ) -> int:
        return f([1, 2, 3], **kws) + 2

    def tester_concat() -> None:
        def fa(x: Iterable[int]) -> int:
            return sum(x)

        func_concat(fa)

        def fb(x: Sequence[int], y: int) -> int:
            return sum(x) + y

        func_concat(fb, y=2)

        # def fc(x: list[int]) -> int:
        #     return sum(x)

        # func_concat(fc)

    from os import PathLike
    from pathlib import Path

    def func_pathlike(x: str | PathLike[Any]) -> Path:
        return Path(x)

    def tester_pathlike() -> None:
        func_pathlike("hello")
        func_pathlike(Path("there"))
