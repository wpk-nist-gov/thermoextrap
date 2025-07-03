"""Check/test typing support"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import cmomy
import numpy as np
import xarray as xr

import thermoextrap as xtrap

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

if TYPE_CHECKING:
    from cmomy.core.typing_compat import TypeVar

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


# * DataSelector --------------------------------------------------------------
def check_dataselector(
    actual: T,
    klass: type[Any],
    data_type: type[Any],
) -> T:
    assert isinstance(actual, klass)

    assert isinstance(actual.data, data_type)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    return actual  # type: ignore[no-any-return]


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
            xtrap.data.DataSelector.from_defaults(data_dataset, mom_dim=dims),
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
    return actual  # type: ignore[no-any-return]


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
    return actual  # type: ignore[no-any-return]


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
            xtrap.data.DataCentralMoments.from_raw(data_dataset),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_vals(
                val_dataarray, xv=val_dataset, order=2, dim=dim
            ),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_data(data_dataset),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_resample_vals(
                uv=val_dataarray, xv=val_dataset, order=2, sampler=2, dim=dim
            ),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_ave_raw(
                u=val_dataset, xu=val_dataset, umom_dim="val"
            ),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )
    check_datacentralmoments(
        assert_type(
            xtrap.data.DataCentralMoments.from_ave_central(
                du=val_dataset, dxdu=val_dataset, umom_dim="val"
            ),
            xtrap.data.DataCentralMoments[xr.Dataset],
        ),
        xtrap.data.DataCentralMoments,
        xr.Dataset,
    )


# def test_thermoextrap_data_datacentralmoments_dataset() -> None:
#     dim = "val"
#     # Dataset
#     check_datacentralmoments(
#         assert_type(
#             xtrap.data.DataCentralMoments(dxduave_dataset),
#             xtrap.data.DataCentralMoments[xr.Dataset],
#         ),
#         xtrap.data.DataCentralMoments,
#         xr.Dataset,
#     )
#     check_datacentralmoments(
#         assert_type(
#             xtrap.data.DataCentralMoments.from_raw(val_dataset),
#             xtrap.data.DataCentralMoments[xr.Dataset],
#         ),
#         xtrap.data.DataCentralMoments,
#         xr.Dataset,
#     )
#     check_datacentralmoments(
#         assert_type(
#             xtrap.data.DataCentralMoments.from_vals(val_dataarray, xv=val_dataset, order=2, dim=dim),
#             xtrap.data.DataCentralMoments[xr.Dataset],
#         ),
#         xtrap.data.DataCentralMoments,
#         xr.Dataset,
#     )
#     check_datacentralmoments(
#         assert_type(
#             xtrap.data.DataCentralMoments.from_data(val_dataset),
#             xtrap.data.DataCentralMoments[xr.Dataset],
#         ),
#         xtrap.data.DataCentralMoments,
#         xr.Dataset,
#     )
#     check_datacentralmoments(
#         assert_type(
#             xtrap.data.DataCentralMoments.from_resample_vals(uv=val_dataarray, xv=val_dataset, order=2, sampler=2, dim=dim),
#             xtrap.data.DataCentralMoments[xr.Dataset],
#         ),
#         xtrap.data.DataCentralMoments,
#         xr.Dataset,
#     )
#     check_datacentralmoments(
#         assert_type(
#             xtrap.data.DataCentralMoments.from_ave_raw(u=val_dataset, xu=val_dataset),
#             xtrap.data.DataCentralMoments[xr.Dataset],
#         ),
#         xtrap.data.DataCentralMoments,
#         xr.Dataset,
#     )
#     check_datacentralmoments(
#         assert_type(
#             xtrap.data.DataCentralMoments.from_ave_central(du=val_dataset, dxdu=val_dataset),
#             xtrap.data.DataCentralMoments[xr.Dataset],
#         ),
#         xtrap.data.DataCentralMoments,
#         xr.Dataset,
#     )
