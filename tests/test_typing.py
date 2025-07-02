"""Check/test typing support"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

import thermoextrap as xtrap

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

if TYPE_CHECKING:
    from cmomy.core.typing_compat import TypeVar
    from numpy.typing import NDArray

    T = TypeVar("T")


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


# * Parameters
val_array: NDArray[np.float64] = np.zeros(10, dtype=np.float64)
val_dataarray = xr.DataArray(val_array)
val_dataset = xr.Dataset({"val": val_dataarray})


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
