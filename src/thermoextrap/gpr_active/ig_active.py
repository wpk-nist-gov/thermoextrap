# Written by Jacob I. Monroe, NIST employee
"""
GPR for ideal gas (:mod:`~thermoextrap.gpr_active.ig_active`)
-------------------------------------------------------------

Generates ideal gas (1D in external potential) data to test GP models and active
learning strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import numpy as np
import xarray as xr
from cmomy.random import validate_rng

from thermoextrap import beta as xpan_beta
from thermoextrap import idealgas
from thermoextrap.data import DataCentralMomentsVals

from .active_utils import DataWrapper

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from thermoextrap.core.typing import OptionalRng
    from thermoextrap.models import ExtrapModel


# Work with fixed ideal gas test set in thermoextrap
# Define function to create ExtrapModel of ideal gas data
# This will be handy later on
def extrap_IG(beta: Any, rng: OptionalRng = None) -> ExtrapModel[xr.DataArray]:
    y_dat, u_dat = idealgas.generate_data((10000, 1000), beta, rng=validate_rng(rng))
    y_dat = xr.DataArray(y_dat[:, None], dims=["rec", "val"])
    u_dat = xr.DataArray(u_dat, dims=["rec"])
    data = DataCentralMomentsVals.from_vals(
        order=3, rec_dim="rec", xv=y_dat, uv=u_dat, central=True
    )
    return xpan_beta.factory_extrapmodel(beta, data)


def multiOutput_extrap_IG(
    beta: Any, rng: OptionalRng = None
) -> ExtrapModel[xr.DataArray]:
    # Use fixed random number
    positions = idealgas.x_sample((10000, 1000), beta, rng=validate_rng(rng))
    y = positions.mean(axis=-1)
    ysq = (positions**2).mean(axis=-1)
    y_dat = xr.DataArray(np.vstack([y, ysq]).T, dims=["rec", "val"])
    u_dat = xr.DataArray(positions.sum(axis=-1), dims=["rec"])
    data = DataCentralMomentsVals.from_vals(
        order=3, rec_dim="rec", xv=y_dat, uv=u_dat, central=True
    )
    return xpan_beta.factory_extrapmodel(beta, data)


# To help test active learning, build DataWrapper and SimWrapper objects for ideal gas
class IG_DataWrapper(DataWrapper):  # noqa: N801
    """Data object for gpr with ideal gas."""

    def __init__(self, beta: Any, rng: OptionalRng = None) -> None:
        self.beta = beta
        self.rng = validate_rng(rng)

    def load_U_info(self) -> NoReturn:
        raise NotImplementedError

    def load_CV_info(self) -> NoReturn:
        raise NotImplementedError

    def load_x_info(self) -> NoReturn:
        raise NotImplementedError

    def get_data(
        self, n_conf: int = 10000, n_part: int = 1000
    ) -> tuple[xr.DataArray, xr.DataArray, NDArray[Any]]:
        # Call thermoextrap.idealgas methods
        x, U = idealgas.generate_data((n_conf, n_part), self.beta, rng=self.rng)
        x = xr.DataArray(x[:, None], dims=["rec", "val"])
        U = xr.DataArray(U, dims=["rec"])
        return U, x, np.ones(U.shape, dtype=U.dtype)

    def build_state(
        self,
        all_data: tuple[xr.DataArray, xr.DataArray, NDArray[Any]] | None = None,
        max_order: int = 6,
    ) -> ExtrapModel[xr.DataArray]:
        if all_data is None:
            all_data = self.get_data()
        U = all_data[0]
        x = all_data[1]
        data = DataCentralMomentsVals.from_vals(
            order=max_order, rec_dim="rec", xv=x, uv=U, central=True
        )
        return xpan_beta.factory_extrapmodel(self.beta, data)


class SimulateIG:
    """Simulation object for ideal gas."""

    def __init__(self, sim_func: None = None) -> None:
        self.sim_func = sim_func  # Will not perform any simulations

    def run_sim(self, unused: Any, beta: Any, n_repeats: Any = None) -> IG_DataWrapper:
        # All this does is creates an IG_DataWrapper object at the specified beta
        # (and returns it)
        return IG_DataWrapper(beta)
