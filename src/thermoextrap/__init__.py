"""Classes/routines to deal with thermodynamic extrapolation."""
# pylint: disable=duplicate-code

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (
        beta,
        data,
        idealgas,
        lnpi,
        models,
        volume,
        volume_idealgas,
    )
    from .core.xrutils import xrwrap_alpha, xrwrap_uv, xrwrap_xv
    from .data import (
        DataCentralMoments,
        DataCentralMomentsVals,
        factory_data_values,
    )

    # expose some data/models
    from .models import (
        Derivatives,
        ExtrapModel,
        ExtrapWeightedModel,
        InterpModel,
        InterpModelPiecewise,
        MBARModel,
        PerturbModel,
        StateCollection,
    )
else:
    import lazy_loader as lazy

    __getattr__, __dir__, _ = lazy.attach(
        __name__,
        submodules=[
            "beta",
            "data",
            "idealgas",
            "lnpi",
            "models",
            "random",
            "volume",
            "volume_idealgas",
        ],
        submod_attrs={
            "data": [
                "DataCentralMoments",
                "DataCentralMomentsVals",
                "factory_data_values",
            ],
            "models": [
                "Derivatives",
                "ExtrapModel",
                "ExtrapWeightedModel",
                "InterpModel",
                "InterpModelPiecewise",
                "MBARModel",
                "PerturbModel",
                "StateCollection",
            ],
            "core.xrutils": ["xrwrap_alpha", "xrwrap_uv", "xrwrap_xv"],
        },
    )


try:
    __version__ = _version("thermoextrap")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "999"


__all__ = [
    "DataCentralMoments",
    "DataCentralMomentsVals",
    "Derivatives",
    "ExtrapModel",
    "ExtrapWeightedModel",
    "InterpModel",
    "InterpModelPiecewise",
    "MBARModel",
    "PerturbModel",
    "StateCollection",
    "__version__",
    "beta",
    "data",
    "factory_data_values",
    "idealgas",
    "lnpi",
    "models",
    "volume",
    "volume_idealgas",
    "xrwrap_alpha",
    "xrwrap_uv",
    "xrwrap_xv",
]
