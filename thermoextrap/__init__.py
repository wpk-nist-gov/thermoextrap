"""
To access data/models/etc use thermoextrap.core.data....
"""

from . import beta, lnpi, volume, volume_idealgas
from .core import idealgas
from .core.data import (
    DataCentralMoments,
    DataCentralMomentsVals,
    DataValues,
    DataValuesCentral,
    factory_data_values,
    resample_indicies,
)

# expose some data/models
from .core.models import (
    ExtrapModel,
    ExtrapWeightedModel,
    InterpModel,
    InterpModelPiecewise,
    MBARModel,
    PerturbModel,
    StateCollection,
)
from .core.xrutils import xrwrap_alpha, xrwrap_uv, xrwrap_xv

# remove explicit imports of gpr stuff
# causing some warnings during tests that are annoying
# from . import gpr, gpr_stack

# updated versioning scheme
try:
    from importlib.metadata import version as _version
except ImportError:
    # if the fallback library is missing, we are doomed.
    from importlib_metadata import version as _version  # type: ignore[no-redef]

try:
    __version__ = _version("cmomy")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


__all__ = [
    "ExtrapModel",
    "ExtrapWeightedModel",
    "InterpModel",
    "InterpModelPiecewise",
    "MBARModel",
    "PerturbModel",
    "StateCollection",
    "DataCentralMoments",
    "DataCentralMomentsVals",
    "DataValues",
    "DataValuesCentral",
    "factory_data_values",
    "resample_indicies",
    "xrwrap_xv",
    "xrwrap_uv",
    "xrwrap_alpha",
    "idealgas",
    "beta",
    # "gpr",
    # "gpr_stack",
    "lnpi",
    "volume",
    "volume_idealgas",
    "__version__",
]
