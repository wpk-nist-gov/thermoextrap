"""
Volume expansion for ideal gas (:mod:`~thermoextrap.volume_idealgas`)
=====================================================================
"""

from functools import lru_cache

from .core.docstrings import DOCFILLER_SHARED
from .models import Derivatives, ExtrapModel

docfiller_shared = DOCFILLER_SHARED.levels_to_top(
    "cmomy", "xtrap", "beta", "volume"
).decorate


class VolumeDerivFuncsIG:
    """
    Calculates specific derivative values at refV with data x and W.
    Only go to first order for volume extrapolation.
    Here W represents the virial instead of the potential energy.
    """

    def __init__(self, refV=1.0) -> None:  # noqa: N803
        # If do not set refV, assumes virial data is already divided by the reference volume
        # If this is not the case, need to set refV
        # Or if need refV to also compute custom term, need to specify
        self.refV = refV

    def __getitem__(self, order):
        # Check to make sure not going past first order
        if order > 1:
            raise ValueError(
                "Volume derivatives cannot go past 1st order"
                + " and received %i" % order
                + "\n(because would need derivatives of forces)"
            )
        return self.create_deriv_func(order)

    def create_deriv_func(self, order):
        # Works only because of local scope
        # Even if order is defined somewhere outside of this class, won't affect returned func

        def func(W, xW):  # noqa: N803
            if order == 0:
                # Zeroth order derivative
                deriv_val = xW[0]

            else:
                # First order derivative
                deriv_val = (xW[1] - xW[0] * W[1]) / (
                    self.refV
                )  # No 3 b/c our IG is 1D
                # Term unique to Ideal Gas... <x>/L
                # Replace with whatever is appropriate to observable of interest
                deriv_val += xW[0] / self.refV
            return deriv_val

        return func


@lru_cache(5)
def factory_derivatives(refV=1.0):  # noqa: N803
    """
    Factory function to provide coefficients of expansion.

    Parameters
    ----------
    refV : float
        reference volume (default 1 - if already divided by volume no need to set)

    Returns
    -------
    derivatives : :class:`thermoextrap.models.Derivatives` object
        Object used to calculate moments
    """
    deriv_funcs = VolumeDerivFuncsIG(refV=refV)
    return Derivatives(deriv_funcs)


def factory_extrapmodel(volume, uv, xv, order=1, alpha_name="volume", **kws):
    """
    Factory function to create Extrapolation model for volume expansion.

    Parameters
    ----------
    order : int
        maximum order
    volume : float
        reference value of volume
    uv, xv : array-like
        values for u and x
    alpha_name, str, default='volume'
        name of expansion parameter
    kws : dict
        extra arguments to `factory_datavalues`

    Returns
    -------
    extrapmodel : ExtrapModel
    """
    if order != 1:
        msg = "only first order supported"
        raise ValueError(msg)

    from .data import factory_data_values

    data = factory_data_values(
        uv=uv, xv=xv, order=order, central=False, xalpha=False, **kws
    )
    derivatives = factory_derivatives(refV=volume)
    return ExtrapModel(
        alpha0=volume,
        data=data,
        derivatives=derivatives,
        order=order,
        minus_log=False,
        alpha_name=alpha_name,
    )


def factory_extrapmodel_data(volume, data, order=1, alpha_name="volume"):
    """
    Factory function to create Extrapolation model for volume expansion.

    Parameters
    ----------
    volume : float
        reference value of volume
    data : object
        Note that this data object should have central=False, deriv_dim=None
    alpha_name, str, default='volume'
        name of expansion parameter

    Returns
    -------
    extrapmodel : ExtrapModel
    """
    if order is None:
        order = data.order

    if order != 1:
        msg = "only first order supported"
        raise ValueError(msg)

    if order > data.order:
        raise ValueError
    if data.central:
        msg = "Only works with raw moments."
        raise ValueError
    if data.deriv is not None:
        raise ValueError

    derivatives = factory_derivatives(refV=volume)
    return ExtrapModel(
        alpha0=volume,
        data=data,
        derivatives=derivatives,
        order=order,
        minus_log=False,
        alpha_name=alpha_name,
    )
