"""
Routines for beta expansion(s)
"""

from __future__ import absolute_import

from functools import lru_cache

import sympy as sp

from .cached_decorators import gcached
from .data import (  # noqa: F401
    DataCentralMoments,
    DataCentralMomentsVals,
    DataValues,
    DataValuesCentral,
    resample_indicies,
)
from .models import (
    Derivatives,
    ExtrapModel,
    PerturbModel,
    SymSubs,
    _get_default_indexed,
    _get_default_symbol,
)

# from cmomy.options import set_options


##############################################################################
# recursive deriatives for beta expansion
###############################################################################

####################
# Central moments
####################


class du_func(sp.Function):
    """
    du_func(beta, n) = <(u - <u>)**n>

    Note
    ----
    sub in {beta: 'None'} to convert to indexed objects
    Have to use 'None' instead of None as sympy does some magic to
    the input arguments
    """

    nargs = 2
    du = _get_default_indexed("du")

    @classmethod
    def deriv_args(cls):
        """list of arguments to function evaluation"""
        return [cls.du]

    def fdiff(self, argindex=1):
        beta, n = self.args
        out = -(+du_func(beta, n + 1) - n * du_func(beta, n - 1) * du_func(beta, 2))
        return out

    @classmethod
    def eval(cls, beta, n):
        if n == 0:
            out = 1
        elif n == 1:
            out = 0
        elif beta is None:
            out = cls.du[n]
        else:
            out = None
        return out


class u_func_central(sp.Function):
    """
    energy function
    """

    nargs = 1
    u = _get_default_symbol("u")

    @classmethod
    def deriv_args(cls):
        return [cls.u] + du_func.deriv_args()

    def fdiff(self, argindex=1):
        (beta,) = self.args
        out = -du_func(beta, 2)
        return out

    @classmethod
    def eval(cls, beta):
        if beta is None:
            return cls.u
        else:
            out = None
        return out


class dxdu_func_nobeta(sp.Function):
    """
    for use when x is not a funciton of beta
    dxdu_func_nobeta(beta, n) = <du**n * dx> = dxdu[n]
    """

    nargs = 2
    dxdu = _get_default_indexed("dxdu")

    @classmethod
    def deriv_args(cls):
        return du_func.deriv_args() + [cls.dxdu]

    def fdiff(self, argindex=1):
        beta, n = self.args
        out = (
            -dxdu_func_nobeta(beta, n + 1)
            + n * dxdu_func_nobeta(beta, n - 1) * du_func(beta, 2)
            + dxdu_func_nobeta(beta, 1) * du_func(beta, n)
        )
        return out

    @classmethod
    def eval(cls, beta, n):
        if n == 0:
            out = 0
        elif beta is None:
            out = cls.dxdu[n]
        else:
            out = None
        return out


class dxdu_func_beta(sp.Function):
    """
    for use when x is a funciton of beta
    dxdu_func(beta, n, d) = <du**n * (x^(d) - <x^(d)>)> = dxdu[n, d]
    where x^(d) is the dth derivative of x
    """

    nargs = 3
    dxdu = _get_default_indexed("dxdu")

    @classmethod
    def deriv_args(cls):
        return du_func.deriv_args() + [cls.dxdu]

    def fdiff(self, argindex=1):
        beta, n, d = self.args
        out = (
            -dxdu_func_beta(beta, n + 1, d)
            + n * dxdu_func_beta(beta, n - 1, d) * du_func(beta, 2)
            + dxdu_func_beta(beta, n, d + 1)
            + dxdu_func_beta(beta, 1, d) * du_func(beta, n)
        )
        return out

    @classmethod
    def eval(cls, beta, n, deriv):
        if n == 0:
            out = 0
        elif beta is None:
            out = cls.dxdu[n, deriv]
        else:
            out = None
        return out


class x_func_central_nobeta(sp.Function):
    """
    function for calculating <x> where x != func(beta)
    """

    nargs = 1
    x1_symbol = _get_default_symbol("x1")

    @classmethod
    def deriv_args(cls):
        return [cls.x1_symbol] + dxdu_func_nobeta.deriv_args()

    def fdiff(self, argindex=1):
        (beta,) = self.args
        return -dxdu_func_nobeta(beta, 1)

    @classmethod
    def eval(cls, beta):
        if beta is None:
            out = cls.x1_symbol
        else:
            out = None
        return out


class x_func_central_beta(sp.Function):
    """
    function for calculation <x(beta)>
    """

    nargs = 2
    x1_indexed = _get_default_indexed("x1")

    @classmethod
    def deriv_args(cls):
        return [cls.x1_indexed] + dxdu_func_beta.deriv_args()

    def fdiff(self, argindex=1):
        # xalpha
        beta, d = self.args
        out = -dxdu_func_beta(beta, 1, d) + x_func_central_beta(beta, d + 1)
        return out

    @classmethod
    def eval(cls, beta, deriv):
        if beta is None:
            out = cls.x1_indexed[deriv]
        else:
            out = None
        return out


# class dxdu_func_gen_nobeta(sp.Function):
#     """
#     Function to calculate derivs of <dx**k * du**n>
#     """

#     nargs = 3
#     dxdu = _get_default_indexed('dxdu')

#     @classmethod
#     def deriv_args(cls):
#         return du_func.deriv_args() + [cls.dxdu]

#     def fdiff(self, argindex=1):
#         beta, n, k = self.args

#         out = (
#             +k * dxdu_func_gen_nobeta(beta, )


# class dx_func_nobeta(sp.Function):
#     """
#     funciton for calculating derivs of <dx**n>
#     """
#     nargs = 2

#     def fdiff(self, argindex=1):
#         beta, n = self.args

#         out = (
#             n * dx_func_nobeta(beta, n-1) * dxdu_func_nobeta(beta, 1)
#             -dxdu_func_nobeta(beta, n)
#         )


####################
# raw moments
####################
class u_func(sp.Function):
    """
    calculate derivatives of <U**n>
    """

    nargs = 2
    u = _get_default_indexed("u")

    @classmethod
    def deriv_args(cls):
        return [cls.u]

    def fdiff(self, argindex=1):
        beta, n = self.args
        return -(u_func(beta, n + 1) - u_func(beta, n) * u_func(beta, 1))

    @classmethod
    def eval(cls, beta, n):
        if n == 0:
            out = 1
        elif beta is None:
            out = cls.u[n]
        else:
            out = None
        return out


class xu_func(sp.Function):
    """
    calculate derivatives of
    xu_func(beta, n) = <x * u**n> = xu[n]
    or (xalpha)
    xu_func(beta, n, d) = <x^(d) * u**n> = xu[n, d]
    """

    nargs = (2, 3)
    xu = _get_default_indexed("xu")

    @classmethod
    def deriv_args(cls):
        return u_func.deriv_args() + [cls.xu]

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            beta, n = self.args
            out = -xu_func(beta, n + 1) + xu_func(beta, n) * u_func(beta, 1)

        else:
            beta, n, d = self.args
            out = (
                -xu_func(beta, n + 1, d)
                + xu_func(beta, n, d + 1)
                + xu_func(beta, n, d) * u_func(beta, 1)
            )
        return out

    @classmethod
    def eval(cls, beta, n, deriv=None):
        if beta is None:
            if deriv is None:
                out = cls.xu[n]
            else:
                out = cls.xu[n, deriv]
        else:
            out = None
        return out


class SymDerivBeta(object):
    """
    provide expressions for d^n <x> / d(beta)^n

    includes ability to use -ln(<x>) directly

    """

    beta = _get_default_symbol("beta")

    def __init__(self, func, args=None, expand=True, post_func=None):
        """
        Parameters
        ----------
        func : sympy function
            Function to differentiate
        args : Sequence of arguments to func
        expand : bool
        post_func : str or callable
        function to perform on funciton.
            For example, `post_fuc = -sympy.log` is equivalent to passing `minus_log=True`
            If a string, then apply the following standard functions

            * minus_log : post_func = -sympy.log
            * pow_{i} : post_func = lambda f: pow(f, i).  E.g., `pow_2` => pow(f, 2)
        """

        if args is None:
            args = func.deriv_args()

        self._func_orig = func
        self._post_func = post_func

        if post_func is not None:
            if isinstance(post_func, str):
                if post_func == "minus_log":
                    post_func = lambda f: -sp.log(f)
                elif post_func.startswith("pow_"):
                    i = int(post_func.split("_")[-1])
                    post_func = lambda f: pow(f, i)
                else:
                    raise ValueError(
                        "post_func must be callable or in {minus_log, pow_1, pow_2, ...}"
                    )
            func = post_func(func)

        self.func = func
        self.args = args
        self.expand = expand

    @classmethod
    def x_ave(cls, xalpha=False, central=False, expand=True, post_func=None):
        """
        General method to find derivatives of <x>

        Paremeters
        ----------
        xalapha : bool, default=False
            If True, then x is a function of the derivative variable
        central : bool, default=False
            If True, work with central moments.  Otherwise, use raw moments.
        expand : bool, default=True
            Whether to expand expressions
        """

        if central:
            if xalpha:
                func = x_func_central_beta(cls.beta, 0)

            else:
                func = x_func_central_nobeta(cls.beta)

        else:
            if xalpha:
                func = xu_func(cls.beta, 0, 0)
            else:
                func = xu_func(cls.beta, 0)

        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    def u_ave(cls, central=True, expand=True, post_func=None):
        """
        General method to find derivatives of <u>
        """

        if central:
            func = u_func_central(cls.beta)
        else:
            func = u_func(cls.beta, 1)

        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    def dun_ave(cls, n, expand=True, post_func=None):
        """
        constructor for derivatives of  <(u - <u>)**n> = <du**n>
        """
        n = int(n)
        assert n > 1
        func = du_func(cls.beta, n)

        # special case for args.
        # for consistency between uave and dun_ave, also include u variable
        args = u_func_central.deriv_args()
        return cls(
            func=func,
            args=args,
            expand=expand,
            post_func=post_func,
        )

    @classmethod
    def dxdun_ave(cls, n, xalpha=False, expand=True, post_func=None, d=None):
        """
        constructor for derivatives of <dx * du**n>

        if xalpha is True, must also specify d, which is the order of deriative on `x`
        """

        # special case for args
        # for consistency between xave and dxdun_ave, also include x1

        assert isinstance(n, int) and n > 0
        if xalpha:
            assert isinstance(d, int)
            func = dxdu_func_beta(cls.beta, n, d)
            args = x_func_central_beta.deriv_args()

        else:
            func = dxdu_func_nobeta(cls.beta, n)
            args = x_func_central_nobeta.deriv_args()

        return cls(
            func=func,
            args=args,
            expand=expand,
            post_func=post_func,
        )

    @classmethod
    def un_ave(cls, n, expand=True, post_func=None):
        """
        constructor for derivatives of <x> = <u**n>
        """
        n = int(n)
        assert n >= 1

        func = u_func(cls.beta, n)
        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    def xun_ave(cls, n, d=None, xalpha=False, expand=True, post_func=None):
        """
        x = <x^(d) * u **n>.
        """

        assert isinstance(n, int) and n >= 0

        if xalpha:
            assert isinstance(d, int) and d >= 0
            func = xu_func(cls.beta, n, d)
        else:
            func = xu_func(cls.beta, n)

        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    def from_name(
        cls,
        name,
        xalpha=False,
        central=False,
        expand=True,
        post_func=None,
        n=None,
        d=None,
    ):
        """
        create a derivative expressions indexer by name

        Parameters
        ----------
        name : {'xave', 'uave', 'dun_ave', 'un_ave'}
        All properties use post_func and expand parameters.
            * x_ave: general average of <x>(central, xalpha)
            * u_ave: <u>(central)
            * dun_ave: derivative of <(u - <u>)**n>(central, n)
            * dxdun_ave: derivatives of <dx^(d) * du**n>(xalpha, n, d)

            * un_ave: derivative of <u**n>(n)
            * xun_ave: derivative of <x^(d) * u**n>(xalpha, n, [d])

        xalpha : bool, default=False
            Whether property depends on alpha (beta)
        central : bool, default=False
            Whether central moments expansion should be used
        expand : bool, default=True
            Whether expressions should be expanded
        n : int, optional
            n parameter used for dun_ave or un_ave
        d : int, optional
            d parameter for dxdun_ave
        """

        func = getattr(cls, name, None)

        if func is None:
            raise ValueError("{name} not found")

        kws = {"expand": expand, "post_func": post_func}
        if name == "x_ave":
            kws.update(xalpha=xalpha, central=central)
        elif name == "u_ave":
            kws.update(central=central)
        elif name in ["dun_ave", "un_ave"]:
            kws.update(n=n)
        elif name in ["dxdun_ave", "xun_ave"]:
            kws.update(n=n, xalpha=xalpha, d=d)

        return func(**kws)

    @gcached(prop=False)
    def __getitem__(self, order):
        if order == 0:
            out = self.func
        else:
            out = self[order - 1].diff(self.beta, 1)
            if self.expand:
                out = out.expand()
        return out

    # def _init_old(self, xalpha=False, central=False, expand=True, _minus_log=False):
    #     if central:
    #         if xalpha:
    #             x = x_func_central(self.beta, 0)
    #             args = [x.x1_indexed]
    #         else:
    #             x = x_func_central(self.beta)
    #             args = [x.x1_symbol]
    #         args += _get_default_indexed("du", "dxdu")

    #     else:
    #         if xalpha:
    #             x = xu_func(self.beta, 0, 0)
    #         else:
    #             x = xu_func(self.beta, 0)
    #         args = _get_default_indexed("u", "xu")

    #     if minus_log:
    #         x = -sp.log(x)

    #     self.xave = x
    #     self.args = args
    #     self.expand = expand


###############################################################################
# Factory functions
###############################################################################
def factory_data(
    uv,
    xv,
    order,
    central=False,
    skipna=False,
    xalpha=False,
    rec_dim="rec",
    umom_dim="umom",
    val_dims="val",
    rep_dim="rep",
    deriv_dim=None,
    chunk=None,
    compute=None,
    **kws
):
    """
    Factory function to produce a Data object

    Parameters
    ----------
    uv : array-like
        energy values
    xv : array-like
        observable values
    order : int
        highest umom_dim to calculate
    skipna : bool, default=False
        if True, skip `np.nan` values in creating averages.
        Can make some "big" calculations slow
    rec_dim, umom_dim, val_dim, rep_dim, deriv_dim : str
        names of record (i.e. time), umom_dim, value, replicate,
        and derivative (with respect to alpha)
    chunk : int or dict, optional
        If specified, perform chunking on resulting uv, xv arrays.
        If integer, chunk with {rec: chunk}
        otherwise, should be a mapping of form {dim_0: chunk_0, dim_1: chunk_1, ...}
    compute : bool, optional
        if compute is True, do compute averages greedily.
        if compute is False, and have done chunking, then defer calculation of averages (i.e., will be dask future objects).
        Default is to do greedy calculation
    kws : dict, optional
        extra arguments
    """

    if central:
        cls = DataValuesCentral
    else:
        cls = DataValues

    if xalpha and deriv_dim is None:
        raise ValueError("if xalpha, must pass string name of derivative")

    return cls.from_vals(
        uv=uv,
        xv=xv,
        order=order,
        skipna=skipna,
        rec_dim=rec_dim,
        umom_dim=umom_dim,
        val_dims=val_dims,
        rep_dim=rep_dim,
        deriv_dim=deriv_dim,
        chunk=chunk,
        compute=compute,
        **kws
    )


@lru_cache(5)
def factory_derivatives(
    name="x_ave",
    n=None,
    d=None,
    xalpha=False,
    central=False,
    post_func=None,
):
    """
    factory function to provide derivative function for expansion

    Parameters
    ----------
    name : {x_ave, u_ave, dxdun_ave, dun_ave, un_ave, xun_ave}
    xalpha : bool, default=False
        whether x = func(beta) or not
    central : bool, default=False
        whether to use central moments or not
    post_func : str or callable

    Returns
    -------
    derivatives : Derivatives object used to calculate taylor series coefficients
    """

    derivs = SymDerivBeta.from_name(
        name=name,
        n=n,
        d=d,
        xalpha=xalpha,
        central=central,
        post_func=post_func,
    )
    exprs = SymSubs(
        derivs,
        subs_all={derivs.beta: "None"},
        expand=False,
        simplify=False,
    )
    return Derivatives.from_sympy(exprs, args=derivs.args)


def factory_extrapmodel(
    beta,
    data,
    name="x_ave",
    n=None,
    d=None,
    xalpha=None,
    central=None,
    order=None,
    alpha_name="beta",
    derivatives=None,
    post_func=None,
    derivatives_kws=None,
):
    """
    factory function to create Extrapolation model for beta expansion

    Parameters
    ----------
    beta : float
        reference value of inverse temperature
    data : Data object
        data object to consider.
        See data.AbstractData
    order : int, optional
        maximum order.
        If not specified, infer from `data`
    xalpha : bool, optional
        Whether or not x = func(alpha).
        If not specified, infer from `data`
    central : bool, optional
        Whether or not to use central moments
        If not specified, infer from `data`
    post_func : str or callable
        apply function to x.  For example, post_func = 'minus_log' or post_func = lambda x: -sympy.log(x)
    alpha_name, str, default='beta'
        name of expansion parameter
    kws : dict
        extra arguments to `factory_data`

    Returns
    -------
    extrapmodel : ExtrapModel object
    """

    if xalpha is None:
        xalpha = data.xalpha
    if central is None:
        central = data.central
    if order is None:
        order = data.order

    assert xalpha == data.xalpha
    assert central == data.central
    assert order <= data.order

    if derivatives is None:
        if name in ["u_ave", "un_ave", "dun_ave"]:
            assert data.x_is_u

        if derivatives_kws is None:
            derivatives_kws = {}
        derivatives = factory_derivatives(
            name=name,
            n=n,
            d=d,
            xalpha=xalpha,
            central=central,
            post_func=post_func,
            **derivatives_kws
        )
    return ExtrapModel(
        alpha0=beta,
        data=data,
        derivatives=derivatives,
        order=order,
        # minus_log=mineus_log,
        alpha_name=alpha_name,
    )


def factory_perturbmodel(beta, uv, xv, alpha_name="beta", **kws):
    """
    factory function to create PerturbModel for beta expansion

    Parameters
    ----------
    beta : float
        reference value of inverse temperature
    uv, xv : array-like
        values of u and x
    alpha_name : str, default='beta'
        name of expansion parameter
    kws : dict
        extra arguments to `models.Data`

    Returns
    -------
    perturbmodel : PerturbModel object
    """
    data = factory_data(uv=uv, xv=xv, order=0, central=False, **kws)
    return PerturbModel(alpha0=beta, data=data, alpha_name=alpha_name)
