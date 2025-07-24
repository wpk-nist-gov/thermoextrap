# pyright: reportMissingTypeStubs=false, reportIncompatibleMethodOverride=false

"""
Inverse temperature (beta) extrapolation (:mod:`~thermoextrap.beta`)
====================================================================
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

from sympy.core.numbers import Number

from .core.sputils import (
    get_default_indexed,
    get_default_symbol,
)
from .core.validate import validate_positive_integer
from .docstrings import DOCFILLER_SHARED
from .models import (
    Derivatives,
    ExtrapModel,
    PerturbModel,
    SymDerivBase,
    SymFuncBase,
    SymSubs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import xarray as xr
    from sympy.core.add import Add
    from sympy.core.mul import Expr
    from sympy.core.symbol import Symbol
    from sympy.tensor.indexed import IndexedBase

    from .core.typing import DataT, PostFunc, SupportsDataProtocol, SymDerivNames
    from .core.typing_compat import Self

docfiller_shared = DOCFILLER_SHARED.levels_to_top("cmomy", "xtrap", "beta")

##############################################################################
# recursive derivatives for beta expansion
###############################################################################

####################
# Central moments
####################


class du_func(SymFuncBase):
    r"""
    Sympy function to evaluate energy fluctuations using central moments.

    :math:`\text{du_func}(\beta, n) = \langle (u(\beta) - \langle u(\beta) \rangle)^n \rangle`

    Notes
    -----
    sub in ``{'beta': 'None'}`` to convert to indexed objects
    Have to use ``'None'`` instead of ``None`` as sympy does some magic to
    the input arguments.
    """

    nargs = 2
    du = get_default_indexed("du")

    # NOTE: only including init so typing makes sense...
    def __init__(self, beta: Symbol | None, n: int | Number, /) -> None:
        super().__init__()

    @classmethod
    def deriv_args(cls) -> tuple[IndexedBase]:
        return (cls.du,)

    def fdiff(self, argindex: int | Number = 1) -> Expr:
        beta, n = self.args
        return -(+du_func(beta, n + 1) - n * du_func(beta, n - 1) * du_func(beta, 2))  # pyright: ignore[reportUnknownVariableType]

    @classmethod
    def eval(cls, beta: Symbol | None, n: int | Number = 0) -> Expr | None:
        if n == 0:
            return Number(1)
        if n == 1:
            return Number(0)
        if beta is None:
            return cls.du[n]  # pyright: ignore[reportUnknownVariableType,reportReturnType]
        return None


class u_func_central(SymFuncBase):
    r"""
    Sympy function to evaluate energy averages using central moments.

    :math:`\text{u_func_central}(beta, n) = \langle u(\beta)^n \rangle`
    """

    nargs = 1
    u = get_default_symbol("u")

    def __init__(self, beta: Symbol | None, /) -> None:
        super().__init__()

    @classmethod
    def deriv_args(cls) -> tuple[Symbol, IndexedBase]:
        return (cls.u, *du_func.deriv_args())

    def fdiff(self, argindex: int | Number = 1) -> Expr:
        (beta,) = self.args
        return -du_func(beta, 2)  # pyright: ignore[reportUnknownVariableType]

    @classmethod
    def eval(cls, beta: Symbol | None) -> Expr | None:
        if beta is None:
            return cls.u
        return None


class dxdu_func_nobeta(SymFuncBase):
    r"""
    Sympy function to evaluate observable energy fluctuations using central moments.

    :math:`\text{dxdu_func_nobeta}(\beta, n) = \langle \delta x (\delta u)^n \rangle`

    for use when x is not a function of beta.
    """

    nargs = 2
    dxdu = get_default_indexed("dxdu")

    def __init__(self, beta: Symbol | None, n: int | Number, /) -> None:
        super().__init__()

    @classmethod
    def deriv_args(cls) -> tuple[IndexedBase, IndexedBase]:
        return (*du_func.deriv_args(), cls.dxdu)

    def fdiff(self, argindex: int | Number = 1) -> Expr:
        beta, n = self.args
        return (  # pyright: ignore[reportUnknownVariableType]
            -dxdu_func_nobeta(beta, n + 1)
            + n * dxdu_func_nobeta(beta, n - 1) * du_func(beta, 2)
            + dxdu_func_nobeta(beta, 1) * du_func(beta, n)
        )

    @classmethod
    def eval(cls, beta: Symbol | None, n: int | Number = 0) -> Expr | None:
        if n == 0:
            return Number(0)
        if beta is None:
            return cls.dxdu[n]  # pyright: ignore[reportUnknownVariableType,reportReturnType]
        return None


class dxdu_func_beta(SymFuncBase):
    r"""
    Sympy function to evaluate derivatives of observable fluctuations using central moments.

    :math:`\text{dxdu_func_beta}(\beta, n, d) = \langle \delta  x^{(d)}(\beta)(\delta u)^n \rangle`, where :math:`x^{(k)} = d^k x / d\beta^k`.

    """

    nargs = 3
    dxdu = get_default_indexed("dxdu")

    def __init__(
        self, beta: Symbol | None, n: int | Number, deriv: int | Number, /
    ) -> None:
        super().__init__()

    @classmethod
    def deriv_args(cls) -> tuple[IndexedBase, IndexedBase]:
        return (*du_func.deriv_args(), cls.dxdu)

    def fdiff(self, argindex: int | Number = 1) -> Add:
        beta, n, d = self.args
        return (  # pyright: ignore[reportUnknownVariableType]
            -dxdu_func_beta(beta, n + 1, d)
            + n * dxdu_func_beta(beta, n - 1, d) * du_func(beta, 2)
            + dxdu_func_beta(beta, n, d + 1)
            + dxdu_func_beta(beta, 1, d) * du_func(beta, n)
        )

    @classmethod
    def eval(
        cls, beta: Symbol | None, n: int | Number = 0, deriv: int | Number = 0
    ) -> Expr | None:
        if n == 0:
            return Number(0)
        if beta is None:
            return cls.dxdu[n, deriv]  # pyright: ignore[reportReturnType, reportUnknownVariableType]
        return None


class x_func_central_nobeta(SymFuncBase):
    r"""Sympy function to evaluate derivatives of observable :math:`\langle x \rangle` using central moments."""

    nargs = 1
    x1_symbol = get_default_symbol("x1")

    def __init__(self, beta: Symbol | None, /) -> None:
        super().__init__()

    @classmethod
    def deriv_args(cls) -> tuple[Symbol, IndexedBase, IndexedBase]:
        return (cls.x1_symbol, *dxdu_func_nobeta.deriv_args())

    def fdiff(self, argindex: int | Number = 1) -> Expr:
        (beta,) = self.args
        return -dxdu_func_nobeta(beta, 1)  # pyright: ignore[reportUnknownVariableType]

    @classmethod
    def eval(cls, beta: Symbol | None) -> Expr | None:
        return cls.x1_symbol if beta is None else None


class x_func_central_beta(SymFuncBase):
    r"""Sympy function to evaluate derivatives of observable :math:`\langle x(\beta) \rangle` using central moments."""

    nargs = 2
    x1_indexed = get_default_indexed("x1")

    def __init__(self, beta: Symbol | None, deriv: int | Number, /) -> None:
        super().__init__()

    @classmethod
    def deriv_args(cls) -> tuple[IndexedBase, IndexedBase, IndexedBase]:
        return (cls.x1_indexed, *dxdu_func_beta.deriv_args())

    def fdiff(self, argindex: int | Number = 1) -> Expr:
        # xalpha
        beta, d = self.args
        return -dxdu_func_beta(beta, 1, d) + x_func_central_beta(beta, d + 1)  # pyright: ignore[reportUnknownVariableType]

    @classmethod
    def eval(cls, beta: Symbol | None, deriv: int | Number = 0) -> Expr | None:
        return cls.x1_indexed[deriv] if beta is None else None  # pyright: ignore[reportReturnType, reportUnknownVariableType]


####################
# raw moments
####################
class u_func(SymFuncBase):
    r"""Sympy function to evaluate derivatives of energy :math:`\langle u \rangle` using raw moments."""

    nargs = 2
    u = get_default_indexed("u")

    def __init__(self, beta: Symbol | None, n: int | Number, /) -> None:
        super().__init__()

    @classmethod
    def deriv_args(cls) -> tuple[IndexedBase]:
        return (cls.u,)

    def fdiff(self, argindex: int | Number = 1) -> Expr:
        beta, n = self.args
        return -(u_func(beta, n + 1) - u_func(beta, n) * u_func(beta, 1))  # pyright: ignore[reportUnknownVariableType]

    @classmethod
    def eval(cls, beta: Symbol | None, n: int | Number = 0) -> Expr | None:
        if n == 0:
            return Number(1)
        if beta is None:
            return cls.u[n]  # pyright: ignore[reportReturnType, reportUnknownVariableType]
        return None


class xu_func(SymFuncBase):
    r"""
    Sympy function to evaluate derivatives of :math:`\langle x u^n \rangle`.

    If ``x`` is a function of ``beta``, then :math:`\text{xu_func}(\beta, n, d) = \langle x^{(d)} u^n \rangle`.
    If ``x`` is not a function of ``beta``, drop argument ``d``.
    """

    nargs = (2, 3)
    xu = get_default_indexed("xu")

    def __init__(
        self, beta: Symbol | None, n: int | Number, deriv: int | Number | None = None, /
    ) -> None:
        super().__init__()

    @classmethod
    def deriv_args(cls) -> tuple[IndexedBase, IndexedBase]:
        return (*u_func.deriv_args(), cls.xu)

    def fdiff(self, argindex: int | Number = 1) -> Add:
        if len(self.args) == 2:
            beta, n = self.args
            return -xu_func(beta, n + 1) + xu_func(beta, n) * u_func(beta, 1)  # pyright: ignore[reportUnknownVariableType]

        beta, n, d = self.args
        return (  # pyright: ignore[reportUnknownVariableType]
            -xu_func(beta, n + 1, d)
            + xu_func(beta, n, d + 1)
            + xu_func(beta, n, d) * u_func(beta, 1)
        )

    @classmethod
    def eval(
        cls,
        beta: Symbol | None,
        n: int | Number = 0,
        deriv: int | Number | None = None,
    ) -> Expr | None:
        if beta is None:
            return cls.xu[n] if deriv is None else cls.xu[n, deriv]  # pyright: ignore[reportReturnType, reportUnknownVariableType]
        return None


@docfiller_shared.inherit(SymDerivBase)
class SymDerivBeta(SymDerivBase):
    r"""Provide symbolic expressions for :math:`d^n \langle x \rangle /d\beta^n`."""

    beta = get_default_symbol("beta")

    @classmethod
    @docfiller_shared.decorate
    def x_ave(
        cls,
        xalpha: bool = False,
        central: bool | None = None,
        expand: bool = True,
        post_func: PostFunc = None,
    ) -> Self:
        r"""
        General method to find derivatives of :math:`\langle x \rangle`.

        Parameters
        ----------
        {xalpha}
        {central}
        {expand}
        {post_func}
        """
        if central is None:
            central = False

        if central:
            if xalpha:
                func = x_func_central_beta(cls.beta, 0)

            else:
                func = x_func_central_nobeta(cls.beta)

        else:
            func = xu_func(cls.beta, 0, 0) if xalpha else xu_func(cls.beta, 0)

        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    @docfiller_shared.decorate
    def u_ave(
        cls,
        central: bool | None = None,
        expand: bool = True,
        post_func: PostFunc = None,
    ) -> Self:
        r"""
        General constructor for symbolic derivatives of :math:`\langle u \rangle`.

        Parameters
        ----------
        {central}
        {expand}
        {post_func}

        """
        if central is None:
            central = False

        func = u_func_central(cls.beta) if central else u_func(cls.beta, 1)

        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    @docfiller_shared.decorate
    def dun_ave(
        cls,
        n: int,
        expand: bool = True,
        post_func: PostFunc = None,
        central: bool | None = None,
    ) -> Self:
        r"""
        Constructor for derivatives of :math:`\langle (\delta u)^n\rangle`.

        Parameters
        ----------
        {n_order}
        {expand}
        {post_func}
        {central}
        """
        if central is not None and not central:
            msg = f"{central=} must be None or evaluate to True"
            raise ValueError(msg)

        if (n := int(n)) <= 1:
            msg = f"{n=} must be > 1."
            raise ValueError(msg)
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
    @docfiller_shared.decorate
    def dxdun_ave(
        cls,
        n: int,
        xalpha: bool = False,
        expand: bool = True,
        post_func: PostFunc = None,
        d: int | None = None,
        central: bool | None = None,
    ) -> Self:
        r"""
        Constructor for derivatives of :math:`\langle \delta x \delta u^n\rangle`.

        Parameters
        ----------
        {n_order}
        {expand}
        {xalpha}
        {post_func}
        {d_order}
        {central}

        Notes
        -----
        If xalpha is True, must also specify d.
        """
        # special case for args
        # for consistency between xave and dxdun_ave, also include x1
        if central is not None and not central:
            msg = f"{central=} nust be `None` or evaluate to `True`"
            raise ValueError(msg)

        if (n := int(n)) <= 0:
            msg = f"{n=} must be positive integer."
            raise ValueError(msg)
        if xalpha:
            func = dxdu_func_beta(cls.beta, n, validate_positive_integer(d, "d"))
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
    @docfiller_shared.decorate
    def un_ave(
        cls,
        n: int,
        expand: bool = True,
        post_func: PostFunc = None,
        central: bool | None = None,
    ) -> Self:
        r"""
        Constructor for derivatives of :math:`\langle u^n\rangle`.

        Parameters
        ----------
        {n_order}
        {expand}
        {post_func}
        {central}
        """
        if central is not None and central:
            msg = f"{central=} must be `None` or evaluate to False"
            raise ValueError(msg)

        if (n := int(n)) < 1:
            msg = f"{n=} must be >=1."
            raise ValueError(msg)

        func = u_func(cls.beta, n)
        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    @docfiller_shared.decorate
    def xun_ave(
        cls,
        n: int,
        d: int | None = None,
        xalpha: bool = False,
        expand: bool = True,
        post_func: PostFunc = None,
        central: bool | None = None,
    ) -> Self:
        r"""
        Constructor for derivatives of :math:`\langle x^{{(d)}} u^n\rangle`.

        Parameters
        ----------
        {n_order}
        {d_order}
        {xalpha}
        {expand}
        {post_func}
        {central}
        """
        if central is not None and central:
            msg = f"{central=} must be `None` or False"
            raise ValueError(msg)

        # assert isinstance(n, int)
        # assert n >= 0
        if (n := int(n)) < 0:
            msg = f"{n=} must be >= 0"
            raise ValueError(msg)

        if xalpha:
            func = xu_func(cls.beta, n, validate_positive_integer(d, "d"))
        else:
            func = xu_func(cls.beta, n)

        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    def from_name(
        cls,
        name: SymDerivNames,
        xalpha: bool = False,
        central: bool | None = None,
        expand: bool = True,
        post_func: PostFunc = None,
        n: int | None = None,
        d: int | None = None,
    ) -> Self:
        """
        Create a derivative expressions indexer by name.

        Parameters
        ----------
        name : {'xave', 'uave', 'dun_ave', 'un_ave'}
        All properties use post_func and expand parameters.
            * x_ave: general average of <x>(central, xalpha)
            * un_ave: <u>(central)
            * dun_ave: derivative of <(u - <u>)**n>(central, n)
            * dxdun_ave: derivatives of <dx^(d) * du**n>(xalpha, n, d)

            * un_ave: derivative of <u**n>(n)
            * xun_ave: derivative of <x^(d) * u**n>(xalpha, n, [d])
            * lnPi_energy: derivatives of <lnPi - beta * mu * N>(central)

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
            msg = f"{name} not found"
            raise ValueError(msg)

        kws: dict[str, Any] = {
            "expand": expand,
            "post_func": post_func,
            "central": central,
        }
        if name == "x_ave":
            kws.update(xalpha=xalpha)
        # elif name in ["u_ave", "lnPi_correction":
        #     kws.update(central=central)
        elif name in {"dun_ave", "un_ave"}:
            kws.update(n=n)
        elif name in {"dxdun_ave", "xun_ave"}:
            kws.update(n=n, xalpha=xalpha, d=d)

        elif name == "lnPi_energy":
            # already have central
            pass

        return cast("Self", func(**kws))


###############################################################################
# Factory functions
###############################################################################


@lru_cache(5)
@docfiller_shared.decorate
def factory_derivatives(
    name: SymDerivNames = "x_ave",
    n: int | None = None,
    d: int | None = None,
    xalpha: bool = False,
    central: bool | None = None,
    post_func: PostFunc = None,
    expand: bool = True,
) -> Derivatives:
    r"""
    Factory function to provide derivative function for expansion.

    Parameters
    ----------
    name : {{x_ave, u_ave, dxdun_ave, dun_ave, un_ave, xun_ave}}
    {xalpha}
    {central}
    {post_func}

    Returns
    -------
    derivatives : :class:`thermoextrap.models.Derivatives` instance
        Object used to calculate taylor series coefficients
    """
    derivs = SymDerivBeta.from_name(
        name=name,
        n=n,
        d=d,
        xalpha=xalpha,
        central=central,
        post_func=post_func,
        expand=expand,
    )
    exprs = SymSubs(
        derivs,
        subs_all={derivs.beta: "None"},
        expand=False,
        simplify=False,
    )
    return Derivatives.from_sympy(exprs, args=derivs.args)


@docfiller_shared.decorate
def factory_extrapmodel(
    beta: float,
    data: SupportsDataProtocol[DataT],
    *,
    name: str = "x_ave",
    n: int | None = None,
    d: int | None = None,
    xalpha: bool | None = None,
    central: bool | None = None,
    order: int | None = None,
    alpha_name: str = "beta",
    derivatives: Derivatives | None = None,
    post_func: PostFunc = None,
    derivatives_kws: Mapping[str, Any] | None = None,
) -> ExtrapModel[DataT]:
    """
    Factory function to create Extrapolation model for beta expansion.

    Parameters
    ----------
    {beta}
    {data}
    {n_order}
    {d_order}
    {order}
    {xalpha}
    {central}
    {post_func}
    {alpha_name}
    kws : dict
        extra arguments to `factory_data_values`

    Returns
    -------
    extrapmodel : :class:`~thermoextrap.models.ExtrapModel`


    Notes
    -----
    Note that default values for parameters ``order``, ``xalpha``, and ``central``
    are inferred from corresponding attributes of ``data``.

    See Also
    --------
    ~thermoextrap.models.ExtrapModel
    """
    if xalpha is None:
        xalpha = data.xalpha
    if central is None:
        central = data.central
    if order is None:
        order = data.order

    # assert xalpha == data.xalpha
    # assert central == data.central
    # assert order <= data.order
    if xalpha != data.xalpha:
        msg = f"{xalpha=} must equal {data.xalpha=}"
        raise ValueError(msg)
    if central != data.central:
        msg = f"{central=} must equal {data.central=}"
        raise ValueError(msg)
    if order > data.order:
        msg = f"{order=} must be <= {data.order=}"
        raise ValueError(msg)

    if derivatives is None:
        if name in {"u_ave", "un_ave", "dun_ave"} and not data.x_is_u:
            msg = "if name in [u_ave, un_ave, dun_ave] must have data.x_is_u"
            raise ValueError(msg)

        if derivatives_kws is None:
            derivatives_kws = {}
        derivatives = factory_derivatives(
            name=name,
            n=n,
            d=d,
            xalpha=xalpha,
            central=central,
            post_func=post_func,
            **derivatives_kws,
        )
    return ExtrapModel(
        alpha0=beta,
        data=data,
        derivatives=derivatives,
        order=order,
        # minus_log=mineus_log,
        alpha_name=alpha_name,
    )


@docfiller_shared.decorate
def factory_perturbmodel(
    beta: float, uv: xr.DataArray, xv: DataT, alpha_name: str = "beta", **kws: Any
) -> PerturbModel[DataT]:
    """
    Factory function to create PerturbModel for beta expansion.

    Parameters
    ----------
    {beta}
    {uv_xv_array}
    {alpha_name}
    kws : dict
        extra arguments to data object

    Returns
    -------
    perturbmodel : :class:`thermoextrap.models.PerturbModel`


    See Also
    --------
    ~thermoextrap.models.PerturbModel
    """
    # from .data import factory_data_values
    # data = factory_data_values(uv=uv, xv=xv, order=0, central=False, **kws)
    from .data import DataValues

    data = DataValues(uv=uv, xv=xv, order=0, **kws)
    return PerturbModel(alpha0=beta, data=data, alpha_name=alpha_name)
