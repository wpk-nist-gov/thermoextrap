"""Validation routines"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cmomy.core.validate import is_xarray

if TYPE_CHECKING:
    from typing import Any

    import attrs


# * Validate
# func(val) -> val
def validate_positive_integer(value: Any, name: str = "value") -> int:
    """Validate that value is a positive integer"""
    if isinstance(value, int):
        if value < 0:
            msg = f"{name}={value} must be a positive integer."
            raise ValueError(msg)
        return value

    msg = f"{name} must be an integer."
    raise TypeError(msg)


# * Validators
# Used by attrs
def validator_dims(self: Any, attribute: attrs.Attribute[Any], dims: Any) -> None:  # noqa: ARG001
    """Attrs validator for dimensions"""
    for d in dims:
        if d not in self.data.dims:
            msg = f"{d} not in data.dimensions {self.data.dims}"
            raise ValueError(msg)


def validator_xarray_typevar(
    self: Any,  # noqa: ARG001
    attribute: attrs.Attribute[Any],  # noqa: ARG001
    x: Any,
) -> None:
    """Attrs validator for xarray"""
    if not is_xarray(x):
        msg = f"Must pass xarray object.  Passed {type(x)}"
        raise TypeError(msg)
