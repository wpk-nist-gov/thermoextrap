"""Validation routines"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def validate_positive_integer(value: Any, name: str = "value") -> int:
    """Validate that value is a positive integer"""
    if isinstance(value, int):
        if value < 0:
            msg = f"{name}={value} must be a positive integer."
            raise ValueError(msg)
        return value

    msg = f"{name} must be an integer."
    raise TypeError(msg)
