"""dust_extinction compatibility helpers.

This module exposes a small helper to return modified local classes for the
commonly used dust-extinction laws. The module follows the project's
public/private pattern: public wrappers are validated with Pydantic and the
implementation lives in a private function decorated with ``functools.wraps``.
"""
__all__ = ["get_dust_law"]

from functools import wraps
from typing import Union

from pydantic import validate_call
from quasar_typing.literals import LawName

from .ccm89 import CCM89
from .o94 import O94

### Public functions


@validate_call
def get_dust_law(law_name: LawName) -> Union[CCM89, O94]:
    """Return a dust extinction law instance by name.

    Public (validated) wrapper around :func:`_get_dust_law`.
    """
    return _get_dust_law(law_name)

### Private functions

@wraps(get_dust_law)
def _get_dust_law(law_name: LawName) -> Union[CCM89, O94]:
    match law_name:
        case "ccm89":
            return CCM89()
        case "o94":
            return O94()
    raise ValueError(
        f"Dust curve '{law_name}' is not supported. "
        "Supported curves are: 'ccm89', 'o94'."
    )
