__all__ = ["get_dust_law"]

from pydantic import validate_call
from quasar_typing.literals import LawName

### Public functions


@validate_call
def get_dust_law(law_name: LawName):
    """Return a dust extinction law instance by name.

    Public (validated) wrapper around :func:`_get_dust_law`.
    """
    return _get_dust_law(law_name)

### Private functions

def _get_dust_law(law_name: LawName):
    match law_name:
        case "ccm89":
            from .ccm89 import CCM89
            return CCM89()
        case "o94":
            from .o94 import O94
            return O94()