"""
This submodule contains modified class definition from the `dust_extinction` 
package.
"""
__all__ = ['get_dust_law']

from typing import Literal, Union
from .ccm89 import CCM89
from .o94 import O94

def get_dust_law(law_name: Literal['ccm89', 'o94']) -> Union[CCM89, O94]:
    match law_name.strip().lower():
        case 'ccm89': 
            return CCM89()
        case 'o94': 
            return O94()
        case _: 
            raise ValueError(
                f"Dust curve '{law_name}' is not supported. Supported curves are: \
                    'ccm89', 'o94'.",
            )
