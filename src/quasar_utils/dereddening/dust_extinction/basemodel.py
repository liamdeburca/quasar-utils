"""
Simplified class based on `dust_extinction.baseclasses.BaseExtRvModel`.
"""
from typing import Self
from hashlib import md5
from astropy.modeling import Fittable1DModel, Parameter
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from quasar_typing.numpy import FloatVector

class BaseModel(Fittable1DModel):
    r"""
    ** NEW BASE CLASS **

    Base class for extinction models that depend on R(V) = A(V)/E(B-V).

    Parameters
    ----------
    Rv: float
        R(V) = A(V)/E(B-V) = total-to-selective extinction
    """
    Rv = Parameter(
        default=3.1,
        bounds=(2.0, 6.0),
        name="R(V)",
        description="R(V) = A(V)/E(B-V) = total-to-selective extinction",
    )

    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, cls):
            msg = f"Expected an instance of {cls.__name__}, \
                got {type(value).__name__}"
            raise PydanticCustomError("validation_error", msg)
        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return no_info_plain_validator_function(cls._validate)
    
    @classmethod
    def get_cache_key(cls, k: FloatVector) -> str:
        cache_key = md5(k.tobytes()).hexdigest()
        return cache_key
    
    @classmethod
    def get_ab_arrays(cls, k: FloatVector) -> tuple[FloatVector, FloatVector]:
        """
        Parameters
        ----------
        k: numpy.array
            Wavenumber array (in units of `x_unit`) at which to evaluate the
            extinction curve.
        """
        raise NotImplementedError
    
    @classmethod
    def evaluate(cls, k, Rv):
        a, b = cls.get_ab_arrays(k)
        return a + b / Rv
    
    @classmethod
    def fit_deriv(cls, k, Rv):
        b = cls.get_ab_arrays(k)[1]
        return [-b / Rv**2]
