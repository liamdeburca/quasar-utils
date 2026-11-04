"""
Loader-specific metadata classes. 
"""
from typing import Annotated, ClassVar

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveInt,
    StringConstraints,
)
from quasar_typing.pathlib import FITSPath


class BaseMeta(BaseModel, frozen=True):
    """
    Base class with general metadata:

    Attributes
    ----------
    path : FITSPath
        Path to the FITS file used to load the data.
    title : str
        Title of the object.
    ra : float
        Right ascension of the object in degrees. Must be in the range [0, 360).
    dec : float
        Declination of the object in degrees. Must be in the range [-90, 90].
    z : float
        Redshift of the object. Must be non-negative.
    luminosity_distance_mpc : float
        Luminosity distance to the object in megaparsecs. Must be non-negative.
    plate : int
        SDSS plate number. Must be positive or -1 if not applicable.
    fiber : int
        SDSS fiber number. Must be positive or -1 if not applicable.
    mjd : int
        SDSS modified Julian date. Must be positive or -1 if not applicable.
    obj_nme : str | None
        Name of the object in the PAQS catalog. Can be None if not applicable.
    obj_uid : int | None
        Unique identifier of the object in the PAQS catalog. Can be None if not 
        applicable.
    specuid : int | None
        Unique identifier of the spectrum in the PAQS catalog. Can be None if 
        not applicable.

    Notes
    -----
    For the `SDSSMeta` class, the PAQS-specific attributes (`obj_nme`, 
    `obj_uid`, `specuid`) are always None.
    """
    path: FITSPath
    title: Annotated[str, StringConstraints(min_length=1)]
    ra: Annotated[float, Field(ge=0.0, lt=360.0)]
    dec: Annotated[float, Field(ge=-90.0, le=90.0)]

    z: NonNegativeFloat
    luminosity_distance_mpc: NonNegativeFloat

    ### SDSS

    plate: PositiveInt | None = None
    fiber: PositiveInt | None = None
    mjd: PositiveInt | None = None

    ### PAQS

    obj_nme: Annotated[str, StringConstraints(min_length=1)] | None = None
    obj_uid: int | None = None
    specuid: int | None = None

    def pprint(self, indent: int = 2) -> None:
        cls = self.__class__
        s = f"{cls.__name__}(\n"
        for field_name in cls.model_fields:
            s += f"{' ' * indent}{field_name}: {getattr(self, field_name)}\n"
        s += ")" 
        print(s)


class SDSSMeta(BaseMeta):
    obj_nme: ClassVar[None] = None
    obj_uid: ClassVar[None] = None
    specuid: ClassVar[None] = None


class PAQSMeta(BaseMeta):
    ...