import math

from astropy.units import Unit
from quasar_typing.astropy import QuantityType


class J2000:
    template: str = "RA = {ra}, DEC = {dec}"

    @classmethod
    def get_name(cls, ra: QuantityType, dec: QuantityType) -> str:
        return cls.template.format(
            ra=cls.right_ascension(ra),
            dec=cls.declination(dec),
        )

    @classmethod
    def right_ascension(cls, ra: QuantityType) -> str:
        if ra < 0:
            ra += 360 * Unit("degree")

        val = ra.to("hourangle").value
        hours = math.floor(val)
        minutes = math.floor(60 * (val - hours))
        seconds = 3600 * (val - hours - minutes / 60)
        return f"{hours:0>2d}h{minutes:0>2d}m{seconds:0>6.3f}s"

    @classmethod
    def declination(cls, dec: QuantityType) -> str:
        sign = "+" if dec >= 0 else "-"
        val = abs(dec.to("degree").value)
        hours = math.floor(val)
        minutes = math.floor(60 * (val - hours))
        seconds = 3600 * (val - hours - minutes / 60)
        return f"{sign}{hours:0>2d}°{minutes:0>2d}'{seconds:0>6.3f}\""


class IGR:
    template: str = "IGR J{ra}{dec}"

    @classmethod
    def get_name(cls, ra: QuantityType, dec: QuantityType) -> str:
        return cls.template.format(
            ra=cls.right_ascension(ra),
            dec=cls.declination(dec),
        )

    @classmethod
    def right_ascension(cls, ra: QuantityType) -> str:
        if ra < 0:
            ra += 360 * Unit("degree")

        val = ra.to("hourangle").value
        hours = math.floor(val)
        minutes = math.floor(60 * (val - hours))
        return f"{hours:0>2d}{minutes:.1f}"

    @classmethod
    def declination(cls, dec: QuantityType) -> str:
        sign = "+" if dec >= 0 else "-"
        val = abs(dec.to("degree").value)
        hours = math.floor(val)
        minutes = math.floor(60 * (val - hours))
        return f"{sign}{hours:0>2d}{minutes:0>2d}"


class SDSS:
    template: str = (
        "Plate = {plate:0>4d}, Fiber = {fiber:0>3d}, MJD = {mjd:0>5d}"
    )

    @classmethod
    def get_name(cls, plate: int, fiber: int, mjd: int) -> str:
        return cls.template.format(
            plate=plate,
            fiber=fiber,
            mjd=mjd,
        )
