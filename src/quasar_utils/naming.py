from math import floor
from astropy.units import Unit

from quasar_typing.astropy import Quantity_

class J2000:
    template: str = "RA = {ra}, DEC = {dec}"
    @classmethod
    def get_name(cls, ra: Quantity_, dec: Quantity_) -> str:
        return cls.template.format(
            ra=cls.right_ascension(ra), 
            dec=cls.declination(dec),
        )
    
    @classmethod
    def right_ascension(cls, ra: Quantity_) -> str:
        if ra < 0: ra += 360 * Unit('degree')
        
        val = ra.to('hourangle').value
        return "{:0>2d}h{:0>2d}m{:0>6.3f}s".format(
            hours := int(floor(val)),
            minutes := int(floor(60 * (val - hours))),
            3600 * (val - hours - minutes / 60),
        )
    
    @classmethod
    def declination(cls, dec: Quantity_) -> str:
        sign = '+' if dec >= 0 else '-'
        val = abs(dec.to('degree').value)
        return '{}{:0>2d}°{:0>2d}\'{:0>6.3f}\"'.format(
            sign,
            hours := int(floor(val)),
            minutes := int(floor(60 * (val - hours))),
            int(floor(3600 * (val - hours - minutes / 60))),
        )

class IGR:
    template: str = "IGR J{ra}{dec}"

    @classmethod
    def get_name(cls, ra: Quantity_, dec: Quantity_) -> str:
        return cls.template.format(
            ra=cls.right_ascension(ra), 
            dec=cls.declination(dec),
        )
    
    @classmethod
    def right_ascension(cls, ra: Quantity_) -> str:
        if ra < 0: ra += 360 * Unit('degree')
        
        val = ra.to('hourangle').value
        return "{:0>2d}{:.1f}".format(
            hours := int(floor(val)),
            int(floor(60 * (val - hours))),
        )
    
    @classmethod
    def declination(cls, dec: Quantity_) -> str:
        sign = '+' if dec >= 0 else '-'
        val = abs(dec.to('degree').value)
        return '{}{:0>2d}{:0>2d}'.format(
            sign,
            hours := int(floor(val)),
            int(floor(60 * (val - hours))),
        )
    
class SDSS:
    template: str = "Plate = {plate:0>4d}, Fiber = {fiber:0>3d}, MJD = {mjd:0>5d}"

    @classmethod
    def get_name(cls, plate: int, fiber: int, mjd: int) -> str:
        return cls.template.format(
            plate=plate,
            fiber=fiber,
            mjd=mjd,
        )