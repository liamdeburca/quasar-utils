from logging import getLogger
from typing import Literal, Iterable, ClassVar, Self
from astropy.units import Quantity, Unit

from pydantic import validate_call

from ..utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds
from quasar_typing.pathlib import Path_, AbsoluteFilePath

logger = getLogger(__name__)

class BalmerInfo(_Info):
    _keys: ClassVar[frozenset[str]] = frozenset([
        'source', 'n_u_min', 'n_u_max', '_edge', '_dens',
        'edge', 'dens',
        '_flux', '_fwhm', '_temp', 'tau', 'scale', 'ratio',
        'flux', 'fwhm', 'temp',
        '_flux_bounds', '_fwhm_bounds', '_temp_bounds',
        'tau_bounds', 'scale_bounds', 'ratio_bounds',
        'flux_bounds', 'fwhm_bounds', 'temp_bounds',
        '_fixed', 'fixed',
        'raster_n',
        'min_fittable_ratio', 'min_fittable_total',
    ])
    _cache: ClassVar[dict[Path_, Self]] = {}

    @validate_call(validate_return=False)
    def __init__(
        self,
        *,
        source: Literal['SH1995'] = 'SH1995',
        n_u_min: int = 7,
        n_u_max: int = 50,
        _edge: float | Quantity_ = 3646 * Unit('angstrom'),
        _dens: float | Quantity_ = 1e9 * Unit('cm^-3'),
        _flux: float | Quantity_ = 1e-17 * Unit('erg/(s.cm2.angstrom)'),
        _fwhm: float | Quantity_ = 5_000 * Unit('km/s'),
        _temp: float | Quantity_ = 15_000 * Unit('K'),
        tau: float = 1.0,
        scale: float = 3.0,
        ratio: float = 0.3,
        _flux_bounds: Iterable[float] | Quantity_ = [1e-18, 1e-15] * Unit('erg/(s.cm2.angstrom)'),
        _fwhm_bounds: Iterable[float] | Quantity_ = [1000, 20_000] * Unit('km/s'),
        _temp_bounds: Iterable[float] | Quantity_ = [5_000, 30_000] * Unit('K'),
        tau_bounds: AstropyBounds = (0.1, 10.0),
        scale_bounds: AstropyBounds = (1.0, 10.0),
        ratio_bounds: AstropyBounds = (0.5, 2.0),
        _fixed: set[str] = {'temp', 'tau', 'scale', 'ratio'},
        raster_n: int = 20,
        min_fittable_ratio: float = 0.6,
        min_fittable_total: int = 100,
    ):
        self.source: Literal['SH1995'] = source
        self.n_u_min: int = n_u_min
        self.n_u_max: int = n_u_max
        self._edge: float | Quantity_ = _edge
        self._dens: float | Quantity_ = _dens

        self._flux: float | Quantity_ = _flux
        self._fwhm: float | Quantity_ = _fwhm
        self._temp: float | Quantity_ = _temp
        self.tau: float = tau
        self.scale: float = scale
        self.ratio: float = ratio

        self._flux_bounds: Iterable[float] | Quantity_ = _flux_bounds
        self._fwhm_bounds: Iterable[float] | Quantity_ = _fwhm_bounds
        self._temp_bounds: Iterable[float] | Quantity_ = _temp_bounds

        self.tau_bounds: AstropyBounds = tau_bounds
        self.scale_bounds: AstropyBounds = scale_bounds
        self.ratio_bounds: AstropyBounds = ratio_bounds
        self._fixed: set[str] = _fixed

        self.raster_n: int = raster_n
        self.min_fittable_ratio: float = min_fittable_ratio
        self.min_fittable_total: int = min_fittable_total

        # Set by 'update' method
        self.edge: float | None = None
        self.dens: float | None = None
        self.flux: float | None = None
        self.fwhm: float | None = None
        self.temp: float | None = None
        self.flux_bounds: AstropyBounds | None = None
        self.fwhm_bounds: AstropyBounds | None = None
        self.temp_bounds: AstropyBounds | None = None
        self.fixed: dict[str, bool] | None = None

    def update(self, info) -> None:
        """
        Convert to unitsless.
        """
        from collections import defaultdict

        logger.debug("Updating 'BalmerInfo' class:")

        # The Balmer edge: 'edge
        msg = ">>> [?/?] 'edge':"
        if isinstance(old := self['_edge'], Quantity):
            self['edge'] = new = info.units.getWavelength(old)
            msg += f" {val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['edge'] = float(old)
            msg += f" {val_and_type(old)}."
        logger.debug(msg)

        # The electron density: 'dens'
        msg = ">>> [?/?] 'dens':"
        if isinstance(old := self['_dens'], Quantity):
            self['dens'] = new = info.units.getDensity(old)
            msg += f" {val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['dens'] = float(old)
            msg += f" {val_and_type(old)}."
        logger.debug(msg)

        # The flux density at the Balmer edge: 'flux'
        msg = ">>> [?/?] 'flux':"
        if isinstance(old := self['_flux'], Quantity):
            self['flux'] = new = info.units.getFlux(old)
            msg += f" {val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['flux'] = float(old)
            msg += f" {val_and_type(old)}."
        logger.debug(msg)

        # The FWHM: 'fwhm'
        msg = ">>> [?/?] 'fwhm':"
        if isinstance(old := self['_fwhm'], Quantity):
            self['fwhm'] = new = info.units.getC(old)
            msg += f" {val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['fwhm'] = float(old)
            msg += f" {val_and_type(old)}."
        logger.debug(msg)

        # The electron temperature: 'temp'
        msg = ">>> [?/?] 'temp':"
        if isinstance(old := self['_temp'], Quantity):
            self['temp'] = new = info.units.getTemperature(old)
            msg += f" {val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['temp'] = float(old)
            msg += f" {val_and_type(old)}."
        logger.debug(msg)

        # Bounds on 'flux': 'flux_bounds'
        msg = ">>> [?/?] 'flux_bounds':"
        if isinstance(old := self['_flux_bounds'], Quantity):
            self['flux_bounds'] = new = tuple(info.units.getFlux(old))
            msg += f" {val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['flux_bounds'] = tuple(old)
            msg += f" {val_and_type(old)}."
        logger.debug(msg)

        # Bounds on 'fwhm': 'fwhm_bounds'
        msg = ">>> [?/?] 'fwhm_bounds':"
        if isinstance(old := self['_fwhm_bounds'], Quantity):
            self['fwhm_bounds'] = new = tuple(info.units.getC(old))
            msg += f" {val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['fwhm_bounds'] = tuple(old)
            msg += f" {val_and_type(old)}."
        logger.debug(msg)

        # Bounds on 'temp': 'temp_bounds'
        msg = ">>> [?/?] 'temp_bounds':"
        if isinstance(old := self['_temp_bounds'], Quantity):
            self['temp_bounds'] = new = tuple(info.units.getTemperature(old))
            msg += f" {val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['temp_bounds'] = tuple(old)
            msg += f" {val_and_type(old)}."
        logger.debug(msg)

        # Fixed parameters: 'fixed'
        msg = ">>> [?/?] 'fixed':"
        if self['fixed'] is None:
            self['fixed'] = defaultdict(lambda: False)
            for key in self._fixed:
                self['fixed'][key] = True

            msg += f" {self['_fixed']} have been fixed."
        else:
            msg += f" {self['_fixed']} are already fixed."
        logger.debug(msg)
 
        logger.debug("... finished updating 'BalmerInfo' class.")

    @classmethod
    @validate_call(validate_return=False)
    def from_file(
        cls, 
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        
        if path is not None and path in cls._cache.keys():
            logger.debug(f"Using cached 'BalmerInfo' for '{path}'.")
            
            binfo = cls._cache[path]
            if create_copy: return binfo.copy()
            else:           return binfo

        binfo: BalmerInfo = BalmerInfo()
        if path is None: return binfo

        logger.debug(f"Configuring 'BalmerInfo' using '{path}':")
        lines = get_lines_from_file.__wrapped__(logger, 'BALMER', path)

        for count, line in enumerate(lines, start=1):
            prefix: str = ''
            key: str = line[0].lower()

            match key:
                case 'source':
                    val = parsing.as_str(line[1])

                case 'n_u_min' | 'n_u_max' | 'min_fittable_total' | 'raster_n':
                    val = parsing.as_int(line[1])

                case 'min_fittable_ratio' | 'tau' | 'scale' | 'ratio':
                    val = parsing.as_float(line[1])

                case 'dens' | 'edge' | 'flux' | 'fwhm' | 'temp':
                    prefix: str = '_'
                    val = parsing.as_scalar_or_quantity(line[1:])

                case 'tau_bounds' | 'scale_bounds' | 'ratio_bounds':
                    val = parsing.as_bounds(line[1:])

                case 'flux_bounds' | 'fwhm_bounds' | 'temp_bounds':
                    prefix: str = '_'
                    val = parsing.as_bounds_of_scalars_or_quantity(line[1:])

            binfo[prefix + key] = val
            logger.debug(
                f">>> [{count}/{len(lines)}] '{prefix + key}': " \
                f"{val_and_type(val)}"
            )

        return binfo