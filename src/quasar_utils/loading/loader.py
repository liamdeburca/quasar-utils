__all__ = ["_Loader"]

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, TypedDict

from numpy import diff, float64, interp, log, ndarray, ones
from pydantic import BaseModel, Field
from quasar_typing.astropy import QuantityType
from quasar_typing.bounds import CoordBounds
from quasar_typing.literals import LawName, MapName
from quasar_typing.numpy import FloatMatrix, FloatVector
from quasar_typing.pathlib import FilePath
from quasar_typing.scipy import CSRMatrix

from quasar_utils.decorators import validated_apply_info_to_method
from quasar_utils.resolving import create_kernels
from quasar_utils.setup import Info

from .metadata import BaseMeta

logger = getLogger(__name__)


class LoaderOutput(TypedDict):
    meta: dict[str, Any]

    x_original: FloatVector
    y_original: FloatVector
    dy_original: FloatVector
    dx_original: FloatVector
    kernels_original: FloatMatrix | CSRMatrix

    x: FloatVector
    y: FloatVector
    dy: FloatVector
    dx: FloatVector
    kernels: FloatMatrix | CSRMatrix

    info: Info
    x_bounds: CoordBounds


class _Loader(BaseModel, ABC):
    """
    Loader helper that encapsulates a single spectrum's loading and
    coordinate-transformation pipeline.
    """
    path: FilePath
    ra: float
    dec: float
    z: float

    x_original: FloatVector | QuantityType
    y_original: FloatVector | QuantityType
    dy_original: FloatVector | QuantityType
    dx_original: FloatVector | QuantityType
    R_original: FloatVector | None
    
    info: Info = Field(default_factory=Info)
    title: str = "missing_title"
    luminosity_distance_mpc: float | None = None

    x: FloatVector | None = Field(None, init=False)
    y: FloatVector | None = Field(None, init=False)
    dy: FloatVector | None = Field(None, init=False)
    dx: FloatVector | None = Field(None, init=False)
    R: FloatVector | None = Field(None, init=False)
    
    kernels_original: FloatMatrix | CSRMatrix | None = Field(None, init=False)
    kernels: FloatMatrix | CSRMatrix | None = Field(None, init=False)

    x_bounds: CoordBounds | None = Field(None, init=False)

    has_been: set[str] = Field(default_factory=set)

    @validated_apply_info_to_method(subjects=("loading",))
    def run(
        self,
        *,
        deredden: tuple[bool, MapName, LawName, float] | None = None,
        rebin: bool | None = None,
    ) -> None:
        """
        Run the default loading pipeline and return processed data.

        This method orchestrates the sequence of coordinate transforms
        and corrections: unit conversion, optional dereddening,
        optional logarithmic re-binning, optional redshift
        correction, and final conversion to C-contiguous arrays.

        Parameters
        ----------
        deredden : tuple[bool, MapName, LawName, float], optional
            If provided, a 4-tuple used to control dereddening:
            ``(apply: bool, map_name: MapName, law_name: LawName, Rv: float)``. If the
            first element is ``False`` the deredden step is skipped.
            When ``None`` the decorator :func:`validated_apply_info_to_method`
            supplies defaults from ``self.info``.
        rebin : bool | None, optional
            Whether to perform logarithmic re-binning. If ``None`` the
            info-loading defaults are applied via the decorator.

        Raises
        ------
        ValidationError
            Pydantic type validation
        """
        logger.debug("Running %s loading pipeline.", self.__class__.__name__)
        logger.debug("(1) `transform_coords`")
        self.transform_coords()
        
        logger.debug("(2) `deredden_coords(deredden = %s)`", deredden)
        self.deredden_coords(deredden)

        logger.debug("(3) `redshift_correct_coords`")
        self.redshift_correct_coords()

        logger.debug("(4) `logbin_coords(rebin = %s)`", rebin)
        self.logbin_coords(rebin)

        logger.debug("(5) `create_kernels`")
        self.create_kernels()

        logger.debug("(6) `calculate_luminosity_distance`")
        self.calculate_luminosity_distance()

        logger.debug("(7) `finalize_coords`")
        self.finalize_coords()

        logger.debug("(8) `set_x_bounds`")
        self.set_x_bounds()


    @abstractmethod
    def _get_metadata(self) -> dict[str, Any]:
        pass


    @abstractmethod
    def get_metadata(self) -> BaseMeta:
        pass


    def __call__(self) -> LoaderOutput:
        return {
            "meta": self._get_metadata(),
            "x": self.x,
            "y": self.y,
            "dy": self.dy,
            "dx": self.dx,
            "kernels": self.kernels,
            "x_original": self.x_original,
            "y_original": self.y_original,
            "dy_original": self.dy_original,
            "dx_original": self.dx_original,
            "kernels_original": self.kernels_original,
            "info": self.info,
            "x_bounds": self.x_bounds,
        }

    ###

    def transform_coords(self) -> None:
        """
        Returns the original coordinates as unitless numpy arrays. For security, 
        these arrays are then made read-only as all future operations will be on 
        rebinned coordinates.
        """
        from .utils import _transform_coords

        if "transformed" in self.has_been:
            logger.debug("> Coordinates have already been transformed.")
            return

        logger.debug(
            "> Transforming coordinates: x=%s, y=%s, dy=%s, dx=%s.",
            not isinstance(self.x_original, ndarray),
            not isinstance(self.y_original, ndarray),
            not isinstance(self.dy_original, ndarray),
            not isinstance(self.dx_original, ndarray),
        )
        out = _transform_coords(
            (self.x_original, self.y_original, self.dy_original),
            self.dx_original,
            self.info,
        )
        self.has_been.add("transformed")

        self.x_original = out[0][0] 
        self.y_original = out[0][1] 
        self.dy_original = out[0][2] 
        self.dx_original = out[1]

        self.x_original.setflags(write=False)
        self.y_original.setflags(write=False)
        self.dy_original.setflags(write=False)
        self.dx_original.setflags(write=False)

    def deredden_coords(self, deredden: tuple[bool, MapName, LawName, float]) -> None:
        """
        Return coordinates after applying Galactic dereddening. If this has 
        already been done, this property does nothing except returning the 
        original data arrays.
        """
        from .utils import _deredden_coords

        if not deredden[0]:
            logger.debug("> Dereddening is disabled.")
            return 

        if "dereddened" in self.has_been:
            logger.debug("> Dereddening has already been applied.")
            return
        
        if self.ra is None:
            logger.debug("> Skipping dereddening due to missing RA.")
            return
        
        if self.dec is None:
            logger.debug("> Skipping dereddening due to missing DEC.")
            return

        logger.debug(
            "> Applying dereddening: RA=%.1f, DEC=%.1f, MAP=%s, LAW=%s, Rv=%.2f.",
            self.ra,
            self.dec,
            deredden[1],
            deredden[2],
            deredden[3],
        )
        dereddened_coords = _deredden_coords(
            (self.x_original, self.y_original, self.dy_original),
            self.ra,
            self.dec,
            deredden[1],
            deredden[2],
            self.info.units.wavelength_unit,
            deredden[3],
        )
        self.has_been.add("dereddened")

        self.y_original.setflags(write=True)
        self.y_original[:] = dereddened_coords[1]
        self.y_original.setflags(write=False)
        
        self.dy_original.setflags(write=True)
        self.dy_original[:] = dereddened_coords[2]
        self.dy_original.setflags(write=False)


    def redshift_correct_coords(self) -> None:
        """
        Return coordinates and spacing after applying redshift correction.
        """
        from .utils import _redshift_correct_coords

        if "redshift_corrected" in self.has_been:
            logger.debug("> Redshift correction has already been applied.")
            return

        if self.z is None or self.z == 0.0:
            logger.debug(
                "> Skipping redshift correction due to missing or zero "
                "redshift."
            )
            return

        logger.debug("> Applying redshift correction: z=%.3f.", self.z)
        out = _redshift_correct_coords(
            (self.x_original, self.y_original, self.dy_original),
            self.dx_original,
            self.z,
        )
        self.has_been.add("redshift_corrected")
        
        self.x_original.setflags(write=True)
        self.x_original[:] = out[0][0]
        self.x_original.setflags(write=False)

        self.y_original.setflags(write=True)
        self.y_original[:] = out[0][1]
        self.y_original.setflags(write=False)

        self.dy_original.setflags(write=True)
        self.dy_original[:] = out[0][2]
        self.dy_original.setflags(write=False)

        self.dx_original.setflags(write=True)
        self.dx_original[:] = out[1]
        self.dx_original.setflags(write=False)


    def logbin_coords(self, rebin: bool) -> None:
        """
        Return coordinates after logarithmic re-binning using ``Info``
        defaults.
        """
        from .utils import _logbin_coords

        if not rebin:
            theo: str = self.info.units.formatC(
                self.info.loading.sigma_res, 
                with_unit=False,
            )
            _sigma_res_obs = diff(log(self.x_original))
            obs_mean: str = self.info.units.formatC(
                _sigma_res_obs.mean(),
                with_unit=False,
            )
            obs_std: str = self.info.units.formatC(
                _sigma_res_obs.std(),
                with_unit=False,
            )
            logger.debug(
                "> Skipping logarithmic re-binning: "
                "assuming 'sigma_res = %s', using 'sigma_res = %s ± %s'.",
                theo, obs_mean, obs_std,
            )
            return

        logger.debug("> Performing logarithmic re-binning.")
        out = _logbin_coords(
            (self.x_original, self.y_original, self.dy_original),
            self.dx_original,
            self.info.loading.sigma_res,
            self.info.loading.conserve,
            self.info.loading.covariance,
        )

        self.x_original.setflags(write=True)
        self.x = out[0][0]
        self.x_original.setflags(write=False)

        self.y_original.setflags(write=True)
        self.y = out[0][1]
        self.y_original.setflags(write=False)

        self.dy_original.setflags(write=True)
        self.dy = out[0][2]
        self.dy_original.setflags(write=False)

        self.dx_original.setflags(write=True)
        self.dx = out[1]
        self.dx_original.setflags(write=False)

        # Interpolate new 'res_kernels'
        if self.R_original is None:
            logger.debug(
                "> No spectral resolution available, skipping re-interpolation."
            )
            self.R = None
        else:
            logger.debug(
                "> Re-interpolating spectral resolution."
            )
            self.R = interp(self.x, self.x_original, self.R_original)
            self.R.setflags(write=False)

    def create_kernels(self) -> None:

        logger.debug("> Creating resolution kernels.")
        if self.R_original is None:
            self.kernels_original = ones((self.x_original.size, 1), float64)
            logger.debug(
                "> [ORIGINAL] No spectral resolution available, "
                "using default kernels: %s.",
                self.kernels_original.shape,
            )
        else:
            self.kernels_original = create_kernels(
                self.R_original, self.x_original, self.dx_original,
                w=self.info.loading.kernel_w,
                n_sigma=self.info.loading.kernel_n_sigma,
                normalize=self.info.loading.kernel_normalize,
                sparse=False,
            )
            logger.debug(
                "> [ORIGINAL] Creating spectral resolution kernels "
                "(w = %s, n_sigma = %s, normalize = %s, sparse = %s): %s.",
                self.info.loading.kernel_w,
                self.info.loading.kernel_n_sigma,
                self.info.loading.kernel_normalize,
                False,
                self.kernels_original.shape
            )

        if self.x is not None:            
            logger.debug(
                "> Creating resolution kernels for log-binned data: %s.", 
                self.x.size,
            )
            if self.R is None:
                logger.debug(
                    "> No spectral resolution available, "
                    "using default kernels: %s.",
                    self.kernels_original.shape,
                )
                self.kernels: FloatMatrix = ones((self.x.size, 1), float64)
            else:
                self.kernels: FloatMatrix = create_kernels(
                    self.R, self.x, self.dx,
                    w=self.info.loading.kernel_w,
                    n_sigma=self.info.loading.kernel_n_sigma,
                    normalize=self.info.loading.kernel_normalize,
                    sparse=False,
                )
                logger.debug(
                    "> Creating spectral resolution kernels "
                    "(w = %s, n_sigma = %s, normalize = %s, sparse = %s): %s.",
                    self.info.loading.kernel_w,
                    self.info.loading.kernel_n_sigma,
                    self.info.loading.kernel_normalize,
                    False,
                    self.kernels.shape
                )
        else:
            logger.debug(
                "> No log-binned data available, using original kernels: %s.",
                self.kernels_original.shape,
            )
            self.kernels = self.kernels_original

        self.kernels_original.setflags(write=False)
        self.kernels.setflags(write=False)

    def calculate_luminosity_distance(self) -> None:
        from .utils import _get_luminosity_distance
        self.luminosity_distance_mpc = _get_luminosity_distance(
            self.info.cosmology.cosmology,
            self.z,
        )
        
    def finalize_coords(self) -> None:
        from .utils import _finalize_coords

        coords = [
            self.x_original,
            self.y_original,
            self.dy_original,
            self.dx_original,
        ]
        if self.x is not None:
            coords.extend([
                self.x,
                self.y,
                self.dy,
                self.dx,
            ])
        out = _finalize_coords(*coords)
        self.x_original = out[0]
        self.y_original = out[1]
        self.dy_original = out[2]
        self.dx_original = out[3]

        if self.x is None:
            self.x = self.x_original
            self.y = self.y_original
            self.dy = self.dy_original
            self.dx = self.dx_original
        else:
            self.x = out[4]
            self.y = out[5]
            self.dy = out[6]
            self.dx = out[7]

    def set_x_bounds(self) -> None:
        self.x_bounds = (
            max(self.x[0], self.info.loading.x_bounds[0]),
            min(self.x[-1], self.info.loading.x_bounds[1])
        )