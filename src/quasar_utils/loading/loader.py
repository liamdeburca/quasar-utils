__all__ = ["_Loader"]

from dataclasses import field
from logging import getLogger
from typing import Literal

from numpy import diff, float64, full, interp, log, ones, stack
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import CoordBounds
from quasar_typing.numpy import CoordsTuple, FloatMatrix, FloatVector
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validated_apply_info_to_method
from ..setup import Info
from .utils import (
    LoaderOutput,
    _deredden_coords,
    _finalize_coords,
    _logbin_coords,
    _redshift_correct_coords,
    _transform_coords,
)

logger = getLogger(__name__)


@dataclass
class _Loader:
    """
    Loader helper that encapsulates a single spectrum's loading and
    coordinate-transformation pipeline.
    """
    path: str | AbsoluteFilePath
    info: Info = field(default_factory=Info)
    z: float | None = None

    title: str = "missing_title"
    ra: float | None = None
    dec: float | None = None

    x_original: FloatVector | Quantity_ = field(init=False)
    y_original: FloatVector | Quantity_ = field(init=False)
    dy_original: FloatVector | Quantity_ = field(init=False)
    dx_original: float | FloatVector | Quantity_ = field(init=False)
    res_kernels_original: FloatMatrix = field(init=False)

    x: FloatVector = field(init=False)
    y: FloatVector = field(init=False)
    dy: FloatVector = field(init=False)
    dx: FloatVector = field(init=False)
    res_kernels: FloatMatrix = field(init=False)

    x_bounds: CoordBounds = field(init=False)

    _is_transformed: bool = field(default=False, init=False)
    _is_dereddened: bool = field(default=False, init=False)
    _is_redshift_corrected: bool = field(default=False, init=False)

    def __post_init__(self):
        """
        Post-initialization hook.
        """
        assert hasattr(self, "x_original")
        assert hasattr(self, "y_original")
        assert hasattr(self, "dy_original")
        assert hasattr(self, "dx_original")
        assert hasattr(self, "res_kernels_original")

        if isinstance(self.dx_original, float):
            self.dx_original = full(
                self.x_original.size, 
                self.dx_original, 
                dtype=float64,
            )

        # Make original arrays read-only
        self.x_original.setflags(write=False)
        self.y_original.setflags(write=False)
        self.dy_original.setflags(write=False)
        self.dx_original.setflags(write=False)
        self.res_kernels_original.setflags(write=False)

    @validated_apply_info_to_method(subjects=("loading",))
    def __call__(
        self,
        *,
        deredden: tuple[
            bool, 
            Literal["sfd", "csfd"], 
            Literal["ccm89", "o94"], 
            float,
        ] | None = None,
        rebin: bool | None = None,
    ) -> LoaderOutput:
        """
        Run the default loading pipeline and return processed data.

        This method orchestrates the sequence of coordinate transforms
        and corrections: unit conversion, optional dereddening,
        optional logarithmic re-binning, optional redshift
        correction, and final conversion to C-contiguous arrays.

        Parameters
        ----------
        deredden : tuple or None, optional
            If provided, a 4-tuple used to control dereddening:
            ``(apply: bool, map_name: Literal['sfd','csfd'],
            law_name: Literal['ccm89','o94'], Rv: float)``. If the
            first element is ``False`` the deredden step is skipped.
            When ``None`` the decorator :func:`validated_apply_info_to_method`
            supplies defaults from ``self.info``.
        sigma_res : float | None, optional
            Target spectral resolution used for logarithmic re-binning
            (used when ``rebin`` is True).
        res_kernels : array-like | None, optional
            The original resolution kernels associated with the data.
        rebin : bool | None, optional
            Whether to perform logarithmic re-binning. If ``None`` the
            info-loading defaults are applied via the decorator.
        conserve : bool | None, optional
            Whether to conserve flux during re-binning. If ``None``
            defaults are applied via the decorator.
        covariance : bool | None, optional
            Whether to return covariance information during re-binning.

        Returns
        -------
        LoaderOutput

        Raises
        ------
        ValidationError
            Pydantic type validation
        """

        msg = f"Running {self.__class__.__name__} loading pipeline: "

        msg += "(1) creating unitless coordinates, "
        self.transform_coords()

        if deredden[0]:
            if self.ra is None:
                msg += "(2) skipping dereddening due to missing coordinates RA, "
            elif self.dec is None:
                msg += "(2) skipping dereddening due to missing coordinates DEC, "
            else:                
                msg += f"(2) applying dereddening correction [RA={self.ra}, DEC={self.dec}, map={deredden[1]}, law={deredden[2]}], Rv={deredden[3]}], "
                self.deredden_coords()
        else:
            msg += "(2) skipping dereddening correction, "

        if self.z != 0.0:
            msg += "(3) applying redshift correction, "
            self.redshift_correct_coords()
        elif self.z is None:
            msg += "(3) skipping redshift correction due to missing redshift, "
        else:
            msg += "(3) spectrum is already in the restframe (z=0.0), "

        if rebin:
            msg += "(4) logarithmic re-binning, "
            self.logbin_coords()
        else:
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
            msg += f"(4) skipping logarithmic re-binning "\
                f"[assuming 'sigma_res'={theo}, actual "\
                f"'sigma_res'={obs_mean}±{obs_std}]"


        msg += "(5) ensuring coordinates are C-contiguous and read-only, "
        self.finalize_coords()

        msg += f"(6) setting x bounds using {self.info.loading.x_bounds}."
        self.set_x_bounds()

        logger.debug(msg)

        return {
            "path": self.path,
            "title": self.title,
            "x": self.x,
            "y": self.y,
            "dy": self.dy,
            "dx": self.dx,
            "res_kernels": self.res_kernels,
            "x_original": self.x_original,
            "y_original": self.y_original,
            "dy_original": self.dy_original,
            "dx_original": self.dx_original,
            "res_kernels_original": self.res_kernels_original,
            "info": self.info,
            "x_bounds": self.x_bounds,
        }

    def transform_coords(self) -> None:
        """
        Returns the original coordinates as unitless numpy arrays. For security, 
        these arrays are then made read-only as all future operations will be on 
        rebinned coordinates.
        """
        if not self._is_transformed:
            out = _transform_coords(
                (self.x_original, self.y_original, self.dy_original),
                self.dx_original,
                self.info,
            )
            self._is_transformed = True

            self.x_original = out[0][0] 
            self.y_original = out[0][1] 
            self.dy_original = out[0][2] 
            self.dx_original = out[1]

        self.x_original.setflags(write=False)
        self.y_original.setflags(write=False)
        self.dy_original.setflags(write=False)
        self.dx_original.setflags(write=False)

    def deredden_coords(self) -> CoordsTuple:
        """
        Return coordinates after applying Galactic dereddening. If this has 
        already been done, this property does nothing except returning the 
        original data arrays.
        """
        if not self._is_dereddened:
            assert self.ra is not None, "RA is required for dereddening"
            assert self.dec is not None, "DEC is required for dereddening"

            out = _deredden_coords(
                (self.x_original, self.y_original, self.dy_original),
                self.ra,
                self.dec,
                self.info.loading.deredden[1],
                self.info.loading.deredden[2],
                self.info.units.wavelength_unit,
                self.info.loading.deredden[3],
            )
            self._is_dereddened = True

            self.y_original.setflags(write=True)
            self.dy_original.setflags(write=True)
            self.y_original[:] = out[1]
            self.dy_original[:] = out[2]

        self.y_original.setflags(write=False)
        self.dy_original.setflags(write=False)

    def redshift_correct_coords(self) -> None:
        """
        Return coordinates and spacing after applying redshift correction.
        """
        if not self._is_redshift_corrected:
            out = _redshift_correct_coords(
                (self.x_original, self.y_original, self.dy_original),
                self.dx_original,
                self.z,
            )
            self._is_redshift_corrected = True
            
            self.x_original.setflags(write=True)
            self.y_original.setflags(write=True)
            self.dy_original.setflags(write=True)
            self.dx_original.setflags(write=True)
            self.x_original[:] = out[0][0]
            self.y_original[:] = out[0][1]
            self.dy_original[:] = out[0][2]
            self.dx_original[:] = out[1]

        self.x_original.setflags(write=False)
        self.y_original.setflags(write=False)
        self.dy_original.setflags(write=False)
        self.dx_original.setflags(write=False)

    def logbin_coords(self) -> None:
        """
        Return coordinates after logarithmic re-binning using ``Info``
        defaults.
        """
        self.x_original.setflags(write=True)
        self.y_original.setflags(write=True)
        self.dy_original.setflags(write=True)
        self.dx_original.setflags(write=True)

        out = _logbin_coords(
            (self.x_original, self.y_original, self.dy_original),
            self.dx_original,
            self.info.loading.sigma_res,
            self.info.loading.conserve,
            self.info.loading.covariance,
        )

        self.x_original.setflags(write=False)
        self.y_original.setflags(write=False)
        self.dy_original.setflags(write=False)
        self.dx_original.setflags(write=False)
        self.x = out[0][0]
        self.y = out[0][1]
        self.dy = out[0][2]
        self.dx = out[1]

        # Interpolate new 'res_kernels'
        if self.res_kernels.shape[0] == 1:
            self.res_kernels = ones((1, self.x.size), dtype=float64)
        else:
            self.res_kernels = stack(
                [
                    interp(self.x, self.x_original, ks) 
                    for ks in self.res_kernels_original
                ],
                axis=0,
            )
        self.res_kernels.setflags(write=False)

    def finalize_coords(self) -> None:
        coords = [
            self.x_original,
            self.y_original,
            self.dy_original,
            self.dx_original,
        ]
        if hasattr(self, "x"):
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

        if hasattr(self, "x"):
            self.x = out[4]
            self.y = out[5]
            self.dy = out[6]
            self.dx = out[7]
        else:
            self.x = self.x_original
            self.y = self.y_original
            self.dy = self.dy_original
            self.dx = self.dx_original

    def set_x_bounds(self) -> None:
        self.x_bounds = (
            max(self.x[0], self.info.loading.x_bounds[0]),
            min(self.x[-1], self.info.loading.x_bounds[1])
        )