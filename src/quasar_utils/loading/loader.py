__all__ = ["_Loader"]

from dataclasses import field
from logging import getLogger
from typing import Literal

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from numpy import ascontiguousarray, diff, float64, full, log
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import CompositeUnit_, Quantity_, Unit_
from quasar_typing.numpy import CoordsTuple, FloatVector
from quasar_typing.pathlib import AbsoluteFilePath

from ..binning import log_resample
from ..decorators import validate_call, validated_apply_info_to_method
from ..dereddening import deredden_spectrum
from ..setup import Info
from .utils import LoaderOutput

logger = getLogger(__name__)


@dataclass
class _Loader:
    """
    Loader helper that encapsulates a single spectrum's loading and
    coordinate-transformation pipeline.

    This dataclass stores the raw inputs (path, title, spectral
    coordinate arrays and errors, and optional astrophysical metadata)
    and provides a sequence of transformation utilities. The
    instance's ``__call__`` method runs the standard loading pipeline:

    1. :py:meth:`transform_coords` -- convert astropy.Quantity inputs to
       unitless numpy arrays using the project's units stored in
       ``Info``.
    2. :py:meth:`deredden_coords` (optional) -- apply Galactic
       extinction correction (if requested and RA/DEC are set).
    3. :py:meth:`logbin_coords` (optional) -- re-sample to logarithmic
       wavelength scale.
    4. :py:meth:`redshift_correct_coords` (optional) -- correct
       coordinates and fluxes to rest-frame given ``z``.
    5. :py:meth:`make_coords_contiguous` -- ensure arrays are
       C-contiguous.

    Parameters
    ----------
    path : str | AbsoluteFilePath
        Path to the input file or source identifier.
    info : Info, optional
        Project configuration and unit conversion helpers used during
        loading. Constructed by default.
    z : float | None, optional
        Redshift to apply when performing redshift correction. If ``0`` redshift 
        correction is skipped.
    title : str, optional
        Human-readable title for this spectrum.
    ra : float | None, optional
        Right ascension in degrees (required for dereddening).
    dec : float | None, optional
        Declination in degrees (required for dereddening).
    """
    path: str | AbsoluteFilePath
    info: Info = field(default_factory=Info)
    z: float | None = None

    title: str = "missing_title"
    ra: float | None = None
    dec: float | None = None

    x: FloatVector | Quantity_ = field(init=False)
    y: FloatVector | Quantity_ = field(init=False)
    dy: FloatVector | Quantity_ = field(init=False)
    dx: float | FloatVector | Quantity_ = field(init=False)

    def __post_init__(self):
        """
        Post-initialization hook.

        If ``dx`` is a single float scalar, expand it to a numpy array
        of length ``len(self.x)`` filled with that scalar value.
        """

        if isinstance(self.dx, float):
            self.dx = full(len(self.x), self.dx, dtype=float64)

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
        sigma_res: float | None = None,
        rebin: bool | None = None,
        conserve: bool | None = None,
        covariance: bool | None = None,
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
            A mapping containing: path, title, x, y, dy, dx, and info.

        Raises
        ------
        ValidationError
            Pydantic type validation
        """

        msg = f"Running {self.__class__.__name__} loading pipeline: "

        msg += "(1) creating unitless coordinates, "
        coords, dx = self.transform_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            dx=self.dx,
            info=self.info,
        )

        if deredden[0]:
            if self.ra is None:
                msg += "(2) skipping dereddening due to missing coordinates RA, "
            elif self.dec is None:
                msg += "(2) skipping dereddening due to missing coordinates DEC, "
            else:                
                msg += f"(2) applying dereddening correction [RA={self.ra}, DEC={self.dec}, map={deredden[1]}, law={deredden[2]}], Rv={deredden[3]}], "
                coords = self.deredden_coords.__wrapped__(
                    self.__class__,
                    coords,
                    self.ra,
                    self.dec,
                    deredden[1],
                    deredden[2],
                    self.info.units["wavelength_unit"],
                    deredden[3],
                )
        else:
            msg += "(2) skipping dereddening correction, "

        if rebin:
            msg += "(3) logarithmic re-binning, "
            coords, dx = self.logbin_coords.__wrapped__(
                self.__class__,
                coords,
                dx=dx,
                sigma_res=sigma_res,
                conserve=conserve,
                covariance=covariance,
            )
        else:
            _sigma_res = diff(log(coords[0]))

            msg += f"(3) skipping logarithmic re-binning [assuming 'sigma_res'={sigma_res=:.2e}, "
            msg += f"actual 'sigma_res'={_sigma_res.mean():.2e}±{_sigma_res.std():.2e}]"

        if self.z != 0.0:
            msg += "(4) applying redshift correction, "
            coords, dx = self.redshift_correct_coords.__wrapped__(
                self.__class__,
                coords,
                dx=dx,
                z=self.z,
            )
        elif self.z is None:
            msg += "(4) skipping redshift correction due to missing redshift, "
        else:
            msg += "(4) spectrum is already in the restframe (z=0.0), "

        msg += "(5) ensuring coordinates are C-contiguous."
        x, y, dy, dx = self.make_coords_contiguous.__wrapped__(
            self.__class__,
            *coords,
            dx,
        )

        logger.debug(msg)

        return {
            "path": self.path,
            "title": self.title,
            "x": x,
            "y": y,
            "dy": dy,
            "dx": dx,
            "info": self.info,
        }

    @property
    def transformed_coords(self) -> tuple[CoordsTuple, FloatVector]:
        """
        Return the input coordinates after unit conversion.

        This property calls :py:meth:`transform_coords` to convert any
        ``astropy.Quantity`` inputs to the project's unitless numpy
        arrays according to ``Info``.

        Returns
        -------
        coords : CoordsTuple
            A tuple ``(x, y, dy)`` of unitless numpy arrays.
        dx : FloatVector
            A numpy array of wavelength bin sizes.
        """
        return self.transform_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            dx=self.dx,
            info=self.info,
        )

    @classmethod
    @validate_call
    def transform_coords(
        cls,
        coords: CoordsTuple,
        *,
        dx: FloatVector | Quantity_,
        info: Info,
    ) -> tuple[CoordsTuple, FloatVector]:
        """
        Convert astropy.Quantity inputs to unitless numpy arrays.

        This validated method accepts ``(x, y, dy)`` possibly expressed
        as ``astropy.Quantity`` objects. It uses the ``Info.units``
        helpers to convert:

        - x quantities -> wavelength unitless arrays via
          ``info.units.getWavelength``
        - y and dy quantities -> flux unitless arrays via
          ``info.units.getFlux``
        - dx quantities -> wavelength spacing via
          ``info.units.getWavelength``

        Parameters
        ----------
        coords : CoordsTuple
            Tuple ``(x, y, dy)``. Each element may be a numpy array or
            an ``astropy.Quantity``.
        dx : FloatVector | Quantity_
            Per-point wavelength spacing or an ``astropy.Quantity``
            spacing.
        info : Info
            Info instance providing unit conversion helpers.

        Returns
        -------
        coords : CoordsTuple
            A tuple ``(x, y, dy)`` of unitless numpy arrays.
        dx : FloatVector
            A numpy array of wavelength bin sizes.

        Raises
        ------
        ValidationError
            Pydantic type validation
        """
        x, y, dy = coords
        if isinstance(x, Quantity):
            x = info.units.getWavelength(x)
        if isinstance(y, Quantity):
            y = info.units.getFlux(y)
        if isinstance(dy, Quantity):
            dy = info.units.getFlux(dy)
        if isinstance(dx, Quantity):
            dx = info.units.getWavelength(dx)

        return (x, y, dy), dx

    @property
    def dereddenned_coords(self) -> CoordsTuple:
        """
        Return coordinates after applying Galactic dereddening.

        RA and DEC must be set on the instance prior to calling this
        property.

        Returns
        -------
        CoordsTuple
            The dereddened ``(x, y, dy)`` coordinate tuple.

        Raises
        ------
        AssertionError
            If RA or DEC are not set on the instance.
        """

        assert self.ra is not None, "RA is required for dereddening"
        assert self.dec is not None, "DEC is required for dereddening"

        return self.deredden_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            self.ra,
            self.dec,
            self.info.loading["dereddening_map"],
            self.info.loading["dereddening_law"],
            self.info.units.getWavelengthUnit(),
            self.info.loading["Rv"],
        )

    @classmethod
    @validate_call
    def deredden_coords(
        cls,
        coords: CoordsTuple,
        ra: float,
        dec: float,
        map_name: Literal["sfd", "csfd"],
        law_name: Literal["ccm89", "o94"],
        wavelength_unit: Unit_ | CompositeUnit_,
        Rv: float,
    ) -> CoordsTuple:
        """
        Apply Galactic dereddening to the input spectrum coordinates.

        This validated method wraps :pyfunc:`deredden_spectrum`. It
        requires target sky coordinates (RA, DEC), a dust map name, an
        extinction law name, the wavelength unit to interpret ``x``,
        and the ``Rv`` parameter to scale the extinction.

        Parameters
        ----------
        coords : CoordsTuple
            Input coordinate tuple ``(x, y, dy)``, in numeric arrays.
            Wavelengths should be interpretable using
            ``wavelength_unit``.
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        map_name : Literal['sfd', 'csfd']
            Dust map to use for E(B-V) estimation.
        law_name : Literal['ccm89', 'o94']
            Extinction law to use for wavelength-dependent extinction.
        wavelength_unit : Unit_ | CompositeUnit_
            Unit or composite unit representing the wavelength unit of
            ``x``.
        Rv : float
            Rv parameter for the extinction law.

        Returns
        -------
        CoordsTuple
            The dereddened ``(x, y, dy)`` coordinate tuple.

        Raises
        ------
        ValidationError
            Pydantic type validation
        """
        return deredden_spectrum(
            coords,
            SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs"),
            map_name=map_name,
            law_name=law_name,
            wavelength_unit=wavelength_unit,
            Rv=Rv,
        )

    @property
    def redshift_corrected_coords(self) -> tuple[CoordsTuple, FloatVector]:
        """
        Return coordinates and spacing after applying redshift
        correction.

        If ``self.z`` is ``None`` or ``0`` the original coordinates and
        ``dx`` are returned unchanged.

        Returns
        -------
        coords : CoordsTuple
            A tuple ``(x, y, dy)`` of unitless numpy arrays corrected to 
            rest-frame if ``z`` != 0.
        dx : FloatVector
            A numpy array of wavelength bin sizes corrected to rest-frame if 
            ``z`` != 0.
        """
        return self.redshift_correct_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            dx=self.dx,
            z=self.z,
        )

    @classmethod
    @validate_call
    def redshift_correct_coords(
        cls,
        coords: CoordsTuple,
        *,
        dx: FloatVector,
        z: float,
    ) -> tuple[CoordsTuple, FloatVector]:
        """
        Correct coordinates and fluxes for cosmological redshift.

        The method shifts wavelength coordinates to the rest frame and
        scales flux and flux uncertainties by ``(1 + z)**2``. The
        ``dx`` wavelength spacing is adjusted by dividing by
        ``(1 + z)``.

        Parameters
        ----------
        coords : CoordsTuple
            Input spectral coordinates ``(x, y, dy)``.
        dx : FloatVector
            Per-point wavelength spacing corresponding to ``x``.
        z : float
            Redshift to correct by. If ``z == 0`` the input coord
            tuple and ``dx`` are returned unchanged.

        Returns
        -------
        coords : CoordsTuple
            The redshift-corrected ``(x, y, dy)`` coordinate tuple.
        dx : FloatVector
            The redshift-corrected wavelength bin size array.

        Raises
        ------
        ValidationError
            Pydantic type validation
        """
        if z == 0:
            return coords, dx

        x, y, dy = coords
        corr = 1 + z
        corr_sq = corr * corr

        x_corr = x / corr
        dx_corr = dx / corr

        y_corr = y * corr_sq
        dy_corr = dy * corr_sq

        return (x_corr, y_corr, dy_corr), dx_corr

    @property
    def logbinned_coords(self) -> tuple[CoordsTuple, FloatVector]:
        """
        Return coordinates after logarithmic re-binning using ``Info``
        defaults.

        This property invokes :py:meth:`logbin_coords` using
        ``sigma_res``, ``conserve`` and ``covariance`` values from
        ``self.info``.

        Returns
        -------
        tuple
            ``((x_r, y_r, dy_r), dx_r)`` where ``x_r`` is log-binned
            wavelengths and ``dx_r`` is the spacing computed as
            ``x_r * sigma_res``.
        """

        return self.logbin_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            dx=self.dx,
            sigma_res=self.info.loading["sigma_res"],
            conserve=self.info.loading["conserve"],
            covariance=self.info.loading["covariance"],
        )

    @classmethod
    @validate_call
    def logbin_coords(
        cls,
        coords: CoordsTuple,
        *,
        dx: FloatVector,
        sigma_res: float,
        conserve: bool,
        covariance: bool,
    ) -> tuple[CoordsTuple, FloatVector]:
        """
        Resample coordinates to a logarithmic wavelength grid.

        This validated method delegates to the project's
        :pyfunc:`log_resample` routine. It returns the log-binned
        ``(x, y, dy)`` and the per-point spacing ``dx`` computed as
        ``x_r * sigma_res``.

        Parameters
        ----------
        coords : CoordsTuple
            Input coordinates (x, y, dy).
        dx : FloatVector
            Input wavelength spacing array corresponding to x.
        sigma_res : float
            Target logarithmic resolution to resample to.
        conserve : bool
            Whether to conserve integrated flux during re-binning.
        covariance : bool
            Whether to propagate covariance information during
            re-binning.

        Returns
        -------
        coords : CoordsTuple
            A tuple ``(x_r, y_r, dy_r)`` of log-binned unitless numpy arrays.
        dx : FloatVector
            A numpy array of wavelength bin sizes computed as
            ``dx_r = x_r * sigma_res``.

        Raises
        ------
        ValidationError
            Pydantic type validation
        """
        xr, yr, dyr = log_resample.__wrapped__(
            *coords,
            sigma_res,
            dx=dx,
            conserve=conserve,
            covariance=covariance,
        )
        dxr = xr * sigma_res
        return (xr, yr, dyr), dxr

    @classmethod
    @validate_call
    def make_coords_contiguous(
        cls,
        *coords: FloatVector,
    ) -> tuple[FloatVector, ...]:
        """
        Ensure coordinate arrays are C-contiguous numpy arrays.

        Iterate over the provided arrays and convert any
        non-C-contiguous arrays to contiguous copies.

        Parameters
        ----------
        *coords : FloatVector
            One or more numeric arrays to ensure are C-contiguous.

        Returns
        -------
        coords : tuple[FloatVector, ...]
            Tuple of arrays that are guaranteed to be C-contiguous.

        Raises
        ------
        ValidationError
            Pydantic type validation
        """
        return tuple(
            arr if arr.flags["C_CONTIGUOUS"] else ascontiguousarray(arr)
            for arr in coords
        )
