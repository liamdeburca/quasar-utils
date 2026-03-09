from logging import getLogger
from typing import Optional, Any
from pydantic import validate_call

from ....utils.utils import _Info
from ..plotting.colors import pet10
from ..typing_.paths import AbsoluteFileLike

logger = getLogger(__name__)

class PlottingInfo(_Info):
    ### Should be updated for better compatibility with rcParams
    dpi: int = 300,
    ext: str = 'png'
    fontsize: int = 10
    units: int = 1
    loc_title: str = 'left'
    loc_xlabel: str = 'center'
    loc_ylabel: str = 'center'
    single_w: float = 3.27
    double_w: float = 6.93
    column_h: float = 9.13
    subplot_sep: float = 0.1
    title_h: float = 0.5
    xlabel_h: float = 0.3
    ylabel_w: float = 0.3
    left: float = 0.5
    right: float = 0.3
    top: float = 0.5
    bottom: float = 0.3
    aspect: float = 16/9
    aspect_wide: float = 9/4
    aspect_WIDE: float = 4
    cycle: dict[str, str] = pet10

    _keys: frozenset[str] = frozenset([
        'dpi', 'ext', 'fontsize', 'units', 'loc_title', 'loc_xlabel', 
        'loc_ylabel', 'single_w', 'double_w', 'column_h', 'subplot_sep', 
        'title_h', 'xlabel_h', 'ylabel_w', 'left', 'right', 'top', 'bottom', 
        'aspect', 'aspect_wide', 'aspect_WIDE', 'cycle',
    ])

    def __init__(self):
        logger.debug("Initialising 'PlottingInfo' class.")

    def update(self, info) -> None:
        from astropy.units.format import LatexInline

        logger.debug("Updating 'PlottingInfo' class:")
        f = LatexInline().to_string

        self['x_unit'] = new = f(info.units['wavelength_unit'])
        logger.debug(f">>> [1/2] 'x_unit': {new}.")
        
        self['y_unit'] = new = f(info.units.getFluxUnit())
        logger.debug(f">>> [2/2] 'y_unit': {new}.")

        self.update_from_self()

    @property
    def rc(self) -> dict[str, Any]:
        return dict()

    def __call__(self):
        from matplotlib import rc_context

        rc = {
            'figure.dpi': self['dpi'],
            'savefig.dpi': self['dpi'],
            
            'font.size': self['fontsize'],
            'axes.titlesize': self['fontsize'] - 2,
            'axes.labelsize': self['fontsize'] - 4,
            'xtick.labelsize': self['fontsize'] - 4,
            'ytick.labelsize': self['fontsize'] - 4,
            'legend.title_fontsize': self['fontsize'] - 4,
            'legend.fontsize': self['fontsize'] - 4,

            'axes.titlelocation': self['loc_title'],
            'xaxis.labellocation': self['loc_xlabel'],
            'yaxis.labellocation': self['loc_ylabel'],

            'lines.linewidth': self['lw']
        }
        rc.update(self['rc'])

        return rc_context(rc)
    
    def update_from_self(self) -> None:
        from ..logs import val_and_type as f

        logger.debug(">>> Filling in missing 'PlottingInfo' attributes.")

        missing = lambda key: key not in self.keys()

        msg = ">>> [1/4] 'column_sep': "
        if missing('column_sep'):
            self['column_sep'] = max([
                self['double_w'] - 2 * self['single_w'],
                0
            ])
            msg += f"... -> {f(self['column_sep'])}."
        else:
            msg += f"{f(self['column_sep'])}."
        logger.debug(msg)

        msg = ">>> [2/4] 'lw': "
        if missing('lw'):
            self['lw'] = self['fontsize'] / 12
            msg += f"... -> {f(self['lw'])}."
        else:
            msg += f"{f(self['lw'])}."
        logger.debug(msg)

        msg = ">>> [3/4] 'lw_thin': "
        if missing('lw_thin'):
            self['lw_thin'] = self['lw'] / 2
            msg += f"... -> {f(self['lw_thin'])}."
        else:
            msg += f"{f(self['lw_thin'])}."
        logger.debug(msg)

        msg =  ">>> [4/4] 'lw_thick': "
        if missing('lw_thick'):
            self['lw_thick'] = self['lw'] * 2        
            msg += f"... -> {f(self['lw_thick'])}."
        else:
            msg += f"{f(self['lw_thick'])}."
        logger.debug(msg)

    @staticmethod
    @validate_call
    def from_file(path: Optional[AbsoluteFileLike] = None):
        from matplotlib import rcParams
        from ..plotting import colors
        from ....utils.utils import _get_lines_from_file
        from ..logs import val_and_type as f
        
        pinfo = PlottingInfo()
        if path is None:
            return pinfo

        logger.debug(f"Configuring 'PlottingInfo' using '{path}'.")  
        lines = _get_lines_from_file.__wrapped__(logger, 'PLOTTING', path)      

        for count, line in enumerate(lines, start=1):
            key = line[0].lower()

            if key in rcParams.keys():
                pinfo['rc'][key] = line[1] \
                    if len(line[1:]) == 1 \
                    else line[1:]
                
                msg = f">>> [{count}/{len(lines)}] (rcParams) '{key}': {f(val)}."
                
            else:
                match key:
                    case 'units':
                        val = bool(line[1])
                    case 'ext' | 'loc_title':
                        val = line[1]
                    case 'dpi' | 'fontsize':
                        val = int(line[1])
                    case 'cycle':
                        val = getattr(colors, line[1], colors.pet10)
                    case _:
                        val = float(line[1])

                pinfo[key] = val
                msg = f">>> [{count}/{len(lines)}] '{key}': {f(val)}."

            logger.debug(msg)

        pinfo.update_from_self()

        return pinfo