from typing import Any

from astropy.units import Unit
from h5py import File, Group, string_dtype
from pandas import DataFrame

from quasar_utils.linelist import LineList
from quasar_utils.setup import Info

STR_COLUMNS: frozenset[str] = frozenset({
    "name",
    "linetype",
    "complex",
    "needs_line",
    "is_copy_of",
})


def linelist_to_hdf5(
    linelist: LineList,
    f: File | Group,
    info: Info,
    group_name: str = "linelist",
    force: bool = False,
    compression: str | None = "gzip",
    compression_opts: int | None = None,
) -> Group:
    """Write a LineList into an HDF5 file.

    The layout is:
      /<group_name>/columns/<colname>  -- dataset per column
      /<group_name>/units/<semantic>   -- unit strings for dimensions

    The function requires an Info instance to obtain canonical unit strings
    for writing dimensional columns. If info is None, unit attributes are
    omitted.
    """
    if group_name in f:
        if not force:
            raise ValueError(f"Group '{group_name}' already exists in the HDF5 file.")
        del f[group_name]

    str_dt = string_dtype(encoding="utf-8")
    
    grp = f.create_group(group_name)
    grp.attrs["format_version"] = "1"

    cols = grp.create_group("columns")
    
    units_grp = grp.create_group("units")
    units_grp.create_dataset(
        "wavelength", 
        data=str(info.units.wavelength_unit), 
        dtype=str_dt,
    )
    units_grp.create_dataset(
        "velocity", 
        data=str(info.units.velocity_unit), 
        dtype=str_dt,
    )
    units_grp.create_dataset(
        "strength", 
        data=str(info.units.strength_unit), 
        dtype=str_dt,
    )
    for col in linelist.columns:
        data = linelist[col].to_numpy()
        dtype = str_dt if col in STR_COLUMNS else None
        # Choose dtype / write as dataset
        cols.create_dataset(
            col,
            data=data,
            dtype=dtype,
            compression=compression,
            compression_opts=compression_opts,
        )

    return grp


def hdf5_to_linelist(
    f: File | Group, 
    info: Info, 
    group_name: str = "linelist",
) -> LineList:
    """Read a LineList from HDF5 and convert dimensional columns to the
    units given by the provided Info instance.

    The function requires an Info instance to convert stored quantities
    (which are saved with unit metadata) into the caller's preferred units
    using info.units.get*(...) helpers.
    """
    if group_name not in f:
        raise ValueError(f"Group '{group_name}' does not exist in the HDF5 file.")

    grp = f[group_name]
    cols = grp["columns"]

    # build units map from units group and per-dataset attrs
    units_map: dict[str, str | None] = {}
    if "units" in grp:
        for k, v in grp["units"].items():
            units_map[k] = v.asstr()[()]

    data: dict[str, Any] = {}
    for name, dset in cols.items():
        if name in STR_COLUMNS:

            dset = dset.asstr()

        arr = dset[()]

        unit = None
        if name == "wave":
            unit = units_map.get("wavelength")
        elif name.startswith(("v_", "fwhm")):
            unit = units_map.get("velocity")
        elif name.startswith("strength"):
            unit = units_map.get("strength")

        if unit is not None:
            q = arr * Unit(unit)
            if name == "wave":
                arr = info.units.getUnitlessWavelength(q)
            elif name.startswith(("v_", "fwhm")):
                arr = info.units.getUnitlessKMS(q)
            elif name.startswith("strength"):
                arr = info.units.getUnitlessStrength(q)
            else:
                raise ValueError(f"Unknown unit conversion for column '{name}'")

        data[name] = arr

    return DataFrame(data)
