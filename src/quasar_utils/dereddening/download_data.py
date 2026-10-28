"""
Utilities for downloading galactic dust maps.
"""
from pathlib import Path
from typing import TYPE_CHECKING

_this_file: Path = Path(__file__).resolve()

PATH_TO_CACHE: Path = _this_file.parent / "__cache__"
PATH_TO_SFD_CACHE: Path = PATH_TO_CACHE / "sfd"
PATH_TO_CSFD_CACHE: Path = PATH_TO_CACHE / "csfd"

if not TYPE_CHECKING:
    PATH_TO_CACHE.mkdir(exist_ok=True)
    PATH_TO_SFD_CACHE.mkdir(exist_ok=True)
    PATH_TO_CSFD_CACHE.mkdir(exist_ok=True)

# From: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EWCNL5&version=1.0
SFD_NGP_URL: str = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/EWCNL5/JOV0CM"
SFD_NGP_PATH: Path = PATH_TO_SFD_CACHE / "SFD_dust_4096_ngp.fits"

SFD_SGP_URL: str = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/EWCNL5/XTLXX5"
SFD_SGP_PATH: Path = PATH_TO_SFD_CACHE / "SFD_dust_4096_sgp.fits"

# From: https://zenodo.org/records/8207175
CSFD_URL: str = "https://zenodo.org/records/8207175/files/csfd_ebv.fits?download=1"
CSFD_PATH: Path = PATH_TO_CSFD_CACHE / "csfd_ebv.fits"

CSFD_MASK_URL: str = "https://zenodo.org/records/8207175/files/mask.fits?download=1"
CSFD_MASK_PATH: Path = PATH_TO_CSFD_CACHE / "mask.fits"


def download_and_dump(
    url: str,
    output_path: Path,
    dry_run: bool = False
) -> None:
    from astropy.io import fits
    data = fits.open(url)
    if not dry_run:
        data.writeto(output_path, overwrite=True)


def main(dry_run: bool = False, force: bool = False) -> None:
    if not SFD_NGP_PATH.exists() or force:
        download_and_dump(SFD_NGP_URL, SFD_NGP_PATH, dry_run=dry_run)
    if not SFD_SGP_PATH.exists() or force:
        download_and_dump(SFD_SGP_URL, SFD_SGP_PATH, dry_run=dry_run)

    if not CSFD_PATH.exists() or force:
        download_and_dump(CSFD_URL, CSFD_PATH, dry_run=dry_run)
    if not CSFD_MASK_PATH.exists() or force:
        download_and_dump(CSFD_MASK_URL, CSFD_MASK_PATH, dry_run=dry_run)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run, force=args.force)
