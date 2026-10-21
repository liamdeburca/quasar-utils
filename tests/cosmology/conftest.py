import pytest
from astropy.cosmology import FlatLambdaCDM, realizations


@pytest.fixture
def known_values() -> list[tuple[float, float]]:
    """
    List of known redshift-luminosity distance pairs for Planck18 cosmology.

    Reference
    ---------
    Website: http://www.astro.ucla.edu/%7Ewright/CosmoCalc.html
    H0: 67.7 km/s/Mpc
    Omega_m: 0.310
    Open/flat: flat
    Omega_vac: 0.714
    """
    return [
        (0.0, 0.0),
        (0.1, 475.6),
        (0.5, 2919.0),
        (1.0, 6791.4),
        (1.5, 11195.0),
        (2.0, 15929.1),
        (3.0, 26026.6),
        (100.0, 1296744.4),
    ]


@pytest.fixture
def cosmo() -> FlatLambdaCDM:
    return realizations.Planck18
