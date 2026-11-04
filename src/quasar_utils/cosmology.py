from astropy.cosmology import realizations
from pydantic import validate_call
from quasar_typing.astropy.cosmology import CosmologyName, CosmologyType


@validate_call
def get_cosmology(name: CosmologyName) -> CosmologyType:
    return getattr(realizations, name)


@validate_call
def get_luminosity_distance(cosmology: CosmologyType, z: float) -> float:
    """Calculate the luminosity distance in Mpc for a given redshift z."""
    return cosmology.luminosity_distance(z).to("Mpc").value
        