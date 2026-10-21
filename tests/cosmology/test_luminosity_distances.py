import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM


def test_luminosity_distances(
    subtests: pytest.Subtests,
    cosmo: FlatLambdaCDM,
    known_values: list[tuple[float, float]],
) -> None:
    for z, d_l in known_values:
        d_l_calculated = cosmo.luminosity_distance(z).to("Mpc").value

        if d_l == 0:
            r = 0.0
        else:
            r = abs(d_l - d_l_calculated) / d_l
        with subtests.test(msg=f"z={z}: {d_l=:.1e}, {d_l_calculated=:.1e}, [{r=:.1e}]"):
            assert np.isclose(d_l_calculated, d_l, rtol=1e-3)