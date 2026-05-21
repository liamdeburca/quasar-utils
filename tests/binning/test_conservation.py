import numpy as np
from quasar_utils import binning

RNG = np.random.default_rng(seed=42)
V_RES: float = 2.3e-4

x0 = 1000.0
N_PIX = 9000

RTOL: float = 1e-9

def generate_test_arrays(n_test: int = 100):
    for _ in range(n_test):
        dx = np.ones(N_PIX)
        x = x0 + np.cumsum(dx)

        z = RNG.uniform(1, 3)
        x /= z
        dx /= z

        x_edges, xr_edges, xr = binning.log_edges(x, V_RES, dx=dx)
        dxr = xr * V_RES

        dy = RNG.uniform(0, 1, size=N_PIX)
        y = RNG.normal(loc=1000, scale=dy, size=N_PIX)

        assert np.isfinite(xr).all()

        yield dict(
            x=x,
            x_edges=x_edges,
            dx=dx,
            xr=xr,
            xr_edges=xr_edges,
            dxr=dxr,
            y=y,
            dy=dy,
        )

def test_conserves_flux_density():
    for data in generate_test_arrays():
        alpha_matrix = binning.alpha_matrix_sparse(
            data['x_edges'], data['xr_edges'], 
            dx=data['dx'],
            dxr=data['dxr'],
            conserve=False,
        )
        yr = alpha_matrix @ data['y']

        sum_init = data['y'].sum()
        sum_final = yr.sum()
        assert np.isclose(sum_init, sum_final, rtol=RTOL)

def test_conserves_total_flux():
    for data in generate_test_arrays():
        alpha_matrix = binning.alpha_matrix_sparse(
            data['x_edges'], data['xr_edges'], 
            dx=data['dx'],
            dxr=data['dxr'],
            conserve=True,
        )
        yr = alpha_matrix @ data['y']

        int_init = np.dot(data['y'], data['dx'])
        int_final = np.dot(yr, data['dxr'])
        assert np.isclose(int_init, int_final, rtol=RTOL)