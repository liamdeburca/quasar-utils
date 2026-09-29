import numpy as np
import pytest
from quasar_typing.numpy import FloatVector


@pytest.fixture(scope="session")
def N() -> int:
    return 20_000

@pytest.fixture(scope="session")
def dx(N: int) -> FloatVector:
    return np.full(N, 0.25, dtype=np.float64)

@pytest.fixture(scope="session")
def x(dx: FloatVector) -> FloatVector:
    return 3700 + dx.cumsum()

@pytest.fixture(scope="session")
def R(x: FloatVector) -> FloatVector:
    # 4MOST: dl ~ 1 angstrom
    return x

@pytest.fixture(scope="session")
def w() -> int:
    return 7

@pytest.fixture(scope="session")
def n_sigma() -> float:
    return 3.0

@pytest.fixture(scope="function")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)