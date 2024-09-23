import numpy as np

from .fitting import intercept_from_vars
from .indicator import intercept_interpolator

__all__ = ["DefaultScreeningData"]

class DefaultScreeningData:
    """Stores default data for screening indicator functions."""

    arrays = {
        "T": np.logspace(7, 9.35, 50, dtype=np.float32),
        "D": np.logspace(-4, 8, 50, dtype=np.float32),
        "abar": np.linspace(2, 80, 15, dtype=np.float32),
        "zbar": np.linspace(2, 30, 15, dtype=np.float32),
        "z2bar": np.linspace(4, 720, 15, dtype=np.float32),
        "z1": np.linspace(1, 30, 15, dtype=np.float32),
        "z2": np.linspace(1, 30, 15, dtype=np.float32)
    }

    _grids = np.meshgrid(*arrays.values(), sparse=True)

    grids = {}
    for j, k in enumerate(arrays):
        grids[k] = _grids[j]

    inputs = [*arrays.values()]

    try:
        intercepts = np.load("intercepts.npy")
    except FileNotFoundError:
        intercepts = intercept_from_vars(**grids, lower=1.005)
        np.save("intercepts", intercepts)

    default_interpolator = intercept_interpolator(inputs, intercepts, method="cubic")
