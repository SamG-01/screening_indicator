import numpy as np

from .fitting import intercept_from_vars
from .indicator import intercept_interpolator

__all__ = ["DefaultScreeningData"]

class DefaultScreeningData:
    """Stores default data for screening indicator functions."""

    arrays = {
        "abar": np.linspace(4, 52, 15, dtype=np.float32),
        "zbar": np.linspace(2, 26, 15, dtype=np.float32),
        "z2bar": np.linspace(4, 700, 15, dtype=np.float32),
        "z1": np.linspace(1, 10, 15, dtype=np.float32),
        "z2": np.linspace(1, 10, 15, dtype=np.float32),
        "T": np.logspace(7, 9.35, 35, dtype=np.float32),
        "D": np.logspace(-4, 8, 35, dtype=np.float32)
    }

    _grids = np.meshgrid(*arrays.values(), sparse=True)

    grids = {}
    for j, k in enumerate(arrays):
        grids[k] = _grids[j]

    inputs = [*arrays.values()][:-2]

    try:
        intercepts = np.load("./intercepts.npy")
    except FileNotFoundError:
        intercepts = intercept_from_vars(**grids)
        np.save("intercepts", intercepts)

    default_interpolator = intercept_interpolator(inputs, intercepts, method="cubic")
