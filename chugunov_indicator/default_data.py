import os

import numpy as np

from .fitting import intercept_from_vars
from .interpolation import intercept_interpolator

__all__ = ["DefaultScreeningData"]

intercept_dir = os.path.dirname(os.path.realpath(__file__))
intercept_file = os.path.join(intercept_dir, 'intercepts.npy')

class DefaultScreeningData:
    """Stores default data for screening indicator functions."""

    arrays = {
        "abar": np.linspace(1, 75, 25, dtype=np.float32),
        "log_z2bar": np.linspace(0.6, 3, 25, dtype=np.float32),
        "z1": np.linspace(1, 20, 25, dtype=np.float32),
        "z2": np.linspace(1, 20, 25, dtype=np.float32),
        "T": np.logspace(7, 9.35, 35, dtype=np.float32),
        "D": np.logspace(-4, 8, 35, dtype=np.float32)
    }

    _grids = np.meshgrid(*arrays.values(), sparse=True, indexing="ij")

    grids = {}
    for j, k in enumerate(arrays):
        grids[k] = _grids[j]

    inputs = [*arrays.values()][:-2]

    try:
        intercepts = np.load(intercept_file)
    except FileNotFoundError:
        intercepts = intercept_from_vars(**grids)
        np.save(intercept_file, intercepts)

    default_interpolator = intercept_interpolator(inputs, intercepts, method="cubic")
