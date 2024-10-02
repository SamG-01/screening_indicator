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
        "abar": np.linspace(1, 75, 20),
        "log_z2bar": np.linspace(0, 3, 20),
        "z1": np.linspace(1, 25, 20),
        "z2": np.linspace(1, 25, 20),
        "log_T": np.linspace(7, 9.35, 35),
        "log_D": np.linspace(-4, 8, 35)
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

    default_interpolator = intercept_interpolator(inputs, intercepts, method="slinear")
