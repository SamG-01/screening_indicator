import os

import numpy as np

from .fitting import intercept_from_vars
from .interpolation import intercept_interpolator

__all__ = ["DefaultScreeningData"]

intercept_dir = os.path.dirname(os.path.realpath(__file__))
intercept_file = os.path.join(intercept_dir, 'intercepts.npy')

class DefaultScreeningData:
    """Stores default data for screening indicator functions."""

    lo = np.array([1, 0, 1, 1])
    hi = np.array([75, 3, 25, 25])

    linspaces = {
        "abar": np.linspace(lo[0], hi[0], 20),
        "log_z2bar": np.linspace(lo[1], hi[1], 20),
        "z1": np.linspace(lo[2], hi[2], 20),
        "z2": np.linspace(lo[3], hi[3], 20),
        "log_T": np.linspace(7, 9.35, 35),
        "log_D": np.linspace(-4, 8, 35)
    }

    _points = np.meshgrid(*linspaces.values(), sparse=True, indexing="ij")

    points = {}
    for j, k in enumerate(linspaces):
        points[k] = _points[j]

    grid = np.array([*linspaces.values()][:-2])

    try:
        intercepts = np.load(intercept_file)
    except FileNotFoundError:
        intercepts = intercept_from_vars(**points)
        np.save(intercept_file, intercepts)

    default_interpolator = intercept_interpolator(
        intercepts, lo, hi,
        verbose=0, prefilter=1/3
    )
