import numpy as np

from .chugunov_2009 import chugunov_2009

__all__ = ["intercept_from_vars"]

def intercept_from_vars(
        log_T: np.ndarray, log_D: np.ndarray,
        abar: float, log_z2bar: float,
        z1: float, z2: float,
        zbar: float = 4, a1: int = 4, a2: int = 12,
        lower: float = 1.005, upper: float = 1.01
    ) -> float:
    """
    Finds the negative y-intercept `c` of the D-T border curve given physical variables.

    Keyword arguments:
        `T`, `D`: the temperature and density grid the curve is on
        `abar`: the average mass number of the composition
        `zbar`: the average atomic number of the composition
        `z2bar`: the average squared atomic number of the composition
        `z1`, `z2`: the atomic numbers of the screening pair nuclei
        `a1`, `a2`: the mass numbers of the screening pair nuclei (shouldn't matter)
        `lower`, `upper`: determines the upper and lower bounds of the border line

    Returns:
        `c`: negative `y`-intercept of the border curve in log-log space
    """

    F = chugunov_2009(10**log_T, 10**log_D, abar, zbar, 10**log_z2bar, z1, z2, a1, a2)

    border = (lower <= F) & (F <= upper)
    log_T_border = np.where(border, log_T, np.nan)
    log_D_border = np.where(border, log_D, np.nan)

    return np.nanmean(3*log_T_border - log_D_border, axis=(-1, -2))
