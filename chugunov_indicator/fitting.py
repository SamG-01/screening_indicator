import numpy as np
from scipy.optimize import curve_fit

from .chugunov_2009 import chugunov_2009

__all__ = ["border_func", "parameters_from_vars"]

def border_func(T_border: np.ndarray, c: float) -> np.ndarray:
    """
    Finds the y-values of the D-T border curve given its x-values.

    Keyword arguments:
        `T_border`: temperatures (x-values) along the border
        `c`: negative of y-intercept on log-log graph

    Returns:
        `D_border`: densities (y-values) along the border
    """

    return 1/10**c * T_border**3

def _parameters_from_border(
        T_border: np.ndarray, D_border: np.ndarray, return_pcov: bool = False
    ) -> float:
    """
    Finds the negative y-intercept `c` of the D-T border curve given its x and y data.

    Keyword arguments:
        `T_border`: temperatures (x-values) along the border
        `D_border`: densities (y-values) along the border
        `return_pcov`: whether to return `pcov`

    Returns:
        `popt`: negative`y`-intercept of the border curve in log-log space
        `pcov`: variance of `popt`
    """

    # pylint: disable=unbalanced-tuple-unpacking
    (popt,), ((pcov,),) = curve_fit(
        border_func, T_border, D_border,
        p0 = (23,), bounds=(18, 27.5)
    )
    if return_pcov:
        return popt, pcov
    return popt

@np.vectorize(signature="(n,n),(n,n)->()")
def parameters_from_border(
        T_border: np.ndarray, D_border: np.ndarray
    ) -> float:
    """
    Finds the negative y-intercept `c` of the D-T border curve given its x and y data. Vectorized version of _paramters_from_border.

    Keyword arguments:
        `T_border`: temperatures (x-values) along the border
        `D_border`: densities (y-values) along the border

    Returns:
        `c`: negative`y`-intercept of the border curve in log-log space
    """

    T_border, D_border = np.ravel(T_border), np.ravel(D_border)
    T_border, D_border = T_border[T_border != 0], D_border[T_border != 0]

    return _parameters_from_border(T_border, D_border, False)

def border_from_grid(
        T: np.ndarray, D: np.ndarray,
        F: np.ndarray, percent: float = 0.995
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the x and y data of the D-T border curve given screening factors on a grid.
    """

    border = (1.01 * percent < F) & (F < 1.01)
    return np.where(border, T, 0), np.where(border, D, 0)

def parameters_from_vars(
        abar: float, zbar: float, z2bar: float, z1: float, z2: float,
        T: np.ndarray, D: np.ndarray,
        a1: float = 4, a2: float = 12, percent: float = 0.99875
    ) -> float:
    """
    Finds the negative y-intercept `c` of the D-T border curve given physical variables.

    Keyword arguments:
        `abar`: the average mass number of the composition
        `zbar`: the average atomic number of the composition
        `z2bar`: the average squared atomic number of the composition
        `z1`, `z2`: the atomic numbers of the screening pair nuclei
        `T`, `D`: the temperature and density `meshgrid` the curve is on
        `a1`, `a2`: the mass numbers of the screening pair nuclei (shouldn't matter)
        `percent`: determines the thickness of the border line

    Returns:
        `c`: negative `y`-intercept of the border curve in log-log space
    """

    F = chugunov_2009(T, D, abar, zbar, z2bar, z1, a1, z2, a2)
    T_border, D_border = border_from_grid(T, D, F, percent)
    return parameters_from_border(T_border, D_border)
