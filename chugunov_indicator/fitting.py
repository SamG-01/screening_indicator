import numpy as np
from scipy.optimize import curve_fit

from .chugunov_2009 import chugunov_2009

__all__ = ["border_func", "parameters_from_vars"]

def border_func(T_border: np.ndarray, c: float, k: float) -> np.ndarray:
    """
    Finds the y-values of the D-T border curve given its x-values.

    Keyword arguments:
        `T_border`: temperatures (x-values) along the border
        `c`: negative of y-intercept on log-log graph
        `k`: exponential growth rate

    Returns:
        `D_border`: densities (y-values) along the border
    """

    return 1/10**c * T_border**k

def parameters_from_border(
        T_border: np.ndarray, D_border: np.ndarray
    ) -> np.ndarray[float, float]:
    """
    Finds the parameters (c, k) of the D-T border curve given its x and y data.

    Keyword arguments:
        `T_border`: temperatures (x-values) along the border
        `D_border`: densities (y-values) along the border

    Returns: `np.ndarray` of the form `(c, k)`
        `c`: `y`-intercept of the border curve in log-log space
        `k`: slope of the border curve in log-log space
    """

    # pylint: disable=unbalanced-tuple-unpacking
    popt, pcov = curve_fit(border_func, T_border, D_border, p0 = (23, 3))
    return popt

def border_from_grid(
        T: np.ndarray, D: np.ndarray,
        F: np.ndarray, percent: float = 0.99875
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the x and y data of the D-T border curve given screening factors on a grid.
    """

    border = (1.01 * percent < F) & (F < 1.01)
    return T[border], D[border]

@np.vectorize(excluded=["T","D"], signature="(),(),(),(),(),(),()->(2)")
def parameters_from_vars(
        abar: float, zbar: float, z2bar: float, z1: float, z2: float,
        T: np.ndarray, D: np.ndarray,
        a1: float = 4, a2: float = 12
    ) -> np.ndarray[float, float]:
    """
    Finds the parameters (c, k) of the D-T border curve given physical variables.

    Keyword arguments:
        `abar`: the average mass number of the composition
        `zbar`: the average atomic number of the composition
        `z2bar`: the average squared atomic number of the composition
        `z1`, `z2`: the atomic numbers of the screening pair nuclei
        `T`, `D`: the temperature and density `meshgrid` the curve is on
        `a1`, `a2`: the mass numbers of the screening pair nuclei (shouldn't matter)
    
    Returns: `np.ndarray` of the form `(c, k)`
        `c`: `y`-intercept of the border curve in log-log space
        `k`: slope of the border curve in log-log space
    """

    F = chugunov_2009(T, D, abar, zbar, z2bar, z1, a1, z2, a2)
    T_border, D_border = border_from_grid(T, D, F)
    return parameters_from_border(T_border, D_border)

def _parameters_from_vars_array(
        X: np.ndarray, T: np.ndarray, D: np.ndarray,
        a1: float = 4, a2: float = 12
    ) -> np.ndarray[float, float]:
    """
    Finds `parameters_from_vars` using an array of inputs.
    """

    abar, zbar, z2bar, z1, z2 = X
    return parameters_from_vars(abar, zbar, z2bar, z1, z2, T, D, a1, a2)

def parameters_fit(T: np.ndarray, D: np.ndarray):
    abar_ = np.linspace(1, 60, 120)
    zbar_ = z1_ = z2_ = np.linspace(1, 30, 60)
    z2bar_ = np.linspace(1, 700, 1400)

    X = np.meshgrid(abar_, zbar_, z2bar_, z1_, z2_)
    for j, x in enumerate(X):
        X[j] = np.ravel(x)

    Y = _parameters_from_vars_array(X, T, D)
