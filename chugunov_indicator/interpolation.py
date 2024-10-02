import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = ["intercept_interpolator", "skip_screening"]

def intercept_interpolator(
        inputs: tuple, intercepts,
        method: str = "cubic"
    ) -> RegularGridInterpolator:
    """
    Creates an Interpolator for finding the y-intercept of the D-T border curve.
    
    Keyword arguments:
        `inputs`: a 4D grid of points (abar, log_z2bar, z1, z2)
        `intercepts`: the negative y-intercepts computed from the `inputs`
    """

    return RegularGridInterpolator(inputs, intercepts, method)

def skip_screening(
        plasma, scn_fac,
        interpolator: RegularGridInterpolator,
        method: str = "cubic"
    ) -> bool:
    """
    Predicts whether screening can be skipped for a given rate calculation.

    Keyword arguments:
        `state`: a `PlasmaState` object
        `scn_fac`: a `ScreenFactors` object
        `interpolator`: an interpolator from `intercept_interpolator`
    """

    abar, z2bar = plasma.abar, plasma.z2bar
    z1, z2 = scn_fac.z1, scn_fac.z2

    xi = [abar, np.log10(z2bar), z1, z2]

    c = interpolator(xi, method)

    T, D = plasma.temp, plasma.dens
    return D < 1/10**c * T**3
