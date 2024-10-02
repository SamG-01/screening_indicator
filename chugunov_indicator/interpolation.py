import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = ["intercept_interpolator", "skip_screening"]

def intercept_interpolator(
        inputs: tuple, intercepts, **kwargs
    ) -> RegularGridInterpolator:
    """
    Creates an Interpolator for finding the y-intercept of the D-T border curve.
    
    Keyword arguments:
        `inputs`: a 4D grid of points (abar, log_z2bar, z1, z2)
        `intercepts`: the negative y-intercepts computed from the `inputs`
        `kwargs`: additional keyword arguments to pass into `interpolator`
    """

    return RegularGridInterpolator(inputs, intercepts, **kwargs)

def skip_screening(
        plasma, scn_fac,
        interpolator: RegularGridInterpolator,
        **kwargs
    ) -> bool:
    """
    Predicts whether screening can be skipped for a given rate calculation.

    Keyword arguments:
        `state`: a `PlasmaState` object
        `scn_fac`: a `ScreenFactors` object
        `interpolator`: an interpolator from `intercept_interpolator`
        `kwargs`: additional keyword arguments to pass into `interpolator.__call__`
    """

    abar, z2bar = plasma.abar, plasma.z2bar
    z1, z2 = scn_fac.z1, scn_fac.z2

    xi = [abar, np.log10(z2bar), z1, z2]

    c = interpolator(xi, **kwargs)

    T, D = plasma.temp, plasma.dens
    return D < 1/10**c * T**3
