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

# pylint: disable=dangerous-default-value
def skip_screening(state, scn_fac, interpolator: RegularGridInterpolator) -> bool:
    """
    Predicts whether screening can be skipped for a given rate calculation.

    Keyword arguments:
        `state`: a `PlasmaState` object
        `scn_fac`: a `ScreenFactors` object
        `interpolator`: an interpolator from `intercept_interpolator`
    """

    abar, z2bar = state.abar, state.z2bar
    z1, z2 = scn_fac.z1, scn_fac.z2

    xi = [abar, np.log10(z2bar), z1, z2]

    c = interpolator(xi)

    T, D = state.temp, state.dens
    return D < 1/10**c * T**3
