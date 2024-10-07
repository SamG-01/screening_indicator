import numpy as np
from numba import njit

@njit()
def skip_screening(plasma, scn_fac) -> bool:
    """
    Predicts whether screening can be skipped for a given rate calculation.

    Keyword arguments:
        `state`: a `PlasmaState` object
        `scn_fac`: a `ScreenFactors` object
    """

    abar, z2bar = plasma.abar, plasma.z2bar
    z1, z2 = scn_fac.z1, scn_fac.z2
    log_z2bar = np.log10(z2bar)

    c = 20.1
    c += log_z2bar + (1/abar**0.44 - 0.31 * abar**0.34)
    c += 7.65*((z1**(0.0262) + z2**(0.0262))**2 - 4)

    T, D = plasma.temp, plasma.dens
    return np.log10(D) < 3 * np.log10(T) - c
