import numpy as np
from numba import njit

@njit()
def _skip_screening(T, D, abar, z2bar, z1, z2) -> bool:
    """
    Predicts whether screening can be skipped for a given rate calculation.
    """

    C0, m1 = -10.132085464080653, 1.03
    a2, k2 = 10.41502026, 0.02076143
    a3, k3 = 7.73078747, 0.02600378

    C = C0 + m1*np.log10(z2bar)
    C += a2*(abar**(-k2) - abar**k2)
    C += a3*(z1**k3 + z2**k3)**2

    return np.log10(D) < 3 * np.log10(T) - C

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
    T, D = plasma.temp, plasma.dens
    return _skip_screening(T, D, abar, z2bar, z1, z2)
