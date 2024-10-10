import numpy as np
from numba import njit

__all__ = ["skip_chugunov_2009"]

@njit()
def _screening_intercept(abar: float, z2bar: float, z1: float, z2: float) -> float:
    """
    Predicts the height of the D-T line that decides when screening is important.
    """

    m1 = 1.03
    a2, k2 = 10.41502026, 0.02076143
    a3, k3 = 7.73078747, 0.02600378

    C0 = -10.132085464080653
    C1 = m1*np.log10(z2bar)
    C2 = a2*(abar**(-k2) - abar**k2)
    C3 = a3*(z1**k3 + z2**k3)**2

    return C0 + C1 + C2 + C3

@njit()
def skip_chugunov_2009(plasma, scn_fac) -> bool:
    """
    Predicts whether screening for `chugunov_2009` can be skipped for a given rate calculation.

    Keyword arguments:
        `state`: a `PlasmaState` object
        `scn_fac`: a `ScreenFactors` object
    """

    abar, z2bar = plasma.abar, plasma.z2bar
    z1, z2 = scn_fac.z1, scn_fac.z2
    T, D = plasma.temp, plasma.dens

    C = _screening_intercept(abar, z2bar, z1, z2)
    return np.log10(D) < 3 * np.log10(T) - C
