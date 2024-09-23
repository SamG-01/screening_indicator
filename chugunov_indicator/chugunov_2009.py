import numpy as np

from pynucastro.constants import constants
from pynucastro.screening.screen import f0

__all__ = ["chugunov_2009"]

def PlasmaStateComps(D: float, abar: float, zbar: float) -> float:
    """Returns `PlasmaState` values used in screening correction factor calculation in `NumPy`-friendly form."""

    # Average mass and total number density
    mbar = abar * np.float32(constants.m_u)
    ntot = D / mbar

    # Electron number density
    # zbar * ntot works out to sum(z[i] * n[i]), after cancelling terms
    n_e = zbar * ntot

    # temperature-independent part of Gamma_e, from Chugunov 2009 eq. 6
    gamma_e_fac = np.float32(constants.q_e ** 2 / constants.k * np.cbrt(4 * np.pi / 3)) * np.cbrt(n_e)

    return gamma_e_fac

def ScreenFactorsComps(
        z1: int, a1: int,
        z2: int, a2: int
    ) -> tuple[float, float]:
    """Returns `ScreenFactors` values used in screening correction factor calculation in `NumPy`-friendly form."""

    #zs13 = np.cbrt(z1 + z2)
    #zhat = (z1 + z2) ** (5/3) - z1 ** (5/3) - z2 ** (5/3)
    #zhat2 = (z1 + z2) ** (5/12) - z1 ** (5/12) - z2 ** (5/12)
    #lzav = (5/3) * np.log(z1 * z2 / (z1 + z2))
    aznut = np.cbrt(z1 ** 2 * z2 ** 2 * a1 * a2 / (a1 + a2), dtype=np.float32)
    ztilde = 0.5 * (np.cbrt(z1) + np.cbrt(z2))

    #return zs13, zhat, zhat2, lzav, aznut, ztilde
    return aznut, ztilde

def chugunov_2009(
        T: float, D: float,
        abar: float, zbar: float, z2bar: float,
        z1: int, z2: int,
        a1: int, a2: int,
    ) -> float:
    """Calculates screening factors based on :cite:t:`chugunov:2009` in `NumPy`-friendly form.

    :param PlasmaState state:     the precomputed plasma state factors
    :param ScreenFactors scn_fac: the precomputed ion pair factors
    :returns: screening correction factor
    """

    # Precomputed Values
    gamma_e_fac = PlasmaStateComps(D, abar, zbar)
    aznut, ztilde = ScreenFactorsComps(z1, a1, z2, a2)

    # z1z2 and zcomp
    z1z2 = z1 * z2
    zcomp = z1 + z2

    # Gamma_e from eq. 6
    Gamma_e = gamma_e_fac / T

    # Coulomb coupling parameters for ions and compound nucleus, eqs. 7 & 9
    Gamma_1 = Gamma_e * z1 ** (5 / 3)
    Gamma_2 = Gamma_e * z2 ** (5 / 3)
    Gamma_comp = Gamma_e * zcomp ** (5 / 3)

    Gamma_12 = Gamma_e * z1z2 / ztilde

    # Coulomb barrier penetrability, eq. 10
    tau_factor = np.cbrt(27 / 2 * (np.pi * constants.q_e ** 2 / constants.hbar) ** 2 * constants.m_u / constants.k, dtype=np.float32)
    tau_12 = tau_factor * aznut / np.cbrt(T, dtype=np.float32)

    # eq. 12
    zeta = 3 * Gamma_12 / tau_12

    # additional fit parameters, eq. 25
    y_12 = 4 * z1z2 / zcomp ** 2
    c1 = 0.013 * y_12 ** 2
    c2 = 0.406 * y_12 ** 0.14
    c3 = 0.062 * y_12 ** 0.19 + 1.8 / Gamma_12

    poly = 1 + zeta*(c1 + zeta*(c2 + c3*zeta))
    t_12 = np.cbrt(poly)

    # strong screening enhancement factor, eq. 23, replacing tau_ij with t_ij
    # Using Gamma/tau_ij gives extremely low values, while Gamma/t_ij gives
    # values similar to those from Chugunov 2007.
    term1 = f0(Gamma_1 / t_12)
    term2 = f0(Gamma_2 / t_12)
    term3 = f0(Gamma_comp / t_12)
    h_fit = np.float32(term1 + term2 - term3)

    # weak screening correction term, eq. A3
    corr_C = (
        3 * z1z2 * np.sqrt(z2bar / zbar) /
        (zcomp ** 2.5 - z1 ** 2.5 - z2 ** 2.5)
    )

    # corrected enhancement factor, eq. A4
    Gamma_12_2 = Gamma_12 ** 2
    numer = corr_C + Gamma_12_2
    denom = 1 + Gamma_12_2
    h12 = numer / denom * h_fit

    # machine limit the output
    h12_max = 200
    h12 = np.where(h12 >= h12_max, h12_max, h12)
    scor = np.exp(h12)

    return scor
