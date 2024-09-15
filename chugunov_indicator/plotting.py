from matplotlib import pyplot as plt

__all__ = ["D_T_colorgraph"]

def D_T_colorgraph(T, D, y, fig, ax):
    """Creates a colormesh of y-data on a D-T graph."""

    cb = ax.pcolormesh(T, D, y)
    cbar = fig.colorbar(cb)

    ax.set_xlabel("$T$")
    ax.set_ylabel("$\\rho$")

    ax.set_xscale("log")
    ax.set_yscale("log")

    return cb, cbar
