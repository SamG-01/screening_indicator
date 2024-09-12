from matplotlib import pyplot as plt

__all__ = ["D_T_colorgraph"]

def D_T_colorgraph(T, D, y):
    """Creates a colormesh of y data on a D-T graph."""

    cb = plt.pcolormesh(T, D, y)
    cbar = plt.colorbar(cb)

    plt.xlabel("$T$")
    plt.ylabel("$\\rho$")

    plt.xscale("log")
    plt.yscale("log")

    return cb, cbar
