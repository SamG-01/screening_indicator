__all__ = ["D_T_setup", "D_T_colorgraph"]

def D_T_setup(ax) -> None:
    """Sets up the axes of a D-T graph."""

    ax.set_xlabel("$T$")
    ax.set_ylabel("$\\rho$")

    ax.set_xscale("log")
    ax.set_yscale("log")

def D_T_colorgraph(T, D, y, fig, ax):
    """Creates a colormesh of y-data on a D-T graph."""

    D_T_setup(ax)

    cb = ax.pcolormesh(T, D, y)
    cbar = fig.colorbar(cb)

    return cb, cbar
