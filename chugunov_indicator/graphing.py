def D_T_setup(ax) -> None:
    """Sets up the axes of a D-T graph."""

    try:
        for ax_ in ax:
            D_T_setup(ax_)
    except TypeError:
        ax.set_xlabel("$T$")
        ax.set_ylabel("$\\rho$")

        ax.set_xscale("log")
        ax.set_yscale("log")

def D_T_colorgraph(T, D, y, fig, ax, norm=None) -> tuple:
    """Creates a colormesh of y-data on a D-T graph."""

    D_T_setup(ax)

    cb = ax.pcolormesh(T, D, y, norm=norm)
    cbar = fig.colorbar(cb)

    return cb, cbar
