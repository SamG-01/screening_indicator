import numpy as np

import pynucastro as pyna
import yt

__all__ = ["DetonationData"]

class DetonationData:
    """Loads and stores double detonation data."""

    reaclib_library = pyna.ReacLibLibrary()

    def __init__(self, file: str, threshold: float = 1.01) -> None:
        self.threshold = threshold

        # Loads the data
        ds = yt.load(file)
        self.raw = ds.all_data()
        self.data = {}

        # Stores temperature and density
        self.data["temp"] = np.array(self.raw["Temp"])
        self.data["dens"] = np.array(self.raw["density"])

        # Builds composition
        nuclei = [field[2:-1] for field in np.array(ds.field_list)[:,1] if "X(" in field]
        self.comp = pyna.Composition(nuclei)

        # Computes abar, zbar, and z2bar
        Xs = np.array([self.raw[f"X({nucleus})"] for nucleus in nuclei]).T
        As = np.array(list(self.comp.A.values()))
        Zs = np.array(list(self.comp.Z.values()))
        Ys = Xs / As

        self.data["abar"] = 1 / np.sum(Ys, axis=1)
        self.data["zbar"] = np.sum(Zs * Ys, axis=1) * self.data["abar"]
        self.data["z2bar"] = np.sum(Zs**2 * Ys, axis=1) * self.data["abar"]

        # Filters out screening pairs
        mynet = self.reaclib_library.linking_nuclei(self.comp.keys())
        pynet = pyna.PythonNetwork(libraries=[mynet])

        screen_map = pyna.screening.get_screening_map(
            pynet.get_rates(),
            symmetric_screening=pynet.symmetric_screening
        )
        self.data["ScreenFactors"] = [
            {
                "z1": pair.n1.Z,
                "a1": pair.n1.A,
                "z2": pair.n2.Z,
                "a2": pair.n2.A
            }
            for pair in screen_map
        ]

    def D_T_meshgrid(self, num=100) -> list[np.ndarray, np.ndarray]:
        """
        Returns a log-log meshgrid for density and temperature plotting.
        
        `num`: side length of the grid
        """

        temp = self.data["temp"]
        dens = self.data["dens"]

        T_min, T_max = temp.min(), temp.max()
        D_min, D_max = dens.min(), dens.max()

        T_ = np.logspace(np.log10(T_min), np.log10(T_max), num=num)
        D_ = np.logspace(np.log10(D_min), np.log10(D_max), num=num)

        return np.meshgrid(T_, D_)
