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

        # Stores temperature and density
        self.temp = np.array(self.raw["Temp"])
        self.dens = np.array(self.raw["density"])

        # Builds composition
        nuclei = [field[2:-1] for field in np.array(ds.field_list)[:,1] if "X(" in field]
        self.comp = pyna.Composition(nuclei)

        # Computes abar, zbar, and z2bar
        Xs = np.array([self.raw[f"X({nucleus})"] for nucleus in nuclei]).T
        As = np.array(list(self.comp.A.values()))
        Zs = np.array(list(self.comp.Z.values()))
        Ys = Xs / As

        self.abar = 1 / np.sum(Ys, axis=1)
        self.zbar = np.sum(Zs * Ys, axis=1) * self.abar
        self.z2bar = np.sum(Zs**2 * Ys, axis=1) * self.abar

        # Filters out screening pairs
        mynet = self.reaclib_library.linking_nuclei(self.comp.keys())
        pynet = pyna.PythonNetwork(libraries=[mynet])

        screen_map = pyna.screening.get_screening_map(
            pynet.get_rates(),
            symmetric_screening=pynet.symmetric_screening
        )
        self.ScreenFactors = [
            [pair.n1.Z, pair.n1.A, pair.n2.Z, pair.n2.A]
            for pair in screen_map
        ]
