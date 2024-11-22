from dataclasses import dataclass

import numpy as np

from ..chugunov_2009 import chugunov_2009

default_rng = np.random.default_rng()

__all__ = ["ScreeningFactorData"]

@dataclass
class ScreeningFactorData:
    """Stores input and output screening data for keras."""

    size: int = 10**6
    rng: np.random.Generator = default_rng

    def __post_init__(self) -> None:
        # generate the D-T grid
        self.log_T = self.rng.uniform(7, 9.35, self.size)
        self.log_D = self.rng.uniform(-4, 8, self.size)
        self.y0 = 3 * self.log_T - self.log_D

        # generate the parameter space
        self.abar = self.rng.uniform(1, 55, self.size)
        self.log_z2bar = self.rng.uniform(0, 3, self.size)
        self.z1 = self.rng.integers(1, 20, self.size, endpoint=True)
        self.z2 = self.rng.integers(1, 20, self.size, endpoint=True)

        # stack the inputs into keras input vectors
        self.inputs = np.column_stack((self.y0, self.abar, self.log_z2bar,
                                       self.z1, self.z2)).squeeze()

        # exponentiates the log'ed variables for later reference
        self.T = 10**self.log_T
        self.D = 10**self.log_D
        self.z2bar = 10**self.log_z2bar

        # finds the screening factors from the input data
        self.F = chugunov_2009(self.T, self.D, self.abar, 1,
                                     self.z2bar, self.z1, self.z2, 1, 1)
        # checks whether the 
        self.outputs = (self.F <= 1.01).astype(int)
