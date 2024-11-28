from dataclasses import dataclass

import numpy as np
import keras

from pynucastro.screening import PlasmaState, ScreenFactors

from .data_generation import ScreeningFactorData

__all__ = ["ScreeningFactorNetwork"]

@dataclass
class ScreeningFactorNetwork:
    """Contains a `keras` neural network trained to identify the importance
    of screening for a given temperature, density, and composition."""

    train: ScreeningFactorData
    validate: ScreeningFactorData
    test: ScreeningFactorData
    seed: int | None = None

    def __post_init__(self) -> None:

        # sets rng
        if self.seed is not None:
            keras.utils.set_random_seed(self.seed)

        # Computes output and class bias\
        self.class_weight = {
            0: np.count_nonzero(self.train.outputs == 0),
            1: np.count_nonzero(self.train.outputs == 1)
        }

        # Normalization layer
        self.normalization = keras.layers.Normalization(axis=-1)
        self.normalization.adapt(self.train.inputs)

        # Model layers
        self.model = keras.Sequential(
            [
                self.normalization,
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        # Model metrics, loss functions, and callbacks
        self.metrics = [
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            #keras.metrics.AUC(name="auc")
        ]

        self.loss = [
            keras.losses.BinaryCrossentropy()
        ]

        self.callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='fp',
                verbose=1,
                patience=25,
                mode='min',
                restore_best_weights=True
            )
        ]

    def compile(self, learning_rate: float = 1e-3) -> None:
        """Compiles the model."""

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.loss,
            metrics=self.metrics
        )

    def fit_model(self, epochs: int = 300, verbose: int = 0) -> keras.callbacks.History:
        """Fits the model to the data and computes its score.
        
        Keyword arguments:
            `verbose`: how verbose `self.model.fit` should be
        """

        self.model.fit(
            x=self.train.inputs,
            y=self.train.outputs,
            batch_size=2048,
            epochs=epochs,
            verbose=verbose,
            callbacks=self.callbacks,
            validation_data=(self.validate.inputs, self.validate.outputs),
            class_weight=self.class_weight
        )

        self.score = self.model.evaluate(x=self.test.inputs, y=self.test.outputs, verbose=verbose)

    def predict_params(self, T: float, D: float,
                       abar: float, z2bar: float,
                       z1: int, z2: int,
                       confidence: float = 0.5) -> bool:
        """Predicts whether screening can be skipped for supplied parameters."""        

        inputs = np.vstack((3*np.log10(T) - np.log10(D), abar, np.log10(z2bar), z1, z2)).T

        prediction = self.model.predict(inputs).squeeze() >= confidence
        if not prediction.ndim:
            return prediction.item()
        return prediction

    def predict_states(self, state: PlasmaState, scn_fac: ScreenFactors) -> bool:
        """
        Predicts whether screening can be skipped for
        a supplied plasma state and screening factors."""

        y0 = 3*np.log10(state.temp) - np.log10(state.dens)
        return self.predict_params(state.temp, state.dens,
                                   state.abar, state.z2bar,
                                   scn_fac.z1, scn_fac.z2)
