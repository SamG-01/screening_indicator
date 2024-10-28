from dataclasses import dataclass
from typing import Callable, Tuple

import matplotlib
from matplotlib.widgets import Slider, Button

__all__ = ["SliderGraph"]
@dataclass()
class SliderGraph:
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    params: dict[str, Tuple[float, float]]
    update: Callable[[float], None] = lambda val: None

    def __post_init__(self) -> None:
        """
        Creates a new slider graph.
        """

        self.fig.subplots_adjust(bottom=0.55)

        self.ax_sliders = {}
        self.sliders = {}

        for j, param in enumerate(self.params):
            # Create axes
            min_, max_ = self.params[param]
            self.ax_sliders[param] = self.fig.add_axes((0.15, 0.4 - j*0.05, 0.65, 0.03))

            # Create sliders
            self.sliders[param] = Slider(self.ax_sliders[param], param, min_, max_, valinit=min_)

        # Create axes for reset button and create button
        self.resetax = self.fig.add_axes((0.8, 0.105, 0.1, 0.04))
        self.button = Button(self.resetax, 'Reset', color='gold', hovercolor='skyblue')

        def resetSlider(event) -> None:
            for slider in self.sliders.values():
                slider.reset()

        self.button.on_clicked(resetSlider)

    def update_func(self, new_update: Callable[[float], None]) -> None:
        """Changes the update function for the sliders."""

        self.update = new_update
        for slider in self.sliders.values():
            slider.on_changed(self.update)

    def slider_vals(self) -> dict[str: float]:
        """Returns the current slider values."""

        return {param: self.sliders[param].val for param in self.params}
