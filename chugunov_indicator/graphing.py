from dataclasses import dataclass
from typing import Callable

import matplotlib
from matplotlib.widgets import Slider, Button

@dataclass()
class SliderGraph:
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    params: dict[str: (float, float)]
    update: Callable[[float], None]

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
            self.ax_sliders[param] = self.fig.add_axes([0.15, 0.4 - j*0.05, 0.65, 0.03])

            # Create sliders
            self.sliders[param] = Slider(self.ax_sliders[param], param, min_, max_, valinit=min_)
            # Call update function when slider value is changed
            self.sliders[param].on_changed(self.update)

        # Create axes for reset button and create button
        self.resetax = self.fig.add_axes([0.8, 0.105, 0.1, 0.04])
        self.button = Button(self.resetax, 'Reset', color='gold', hovercolor='skyblue')

        def resetSlider(event) -> None:
            for slider in self.sliders.values():
                slider.reset()

        self.button.on_clicked(resetSlider)

    def change_update(self, new_update: Callable[[float], None]) -> None:
        """Changes the update function for the sliders."""

        for slider in self.sliders.values():
            slider.on_changed(new_update)
