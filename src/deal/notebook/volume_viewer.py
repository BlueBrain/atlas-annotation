"""Implementation of the VolumeViewer class."""
from functools import partial

import IPython.display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from .util import image_grid


class VolumeViewer(widgets.VBox):
    """Widget for plotting 3D volumes.

    This widget takes one or multiple brain volumes of the same
    shape. These volumes are slices along a given axis and the
    2D slices are shown in a grid.

    Using interactive widgets it is possible to change the axis
    and the slice index.

    Parameters
    ----------
    volumes : dict
        Dictionary with volume titles as keys and the actual volumes
        as values.
    rotations : sequence
        A sequence of three integers representing the default rotation
        along each of the three axes in multiples of 90 degrees.
        Positive sign corresponds to rotations counter-clockwise.
    n_columns : int
        The number of columns in the image grid plot.
    plot_width : float
        The width of the volume plot in inches.
    """

    def __init__(self, volumes, rotations=None, n_columns=2, plot_width=12):
        super().__init__()

        if len(volumes) == 0:
            raise ValueError("Need at least one volume")
        shapes = [volume.shape for volume in volumes.values()]
        if any(len(shape) < 3 for shape in shapes):
            raise ValueError("A volume has fewer than 3 dimensions")
        if any(s1[:3] != s2[:3] for s1, s2 in zip(shapes, shapes[1:])):
            raise ValueError("Volumes have different x-y-z dimensions")
        if rotations is None:
            rotations = [0, 0, 0]
        elif len(list(rotations)) != 3 or not all(
            isinstance(x, int) for x in rotations
        ):
            raise ValueError("Rotations must be a sequence of 3 integers")

        self.volumes = volumes
        self.dims = list(volumes.values())[0].shape[:3]
        self.axis = 0
        self.ids = [dim // 2 for dim in self.dims]
        self.rot = list(rotations)
        self.plot_width = plot_width
        self.n_columns = n_columns

        # Widgets
        self.axis_select = widgets.ToggleButtons(
            options=[("X", 0), ("Y", 1), ("Z", 2)],
            button_style="success",
            style={"button_width": "auto"},
        )
        self.rot_left = widgets.Button(
            icon="rotate-left",
            button_style="success",
            layout={"width": "auto"},
        )
        self.rot_right = widgets.Button(
            icon="rotate-right",
            button_style="success",
            layout={"width": "auto"},
        )
        self.idx_slider = widgets.IntSlider(
            value=self.ids[self.axis],
            max=self.dims[self.axis] - 1,
            continuous_update=False,
        )
        self.output = widgets.Output()

        # Widget callbacks
        self.idx_slider.observe(self._cb_idx_changed, "value")
        self.axis_select.observe(self._cb_axis_changed, "value")
        self.rot_left.on_click(partial(self._cb_rotate, 1))
        self.rot_right.on_click(partial(self._cb_rotate, -1))

        # Main layout
        label_layout = widgets.Layout(
            width="10ch",
            display="flex",
            justify_content="center",
        )
        self.children = (
            widgets.HBox(
                children=[
                    widgets.Label("Axis", layout=label_layout),
                    self.axis_select,
                    widgets.Label("Rotation", layout=label_layout),
                    self.rot_left,
                    self.rot_right,
                ]
            ),
            widgets.HBox(
                children=[
                    widgets.Label("Slice index", layout=label_layout),
                    self.idx_slider,
                ]
            ),
            self.output,
        )
        self.layout = {"border": "1px solid black"}

        # Display
        self._set_axis(0)
        self._render()

    def _render(self):
        images = {
            k: np.rot90(
                volume.take(self.ids[self.axis], axis=self.axis),
                self.rot[self.axis],
            )
            for k, volume in self.volumes.items()
        }
        fig = image_grid(images, n_columns=self.n_columns, plot_width=self.plot_width)

        with self.output:
            IPython.display.clear_output(wait=True)
            plt.show(fig)

    def _set_axis(self, axis):
        # Save slider state for the old axis
        self.ids[self.axis] = self.idx_slider.value

        # Set new axis and update the index slider
        self.axis = axis
        self.idx_slider.value = self.ids[self.axis]
        self.idx_slider.max = self.dims[self.axis] - 1

    def _cb_axis_changed(self, change):
        """Handle callback for the axis selection widget."""
        self._set_axis(change["new"])
        self._render()

    def _cb_idx_changed(self, change):
        """Handle callback for the index slider."""
        self.ids[self.axis] = change["new"]
        self._render()

    def _cb_rotate(self, direction, button):  # noqa
        """Handle callback for rotation buttons."""
        self.rot[self.axis] += direction
        self.rot_left.disabled = True
        self.rot_right.disabled = True
        self._render()
        self.rot_left.disabled = False
        self.rot_right.disabled = False
