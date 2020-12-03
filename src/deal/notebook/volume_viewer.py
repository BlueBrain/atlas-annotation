"""Implementation of the VolumeViewer class."""
import IPython.display
import ipywidgets as widgets
import matplotlib.pyplot as plt

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
    n_columns : int
        The number of columns in the image grid plot.
    plot_width : float
        The width of the volume plot in inches.
    """

    def __init__(self, volumes, n_columns=2, plot_width=12):
        super().__init__()

        if len(volumes) == 0:
            raise ValueError("Need at least one volume")
        shapes = [volume.shape for volume in volumes.values()]
        if any(len(shape) < 3 for shape in shapes):
            raise ValueError("A volume has fewer than 3 dimensions")
        if any(s1[:3] != s2[:3] for s1, s2 in zip(shapes, shapes[1:])):
            raise ValueError("Volumes have different x-y-z dimensions")

        self.volumes = volumes
        self.dims = list(volumes.values())[0].shape[:3]
        self.axis = 0
        self.ids = [dim // 2 for dim in self.dims]
        self.plot_width = plot_width
        self.n_columns = n_columns

        self.axis_select = widgets.ToggleButtons(
            options=[("x", 0), ("y", 1), ("z", 2)],
            rows=3,
            button_style="success",
            description="Axis",
        )
        self.idx_slider = widgets.IntSlider(
            value=self.dims[self.axis],
            max=self.ids[self.axis] - 1,
            continuous_update=False,
            description="Slice index",
        )
        self.output = widgets.Output()
        self.idx_slider.observe(self._cb_idx_changed, "value")
        self.axis_select.observe(self._cb_axis_changed, "value")

        self.children = (
            self.axis_select,
            self.idx_slider,
            self.output,
        )
        self.layout = {"border": "1px solid black"}

        self._set_axis(0)
        self._render()

    def _render(self):
        images = {
            k: volume.take(self.ids[self.axis], axis=self.axis)
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
