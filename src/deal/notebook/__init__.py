"""Notebook utils."""
import sys

import matplotlib.pyplot as plt

from deal.atlas import get_misalignment


def print_misalignments(unfurled_atlas_1, unfurled_atlas_2, file=None):
    """Print misalignment for every region hierarchy level.

    Parameters
    ----------
    unfurled_atlas_1 : np.ndarray
        First atlas of shape (n_level, n_slice, height, width).
    unfurled_atlas_2 : np.ndarray
        Second atlas of the same shape as unfurled_atlas_1.
    file
        The output file. If None then sys.stdout is used.
    """
    if file is None:
        file = sys.stdout

    max_level = len(unfurled_atlas_1)
    for level in range(max_level):
        atlas_1 = unfurled_atlas_1[level]
        atlas_2 = unfurled_atlas_2[level]
        mis = get_misalignment(atlas_1, atlas_2)
        mis_fg = get_misalignment(atlas_1, atlas_2, fg_only=True)
        print(
            f"Misalignment at level {max_level - level - 1:2d} (all / foreground): "
            f"{mis * 100:6.2f}% / {mis_fg * 100:6.2f}%",
            file=file,
        )


def image_grid(image_dict, n_columns=2, plot_width=12, fig_title=None, save_as=None):
    """Plot images in a grid.

    Parameters
    ----------
    image_dict : dict
        Mapping image title => image data.
    n_columns : int
        The number of columns in the plot grid.
    plot_width : int
        The width of the plot in inches (same as for the figsize parameter
        in the matplotlib library).
    fig_title : str or None, optional
        The figure title.
    save_as : str or pathlib.Path, optional
        Save the figure as the given file.
    """
    # Compute the number or fows
    n_rows = len(image_dict) // n_columns
    if n_rows * n_columns < len(image_dict):
        n_rows += 1

    # Compute the size of individual axes
    max_hw_ratio = max(img.shape[0] / img.shape[1] for img in image_dict.values())
    ax_width = plot_width / n_columns
    ax_height = max_hw_ratio * ax_width

    # Create figure
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(n_columns * ax_width, n_rows * ax_height),
        constrained_layout=True,
    )

    # Don't plot the axes
    for ax in axs.ravel():
        ax.set_axis_off()

    for ax, (title, img) in zip(axs.ravel(), image_dict.items()):
        ax.set_title(title)
        ax.imshow(img)

    if fig_title is not None:
        fig.suptitle(fig_title)

    if save_as is not None:
        fig.savefig(save_as)
