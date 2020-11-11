"""Notebook utils."""
from deal.atlas import get_misalignment


def print_misalignments(unfurled_atlas_1, unfurled_atlas_2):
    """Print misalignment for every region hierarchy level.

    Parameters
    ----------
    unfurled_atlas_1: np.ndarray
        First atlas of shape (n_level, n_slice, height, width).

    unfurled_atlas_2: np.ndarray
        Second atlas of the same shape as unfurled_atlas_1.
    """
    max_level = len(unfurled_atlas_1)
    for level in range(max_level):
        atlas_1 = unfurled_atlas_1[level]
        atlas_2 = unfurled_atlas_2[level]
        mis = get_misalignment(atlas_1, atlas_2)
        mis_fg = get_misalignment(atlas_1, atlas_2, fg_only=True)
        print(
            f"Misalignment at level {max_level - level - 1:2d} (all / foreground): "
            f"{mis * 100:6.2f}% / {mis_fg * 100:6.2f}%"
        )
