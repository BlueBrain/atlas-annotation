"""Functionality common to all atlas merge strategies."""
from __future__ import annotations

import numpy as np


def replace(array: np.ndarray, old_value: int, new_value: int) -> None:
    """Replace integer values in a numpy array.

    Parameters
    ----------
    array
        An arbitrary numpy array.
    old_value
        The value to replace.
    new_value
        The new value that replaces the old one.
    """
    array[array == old_value] = new_value


def atlas_remap(
    atlas: np.ndarray, values_from: np.ndarray, values_to: np.ndarray
) -> np.ndarray:
    """Remap atlas values fast.

    This only works if

    * ``values_from`` contains all unique values in ``atlas``,
    * ``values_from`` is sorted.

    In other words, it must be that ``values_from = np.unique(atlas)``.

    Source: https://stackoverflow.com/a/35464758/2804645

    Parameters
    ----------
    atlas
        The atlas volume to remap. Can be of any shape.
    values_from
        The values to map from. It must be that
        ``values_from = np.unique(atlas)``.
    values_to
        The values to map to. Must have the same shape as ``values_from``.

    Returns
    -------
    np.ndarray
        The remapped atlas.
    """
    idx = np.searchsorted(values_from, atlas.ravel())
    new_atlas = values_to[idx].reshape(atlas.shape)

    return new_atlas
