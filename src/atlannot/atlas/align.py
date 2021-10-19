# Copyright 2021, Blue Brain Project, EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions related to atlas alignment."""
import warnings

import numpy as np


def unfurl_regions(atlas, meta, progress_bar=None):
    """Separate regions by hierarchy level.

    Each slice of the brain atlas is expanded into multiple
    copies with each subsequent copy having the last hierarchy
    level of the previous copy removed.

    For example, if a given slice has the region hierarchy up to
    depth 2, i. e. it has regions at level 0 (= the background),
    level 1, and level 2, then the slice will be expanded into
    3 slices:

        - The original slice
        - The slice with regions at level 2 removed
        - The slice with regions at levels 2 and 1 removed (leaving
          just the background)

    Parameters
    ----------
    atlas : np.ndarray
        An annotation atlas volume with shape `(n_slices, height, width)`.
    meta : atlannot.atlas.RegionMeta
        The region metadata. Holds the information about the region
        hierarchy in the atlas.
    progress_bar : callable, optional
        A progress bar function that maps an iterable onto itself
        and produces a progress bar as a side effect. Notable examples
        are `tqdm.tqdm` and `tqdm.notebook.tqdm`.

    Returns
    -------
    unfurled_atlas : np.ndarray
        The unfurled atlas. It will have the shape
        `(n_levels, n_slices, height, width)` where `n_levels` is
        the maximal region hierarchy level across all slices.
    """
    max_level = max(meta.level.values())
    unfurled = [atlas.copy()]

    to_remove = range(max_level, 0, -1)
    if progress_bar is not None:
        to_remove = progress_bar(to_remove)

    for remove_level in to_remove:
        atlas = atlas.copy()
        # Map regions at `remove_level` to the IDs of their parents
        for region_id in meta.ids_at_level(remove_level):
            parent_id = meta.parent_id[region_id]
            atlas[atlas == region_id] = parent_id
        unfurled.append(atlas)

    unfurled_atlas = np.stack(unfurled)

    return unfurled_atlas


def get_misalignment(data_1, data_2, fg_only=False):
    """Compute misalignment between annotation data.

    Parameters
    ----------
    data_1 : np.ndarray
        The first annotation data. Can have any shape.
    data_2 : np.ndarray
        The second annotation data. Shape should match that of `data_1`.
    fg_only : bool, optional
        If true then only the foreground is considered for the evaluation.
        Foreground pixels are complimentary to background. Background is
        where both data arrays are zero.

    Returns
    -------
    misalignment : float
        The misalignment between the annotation data.

    Raises
    ------
    ValueError
        If the shapes of the data don't match.
    """
    if data_1.shape != data_2.shape:
        raise ValueError("Data have to be of the same shape")
    unequal = data_1 != data_2
    if fg_only:
        mask = (data_1 != 0) | (data_2 != 0)
        unequal = unequal[mask]

    misalignment = np.sum(unequal) / (unequal.size or 1)

    return misalignment


def specific_label_iou(data_1, data_2, specific_label):
    """Compute intersection over union for a given label.

    Parameters
    ----------
    data_1 : np.ndarray
        The first annotation data. Can have any shape.
    data_2 : np.ndarray
        The second annotation data. Shape should match that of `data_1`.
    specific_label : int
        Label for which it is wanted to compute the IOU.

    Returns
    -------
    iou : float
        The IOU for the given label.

    Raises
    ------
    ValueError
        If the shapes of the data don't match.
    """
    if data_1.shape != data_2.shape:
        raise ValueError("Data have to be of the same shape")

    data_1 = data_1 == specific_label
    data_2 = data_2 == specific_label

    intersection = np.logical_and(data_1, data_2)
    union = np.logical_or(data_1, data_2)

    if union.sum() == 0:
        iou = np.nan
        warnings.warn(
            f"It seems the specific label "
            f"{specific_label} does not exist on the input images."
        )
    else:
        iou = intersection.sum() / union.sum()

    return iou


def warp(atlas, df_per_slice):
    """Warp the atlas with displacement fields.

    Parameters
    ----------
    atlas : iterable of np.ndarray
        An annotation atlas. Can be an `np.ndarray` of shape
        `(n_slices, ...)` or any iterable over atlas slices.
    df_per_slice : iterable of atldld.base.DisplacementField
        The displacement fields for each brain slice.

    Returns
    -------
    warped_atlas : np.ndarray
        The warped atlas.
    """
    warped_atlas = np.stack(
        [df.warp_annotation(img) for df, img in zip(df_per_slice, atlas)]
    )
    return warped_atlas
