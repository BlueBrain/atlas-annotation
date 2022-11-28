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
"""Evaluation."""
from __future__ import annotations

import logging
import warnings
from collections.abc import Collection
from typing import Any

import numpy as np
from numpy import ma
from scipy import stats

from atlannot._atlalign import iou_score
from atlannot.region_meta import RegionMeta

logger = logging.getLogger(__name__)

REGIONS_TO_EVALUATE = {
    "Somatosensory Cortex": [453],
    "Visual Cortex": [669],
    "Rest of Cortex": [315],
    "Thalamus": [549],
    "Hippocampus": [1080],
    "Cerebullum": [512],
    "Basal Ganglia": [477, 1022, 470, 381],
}


def check_installed() -> bool:
    """Check whether the atlas_alignment_meter package is installed.
    Returns
    -------
    bool
        Whether the atlas_alignment_meter package is installed
    """
    try:
        import atlas_alignment_meter  # noqa: F401
    except ImportError:
        return False
    else:
        return True


def how_to_install_msg() -> str:
    """Get installation instructions for atlas_alignment_meter.
    Returns
    -------
    str
        The instructions on how to install atlas_alignment_meter.
    """
    return (
        "To install atlas_alignment_meter package run "
        '"pip install git+https://github.com/BlueBrain/atlas-alignment-meter.git". '
        'The annotation package was tested using atlas_alignment_meter version "1.0.0".'
    )


def warn_if_not_installed() -> None:
    """Issue a UserWarning if atlas_alignment_meter is not installed."""
    if not check_installed():
        warnings.warn(
            f"atlas_alignment_meter is not installed. {how_to_install_msg()}",
            stacklevel=3,
        )


def jaggedness(
    volume: np.ndarray,
    axis: int = 0,
    region_ids: Collection[int] | None = None,
    all_region_ids: Collection[int] | None = None,
) -> dict[int, float]:
    """Compute the jaggedness of given region IDs for the specified volume.

    Parameters
    ----------
    volume
        An annotation volume.
    axis
        Axis along which to compute the jaggedness.
    region_ids
        A collection of region IDs to compute the jaggedness. If None, the
        jaggedness is computed for all the region IDs present in the volume.
    all_region_ids
        A collection of unique IDs in the volume provided. Can be useful to
        speed up the computation. It's typically computed using
        `np.unique(volume)`. If not provided, it will be set to
        `np.unique(volume)`.

    Returns
    -------
    results: dict[int, float]
        Dictionary containing the region id as keys and the mean of the
        jaggedness of that given region id as values.
    """
    from atlas_alignment_meter import core

    if all_region_ids is None:
        all_region_ids = np.unique(volume)
    all_region_ids = set(all_region_ids)

    if region_ids is None:
        missing: set[int] = set()
        region_ids = all_region_ids
    else:
        missing = {id_ for id_ in region_ids if id_ not in all_region_ids}
        region_ids = set(region_ids) - missing
    region_ids.discard(0)

    # Set the score of region IDs not found in the annotation volume to NaN.
    # This behaviour is consistent with what happens in the iou function.
    results = {id_: np.nan for id_ in missing}

    # core.compute breaks if region_ids is empty, so short-circuit.
    if not region_ids:
        return results

    metrics = core.compute(
        volume,
        coronal_axis_index=axis,
        regions=list(region_ids),
        precomputed_all_region_ids=list(all_region_ids),
    )

    for region_id, scores in metrics["perRegion"].items():
        results[region_id] = scores["mean"]

    return results


def iou(
    annot_vol_1: np.ndarray,
    annot_vol_2: np.ndarray,
    region_ids: Collection[int] | None = None,
) -> dict[int, float]:
    """Compute the intersection over union of given region IDs.

    Parameters
    ----------
    annot_vol_1
        The first annotation volume.
    annot_vol_2
        The second annotation volume.
    region_ids
        A sequence of region IDs to compute the intersection over union.
        If None, the IoU is computed for all the region IDs present in at least
        one of the volumes.

    Returns
    -------
    results: dict[int, float]
        Dictionary with the region IDs as keys and the intersection over union
        scores as values.
    """
    if region_ids is None:
        region_ids = np.union1d(np.unique(annot_vol_1), np.unique(annot_vol_2))
    else:
        region_ids = sorted(set(region_ids))

    scores = {}
    for id_ in region_ids:
        # The check we disable would check if the shapes of the volumes match
        # and that the region ID is present in both annotation volumes. This
        # can be expensive, so we disable it. If a region ID is not in any of
        # the volumes then the IoU will be NaN.
        score, _ = iou_score(annot_vol_1, annot_vol_2, k=id_)
        scores[id_] = score

    return scores


def entropy(
    arr: np.ndarray | ma.MaskedArray,
    *,
    n_bins: int = 256,
    value_range: tuple[int, int] | None = None,
) -> float:
    """Compute the entropy of the value distribution in an array.

    The entropy is computed over the discrete value distribution of the
    given data, which is obtained by putting all values into a given number
    of bins that are uniformly distributed over a given value range. Note that
    the exact pixel positions and the shape of the data array don't matter,
    only the value distribution.

    In order to obtain results that are compatible with each other it is
    important that the entropy is computed using the same bins. This can be
    ensured by using fixed values for the `n_bins` and `value_range` parameters.

    Parameters
    ----------
    arr
        The array for which to compute the entropy.
    n_bins
        The number of histogram bins. The entropy is computed over the
        discretized value distribution of the given array. To obtain the
        discretization, a histogram of all values in the given array is
        computed using the given number of equally sized bins across the
        value range specified by the `value_range` parameter.
    value_range
        The value range over which to compute the data distribution
        specified the lower and upper bound. If not provided then the min and
        max of all array values will be used. In order to obtain compatible
        results for different arrays or for different masked regions of the
        same array it is important to make sure that the histograms are
        computed over exactly the same bins. The bins are uniquely specified
        by the `n_bins` and `value_range` parameters, therefore, for
        compatible results it is important to keep these constant.

    Returns
    -------
    float
        Entropy of the (masked) array.
    """
    if ma.isMaskedArray(arr):
        arr = arr.compressed()
        if value_range is None:
            warnings.warn(
                "Computing entropy on a masked array, but the value range "
                "was not provided. As a result the bins will be determined "
                "based on the intensities within the masked region only. "
                "This will give different bins for different masks.",
                stacklevel=2,
            )
    if value_range is None:
        value_range = arr.min(), arr.max()

    bins = np.linspace(*value_range, n_bins + 1)
    hist, _ = np.histogram(arr.ravel(), bins=bins)

    return stats.entropy(hist, base=n_bins)


def conditional_entropy(
    nissl: np.ndarray,
    atlas: np.ndarray,
) -> float:
    """Compute entropies of Nissl densities.

    Parameters
    ----------
    nissl
        Nissl volume.
    atlas
        Annotation atlas.

    Returns
    -------
    conditional_entropy: float
        Conditional entropy of the densities of Nissl depending on the brain regions.
    """
    value_range = nissl.min(), nissl.max()
    weighted_entropies = []
    n_voxels = 0
    for label, count in zip(*np.unique(atlas, return_counts=True)):
        if label == 0:  # skip background
            continue
        n_voxels += count
        nissl_region = ma.masked_where(atlas != label, nissl)
        entropy_score = entropy(nissl_region, value_range=value_range)
        weighted_entropies.append(entropy_score * count)

    return np.sum(weighted_entropies) / n_voxels


def evaluate_region(
    region_ids: list[int],
    atlas: np.ndarray,
    reference: np.ndarray,
    region_meta: RegionMeta,
) -> dict[str, Any]:
    """Evaluate the atlas.

    Parameters
    ----------
    region_ids
        Region IDs to evaluate.
    atlas
        Atlas to evaluate.
    reference
        Reference atlas.
    region_meta
        Region Meta containing all the information concerning the labels.

    Returns
    -------
    results: dict[str, Any]
        Dictionary containing the results of the region evaluation.
    """
    desc = list(region_meta.descendants(region_ids))

    # Put some metadata about the region
    results: dict[str, dict[str, Any] | list[int]] = {
        "region_ids": region_ids,
        "level": [region_meta.level[id_] for id_ in region_ids],
        "descendants": desc,
    }

    # Jaggedness
    mask = np.isin(atlas, desc)
    global_jaggedness = jaggedness(mask, region_ids=[1])[1]
    per_region_jaggedness = jaggedness(atlas, region_ids=desc)

    results["jaggedness"] = {
        "global": global_jaggedness,
        "per_region": per_region_jaggedness,
    }

    # Intersection Over Union
    mask_ref = np.isin(reference, desc)
    global_iou = iou(mask_ref, mask, region_ids=[1])[1]
    per_region_iou = iou(reference, atlas, region_ids=desc)

    results["iou"] = {
        "global": global_iou,
        "per_region": per_region_iou,
    }
    return results


def evaluate(
    atlas: np.ndarray,
    nissl: np.ndarray,
    reference: np.ndarray,
    region_meta: RegionMeta,
    regions_to_evaluate: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Evaluate the atlas.

    Parameters
    ----------
    atlas
        Atlas to evaluate.
    nissl
        Corresponding Nissl Volume.
    reference
        Reference atlas.
    region_meta
        Region Meta containing all the information concerning the labels.
    regions_to_evaluate
        Regions to evaluate. If None, regions to evaluation considered are
        REGIONS_TO_EVALUATE = {
        "Somatosensory Cortex": [453],
        "Visual Cortex": [669],
        "Rest of Cortex": [315],
        "Thalamus": [549],
        "Hippocampus": [1080],
        "Cerebullum": [512],
        "Basal Ganglia": [477, 1022, 470, 381],}

    Returns
    -------
    results: dict[str, Any]
        Dictionary containing the results of the evaluation.
    """
    results = {}
    if regions_to_evaluate is None:
        regions_to_evaluate = REGIONS_TO_EVALUATE

    for name, region_ids in regions_to_evaluate.items():
        results[name] = evaluate_region(region_ids, atlas, reference, region_meta)

    # Entropies
    brain_entropy = entropy(nissl[atlas != 0])
    cond_entropy = conditional_entropy(nissl, atlas)
    results["global"] = {
        "brain_entropy": brain_entropy,
        "conditional_entropy": cond_entropy,
    }

    return results
