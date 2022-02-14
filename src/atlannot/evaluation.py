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

from typing import Any

import numpy as np
from atlalign.metrics import iou_score
from atlas_alignment_meter import core
from scipy import stats

from atlannot.region_meta import RegionMeta

REGIONS_TO_EVALUATE = {
    "Somatosensory Cortex": [453],
    "Visual Cortex": [669],
    "Rest of Cortex": [315],
    "Thalamus": [549],
    "Hippocampus": [1080],
    "Cerebullum": [512],
    "Basal Ganglia": [477, 1022, 470, 381],
}


def compute_jaggedness(
    volume: np.ndarray,
    region_ids: list[int] | None = None,
    axis: int = 0,
    precomputed_region_ids: list[int] | None = None,
) -> dict[int, float]:
    """Compute the jaggedness of given region IDs for the specified volume.

    Parameters
    ----------
    volume
        Input volume.
    region_ids
        List of region IDs to compute the jaggedness. If None, the jaggedness
        is computed for all the region IDs present in the volume.
    axis
        Axis along which to compute the jaggedness.
    precomputed_region_ids
        Optional: Precomputed unique ids present in the volume (from np.unique)

    Returns
    -------
    results: dict[int, float]
        Dictionary containing the region id as keys and the mean of the
        jaggedness of that given region id as values.
    """
    if precomputed_region_ids is None:
        all_regions_ids = np.unique(volume)
    else:
        all_regions_ids = precomputed_region_ids

    if region_ids is None:
        region_ids = all_regions_ids
        region_ids = np.delete(region_ids, np.where(region_ids == 0))
    else:
        keep = list(np.isin(region_ids, all_regions_ids, assume_unique=True))
        region_ids = np.array(region_ids)[keep]

    metrics = core.compute(
        volume,
        coronal_axis_index=axis,
        regions=list(region_ids),
        precomputed_all_region_ids=all_regions_ids,
    )

    return {
        region_id: metrics["perRegion"][region_id]["mean"] for region_id in region_ids
    }


def compute_iou(
    vol_true: np.ndarray,
    vol_pred: np.ndarray,
    region_ids: list[int] | None = None,
) -> dict[int, float]:
    """Compute the intersection over union of given region IDs.

    Parameters
    ----------
    vol_true
        Input volume considered as ground truth.
    vol_pred
        Input volume considered as prediction.
    region_ids
        List of region IDs to compute the intersection over union.
        If None, the IoU is computed for all the region IDs present in the volume.

    Returns
    -------
    results: dict[int, float]
        Dictionary containing the region id as keys and the average of the
        intersection over union of that given region id as values.
    """
    results = {}
    label_vol_true = np.unique(vol_true)
    if region_ids is None:
        region_ids = label_vol_true.copy()

    for region_id in region_ids:
        if region_id in label_vol_true:
            results[region_id] = iou_score(vol_true, vol_pred, k=region_id)[0]

    return results


def dist_entropy(
    data: np.ndarray,
    bins: int = 100,
) -> float:
    """Compute the entropy of the value distribution of data.

    Parameters
    ----------
    data
        Data for which to compute the entropy.
    bins
        Number of bins used to compute the histogram of the densities.

    Returns
    -------
    float
        Entropy of the densities of Nissl at a given region value.
    """
    hist, _ = np.histogram(data, bins=bins)
    return stats.entropy(hist)


def compute_conditional_entropy(
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
    n_pixels = (atlas != 0).sum()
    label_values, count_values = np.unique(atlas[atlas != 0], return_counts=True)
    all_region_entropy = []
    for label, count in zip(label_values, count_values):
        all_region_entropy.append(dist_entropy(nissl[atlas == label]) * count)

    conditional_entropy = np.sum(all_region_entropy) / n_pixels
    return conditional_entropy


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
    results = {
        "region_ids": region_ids,
        "level": [region_meta.level[id_] for id_ in region_ids],
        "descendants": desc,
    }

    # Jaggedness
    mask = np.isin(atlas, desc)
    global_jaggedness = compute_jaggedness(mask, region_ids=[1])[1]
    per_region_jaggedness = compute_jaggedness(atlas, region_ids=desc)

    results["jaggedness"] = {
        "global": global_jaggedness,
        "per_region": per_region_jaggedness,
    }

    # Intersection Over Union
    mask_ref = np.isin(reference, desc)
    global_iou = compute_iou(mask_ref, mask, region_ids=[1])[1]
    per_region_iou = compute_iou(reference, atlas, region_ids=desc)

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
    brain_entropy = dist_entropy(nissl[atlas != 0])
    conditional_entropy = compute_conditional_entropy(nissl, atlas)
    results["global"] = {
        "brain_entropy": brain_entropy,
        "conditional_entropy": conditional_entropy,
    }

    return results
