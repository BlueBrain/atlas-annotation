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
from collections.abc import Collection
from typing import Any

import numpy as np
from atlalign.metrics import iou_score
from atlas_alignment_meter import core
from scipy import stats

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


def jaggedness(
    annot_vol: np.ndarray,
    region_id: int,
    region_meta: RegionMeta | None = None,
    axis: int = 0,
) -> float:
    """Compute the jaggedness of region ID for the specified annotation volume.

    Parameters
    ----------
    annot_vol
        An annotation volume.
    region_id
        Region ID to compute the jaggedness.
    region_meta
        Region Meta containing all the information concerning the labels.
        If Region Meta is specified, the hierarchy of the regions is taken
        into account: all descendants of the specified region id are replaced
        by the value of the region id before the jaggedness computation.
        If Region Meta is None, the jaggedness is going to be computed on the
        volume as is. No resolution of hierarchy is done before the computation.
    axis
        Axis along which to compute the jaggedness.

    Returns
    -------
    score: float
        The mean of the jaggedness of the given region id.
    """
    if region_meta and not region_meta.is_leaf(region_id):
        descendants = list(region_meta.descendants(region_id))
        annot_vol = np.isin(annot_vol, descendants).astype(int) * region_id

    scores = core.compute(
        annot_vol,
        coronal_axis_index=axis,
        regions=[region_id],
        precomputed_all_region_ids=[region_id],
    )
    score = scores["perRegion"][region_id]["mean"]
    if score is None:
        return np.nan
    else:
        return score


def iou(
    annot_vol_1: np.ndarray,
    annot_vol_2: np.ndarray,
    region_id: int,
    region_meta: RegionMeta | None = None,
) -> float:
    """Compute the intersection over union of given region ID.

    Parameters
    ----------
    annot_vol_1
        The first annotation volume.
    annot_vol_2
        The second annotation volume.
    region_id
        A region ID to compute the intersection over union.
    region_meta
        Region Meta containing all the information concerning the labels.
        If Region Meta is specified, the hierarchy of the regions is taken
        into account: all descendants of the specified region id are replaced
        by the value of the region id before the jaggedness computation.
        If Region Meta is None, the jaggedness is going to be computed on the
        volume as is. No resolution of hierarchy is done before the computation.

    Returns
    -------
    float
        Intersection Over Union of that given region ID.
    """
    if region_meta and not region_meta.is_leaf(region_id):
        descendants = list(region_meta.descendants(region_id))
        annot_vol_1 = np.isin(annot_vol_1, descendants).astype(int) * region_id
        annot_vol_2 = np.isin(annot_vol_2, descendants).astype(int) * region_id

    return float(
        iou_score(annot_vol_1, annot_vol_2, k=region_id, disable_check=True)[0]
    )


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
    cond_entropy: float
        Conditional entropy of the densities of Nissl depending on the brain regions.
    """
    n_pixels = (atlas != 0).sum()
    label_values, count_values = np.unique(atlas[atlas != 0], return_counts=True)
    all_region_entropy = []
    for label, count in zip(label_values, count_values):
        all_region_entropy.append(dist_entropy(nissl[atlas == label]) * count)

    cond_entropy = np.sum(all_region_entropy) / n_pixels
    return cond_entropy


def evaluate_region(
    region_ids: Collection[int],
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
        "region_ids": sorted(region_ids),
        "level": [region_meta.level[id_] for id_ in region_ids],
        "descendants": desc,
    }

    # Jaggedness
    mask = np.isin(atlas, desc)
    global_jaggedness = jaggedness(mask, region_id=1, region_meta=None)
    per_region_jaggedness = {}
    for region_id in region_ids:
        per_region_jaggedness[region_id] = jaggedness(
            atlas, region_id=region_id, region_meta=region_meta
        )

    results["jaggedness"] = {
        "global": global_jaggedness,
        "per_region": per_region_jaggedness,
    }

    # Intersection Over Union
    mask_ref = np.isin(reference, desc)
    global_iou = iou(mask_ref, mask, region_id=1, region_meta=None)
    per_region_iou = {}
    for region_id in region_ids:
        per_region_iou[region_id] = iou(
            reference, atlas, region_id=region_id, region_meta=region_meta
        )

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
    cond_entropy = conditional_entropy(nissl, atlas)
    results["global"] = {
        "brain_entropy": brain_entropy,
        "conditional_entropy": cond_entropy,
    }

    return results
