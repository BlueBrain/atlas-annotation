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
from collections import defaultdict
from collections.abc import Collection
from typing import Any

import numpy as np
from atlalign.metrics import iou_score
from atlas_alignment_meter import core
from scipy import stats

from atlannot.merge.common import atlas_remap
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
    if all_region_ids is None:
        all_region_ids = np.unique(volume)
    all_region_ids = set(all_region_ids)

    if region_ids is None:
        missing = {}
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


def new_jaggedness(
    annot_vol: np.ndarray,
    region_id: int,
    region_meta: RegionMeta,
    axis: int = 0,
) -> float:
    if not region_meta.is_leaf(region_id):
        descendants = region_meta.descendants(region_id)
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


def new_iou(
    annot_vol_1: np.ndarray,
    annot_vol_2: np.ndarray,
    region_id: int,
    region_meta: RegionMeta,
) -> float:
    if not region_meta.is_leaf(region_id):
        descendants = region_meta.descendants(region_id)
        annot_vol_1 = np.isin(annot_vol_1, descendants).astype(int) * region_id
        annot_vol_2 = np.isin(annot_vol_2, descendants).astype(int) * region_id

    return iou_score(annot_vol_1, annot_vol_2, k=region_id, disable_check=True)


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
        score, _ = iou_score(annot_vol_1, annot_vol_2, k=id_, disable_check=True)
        scores[id_] = score

    return scores


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


def jaggedness_along_tree(
    region_ids: Collection[int],
    atlas: np.ndarray,
    region_meta: RegionMeta,
) -> dict[int, float]:
    """Compute Jaggedness for each label ascendants of the region IDs.

    Parameters
    ----------
    region_ids
        Region IDs to evaluate.
    atlas
        Atlas to evaluate.
    region_meta
        Region Meta containing all the information concerning the labels.

    Returns
    -------
    results: dict[str, Any]
        Dictionary containing the results of the jaggedness.
    """
    results = {}

    for region_id in region_ids:
        desc = region_meta.descendants(region_id)
        ids_per_level = defaultdict(list)
        for d in desc:
            ids_per_level[region_meta.level[d]].append(d)

        values_from = np.unique(atlas)
        values_to = np.zeros_like(values_from)

        for _, children in ids_per_level.items():
            for child in children:
                desc = region_meta.descendants(child)
                values_to = [
                    child if value_from in desc else value_to
                    for value_from, value_to in zip(values_from, values_to)
                ]

            new_atlas = atlas_remap(atlas, values_from, np.array(values_to))
            regions_to_consider = list(np.unique(new_atlas[new_atlas != 0]))
            results.update(jaggedness(new_atlas, region_ids=regions_to_consider))

    return results


def iou_along_tree(
    region_ids: Collection[int],
    atlas: np.ndarray,
    reference: np.ndarray,
    region_meta: RegionMeta,
) -> dict[int, float]:
    """Compute IoU for each label ascendants of the region IDs.

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
        Dictionary containing the results of the IoU.
    """
    results = {}

    for region_id in region_ids:
        desc = region_meta.descendants(region_id)
        ids_per_level = defaultdict(list)
        for d in desc:
            ids_per_level[region_meta.level[d]].append(d)

        values_from = np.unique(atlas)
        values_to = np.zeros_like(values_from)

        for _, children in ids_per_level.items():
            for child in children:
                desc = region_meta.descendants(child)
                values_to = [
                    child if value_from in desc else value_to
                    for value_from, value_to in zip(values_from, values_to)
                ]

            new_atlas = atlas_remap(atlas, values_from, np.array(values_to))
            new_reference = atlas_remap(reference, values_from, np.array(values_to))
            regions_to_consider = list(np.unique(new_atlas[new_atlas != 0]))
            results.update(
                iou(new_reference, new_atlas, region_ids=regions_to_consider)
            )

    return results


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
    global_jaggedness = jaggedness(mask, region_ids=[1])[1]
    per_region_jaggedness = jaggedness_along_tree(region_ids, atlas, region_meta)

    results["jaggedness"] = {
        "global": global_jaggedness,
        "per_region": per_region_jaggedness,
    }

    # Intersection Over Union
    mask_ref = np.isin(reference, desc)
    global_iou = iou(mask_ref, mask, region_ids=[1])[1]
    per_region_iou = iou_along_tree(
        region_ids,
        atlas,
        reference,
        region_meta,
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
