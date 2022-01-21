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
import numpy as np
from atlalign.metrics import iou_score
from atlas_alignment_meter import core
from scipy.stats import entropy

from atlannot.merge.common import atlas_remap

REGIONS_TO_EVALUATE = {
    "Somatosensory Cortex": [453],
    "Visual Cortex": [669],
    "Rest of Cortex": [315],
    "Thalamus": [549],
    "Hippocampus": [1080],
    "Cerebullum": [512],
    "Basal Ganglia": [477, 1022, 470, 381],
}


def compute_jaggedness(volume, region_ids=None, axis=0):
    """Compute the jaggedness of given region IDs for the specified volume.

    Parameters
    ----------
    volume: np.ndarray
        Input volume.
    region_ids: list[int] | None
        List of region IDs to compute the jaggedness. If None, the jaggedness
        is computed for all the region IDs present in the volume.
    axis: int
        Axis along which to compute the jaggedness.

    Returns
    -------
    results: dict[int, float]
        Dictionary containing the region id as keys and the mean of the
        jaggedness of that given region id as values.
    """
    metrics = core.compute(volume, coronal_axis_index=axis, regions=region_ids)
    if region_ids is None:
        return {
            region_id: values["mean"]
            for region_id, values in metrics["perRegion"].items()
        }

    results = {}
    for region_id in region_ids:
        results[region_id] = metrics["perRegion"][region_id]["mean"]
    return results


def compute_iou(vol_true, vol_pred, region_ids=None):
    """Compute the intersection over union of given region IDs.

    Parameters
    ----------
    vol_true: np.ndarray
        Input volume considered as ground truth.
    vol_pred: np.ndarray
        Input volume considered as prediction.
    region_ids: list[int] | None
        List of region IDs to compute the intersection over union.
        If None, the IoU is computed for all the region IDs present in the volume.

    Returns
    -------
    results: dict[int, float]
        Dictionary containing the region id as keys and the average of the
        intersection over union of that given region id as values.
    """
    results = {}
    if region_ids is None:
        labels = np.unique(vol_true)
        for label in labels:
            results[label] = iou_score(vol_true, vol_pred, k=label)[0]

    for region_id in region_ids:
        results[region_id] = iou_score(vol_true, vol_pred, k=region_id)[0]

    return results


def compute_region_entropy(nissl, atlas, label_value, bins=100):
    """Compute the entropy of densities of Nissl for a given label value in the atlas.

    Parameters
    ----------
    nissl: np.ndarray
        Nissl volume.
    atlas: np.ndarray
        Annotation atlas.
    label_value: int
        Label Value to consider.
    bins: int
        Number of bins used to compute the histogram of the densities.

    Returns
    -------
    region_entropy: float
        Entropy of the densities of Nissl at a given region value.
    """
    prob_region, _ = np.histogram(nissl[atlas == label_value], bins=bins)
    region_entropy = entropy(prob_region)
    return region_entropy


def compute_entropies(nissl, atlas):
    """Compute entropies of Nissl densities.

    Parameters
    ----------
    nissl: np.ndarray
        Nissl volume.
    atlas: np.ndarray
        Annotation atlas.

    Returns
    -------
    brain_entropy: float
        Entropy of the densities of Nissl for the entire brain.

    conditional_entropy: float
        Conditional entropy of the densities of Nissl depending on the brain regions.
    """
    # 1: For the entire brain
    prob_entire_brain, _ = np.histogram(nissl[atlas != 0], bins=100)
    brain_entropy = entropy(prob_entire_brain)

    # 2. Computation of the conditional entropy
    n_pixels = (atlas != 0).sum()
    label_values, count_values = np.unique(atlas, return_counts=True)
    all_region_entropy = []
    for label, count in zip(label_values, count_values):
        all_region_entropy.append(compute_region_entropy(nissl, atlas, label) * count)

    conditional_entropy = np.sum(all_region_entropy) / n_pixels
    return brain_entropy, conditional_entropy


def evaluate(atlas, nissl, reference, region_meta):
    """Evaluate the atlas.

    Parameters
    ----------
    atlas: np.ndarray
        Atlas to evaluate.
    nissl: np.ndarray
        Corresponding Nissl Volume.
    reference: np.ndarray
        Reference atlas.
    region_meta: atlannot.atlas.RegionMeta
        Region Meta containing all the information concerning the labels.

    Returns
    -------
    results: dict[str, Any]
        Dictionary containing the results of the evaluation.
    """
    results = {}
    for name, region_ids in REGIONS_TO_EVALUATE.items():
        desc = list(region_meta.descendants(region_ids))

        # Put some metadata about the region
        results[name] = {
            "region_ids": region_ids,
            "level": [region_meta.level[id_] for id_ in region_ids],
            "descendants": desc,
        }

        # Jaggedness
        values_from = np.unique(atlas)
        values_to = [1 if value in desc else 0 for value in values_from]
        mask = atlas_remap(atlas, values_from, values_to)
        global_jaggedness = compute_jaggedness(mask, region_ids=[1])[1]
        per_region_jaggedness = compute_jaggedness(mask, region_ids=desc)

        results[name]["jaggedness"] = {
            "global": global_jaggedness,
            "per_region": per_region_jaggedness,
        }

        # Intersection Over Union
        mask_ref = atlas_remap(reference, values_from, values_to)
        global_iou = compute_iou(mask_ref, mask, region_ids=[1])[1]
        per_region_iou = compute_iou(reference, atlas, region_ids=desc)

        results[name]["iou"] = {
            "global": global_iou,
            "per_region": per_region_iou,
        }

    # Entropies
    brain_entropy, conditional_entropy = compute_entropies(nissl, atlas)
    results["global"] = {
        "brain_entropy": brain_entropy,
        "conditional_entropy": conditional_entropy,
    }

    return results
