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
from scipy import stats

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
        region_ids = sorted(metrics["perRegion"].keys())

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
        region_ids = np.unique(vol_true)

    for region_id in region_ids:
        results[region_id] = iou_score(vol_true, vol_pred, k=region_id)[0]

    return results


def dist_entropy(data, bins=100):
    """Compute the entropy of the value distribution of data.

    Parameters
    ----------
    data: np.ndarray
        Data for which to compute the entropy.
    bins: int
        Number of bins used to compute the histogram of the densities.

    Returns
    -------
    float
        Entropy of the densities of Nissl at a given region value.
    """
    hist, _ = np.histogram(data, bins=bins)
    return stats.entropy(hist)


def compute_conditional_entropy(nissl, atlas):
    """Compute entropies of Nissl densities.

    Parameters
    ----------
    nissl: np.ndarray
        Nissl volume.
    atlas: np.ndarray
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


def evaluate(
    atlas,
    nissl,
    reference,
    region_meta,
    regions_to_evaluate=None,
):
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
    regions_to_evaluate: dict[str, list[int]] | None
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
        desc = list(region_meta.descendants(region_ids))

        # Put some metadata about the region
        results[name] = {
            "region_ids": region_ids,
            "level": [region_meta.level[id_] for id_ in region_ids],
            "descendants": desc,
        }

        # Jaggedness
        mask = np.isin(atlas, desc)
        global_jaggedness = compute_jaggedness(mask, region_ids=[1])[1]
        per_region_jaggedness = compute_jaggedness(atlas, region_ids=desc)

        results[name]["jaggedness"] = {
            "global": global_jaggedness,
            "per_region": per_region_jaggedness,
        }

        # Intersection Over Union
        mask_ref = np.isin(reference, desc)
        global_iou = compute_iou(mask_ref, mask, region_ids=[1])[1]
        per_region_iou = compute_iou(reference, atlas, region_ids=desc)

        results[name]["iou"] = {
            "global": global_iou,
            "per_region": per_region_iou,
        }

    # Entropies
    brain_entropy = dist_entropy(nissl[atlas != 0])
    conditional_entropy = compute_conditional_entropy(nissl, atlas)
    results["global"] = {
        "brain_entropy": brain_entropy,
        "conditional_entropy": conditional_entropy,
    }

    return results
