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
from __future__ import annotations

import numpy as np


def iou_score_single(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int | None = None,
    excluded_labels: list[int] | None = None,
) -> float:
    """Compute intersection over union of a class `k`.

    Parameters
    ----------
    y_true
        A np.ndarray of shape `(h, w)` representing the ground truth
        annotation.

    y_pred
        A np.ndarray of shape `(h, w)` representing the predicted annotation.

    k
        A class label. If None, then averaging based on label distribution
        in each true image is performed.

    excluded_labels
        If None then no effect. If a list of ints then the provided labels
        won't be used in the averaging over labels (in case
        `k` is None).

    Returns
    -------
    float
        The IOU score
    """
    if k is not None:
        mask_true = y_true == k
        mask_pred = y_pred == k

        intersection = np.logical_and(mask_true, mask_pred)
        union = np.logical_or(mask_true, mask_pred)

        res = (intersection.sum() / union.sum()) if not np.all(union == 0) else np.nan

    else:
        excluded_labels = excluded_labels or []
        n_pixels = (~np.isin(y_true, excluded_labels)).sum()

        # true image distribution
        labels = (set(np.unique(y_true)) | set(np.unique(y_pred))) - set(
            excluded_labels
        )
        w = {label: (y_true == label).sum() / n_pixels for label in labels}

        weighted_average = sum(
            iou_score_single(
                y_true,
                y_pred,
                k,
            )
            * p
            for k, p in w.items()
        )

        res = weighted_average

    return res


def iou_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int | None = None,
    excluded_labels: list[int] | None = None,
) -> tuple[float, np.ndarray]:
    """Compute IOU score for multiple samples.

    Parameters
    ----------
    y_true
        A np.ndarray of shape `(N, h, w)` representing the ground truth
        annotation.

    y_pred
        A np.ndarray of shape `(N, h, w)` representing the predicted annotation.

    k
        A class label. If None, then averaging based on label distribution
        in each true image is performed.

    excluded_labels
        If None then no effect. If a list of ints then the provided labels
        won't be used in the averaging over labels (in case
        `k` is None).

    Returns
    -------
    iou_average : float
        Average IOU score

    iou_per_sample : np.ndarray
        Per sample IOU scores.

    """
    n = len(y_true)
    per_sample = [
        iou_score_single(y_true[i], y_pred[i], k, excluded_labels) for i in range(n)
    ]

    per_sample_array = np.array(per_sample)
    mean = np.nanmean(per_sample_array)

    return mean, per_sample_array
