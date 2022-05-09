from __future__ import annotations


import numpy as np


def iou_score_single(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int | None = None,
    excluded_labels: list[int] | None = None,
) -> float:
    """
    Compute intersection over union of a class `k`.


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
        result
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
    k: int | None = 0,
    excluded_labels: list[int] | None = None,
):
    pass
