from __future__ import annotations


import numpy as np

def iou_score_single(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int | None = None,
        excluded_labels: list[int] | None = None,
    ):
    """
    Compute intersection over union of a class `k`.
    """

    n_pixels = (
        ~np.isin(y_true, np.array(excluded_labels or []))
    ).sum()  # only non excluded pixels

    if k is not None:
        mask_true = y_true == k
        mask_pred = y_pred == k

        intersection = np.logical_and(mask_true, mask_pred)
        union = np.logical_or(mask_true, mask_pred)

        res = (
            (intersection.sum() / union.sum()) if not np.all(union == 0) else np.nan
        )

    else:
        # true image distribution
        w = {
            label: (y_true == label).sum() / n_pixels
            for label in (set(np.unique(y_true)) | set(np.unique(y_pred)))
            - set(excluded_labels or [])
        }

        weighted_average = sum(
            iou_score_single(
                y_true[np.newaxis, :, :],
                y_pred[np.newaxis, :, :],
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
