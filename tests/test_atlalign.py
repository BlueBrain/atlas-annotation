import numpy as np
import pytest


from atlannot._atlalign import iou_score, iou_score_single


class TestIOUScoreSingle:
    def test_perfect_score(self):
        y_true = np.array(
            [
                [2, 5, 4],
                [2, 2, 1],
                [3, 7, 2],
            ],
            dtype=np.int8,
        )
        y_pred = y_true

        # fixed k
        assert np.isnan(iou_score_single(y_true, y_pred, 0))  # 0 not present
        assert iou_score_single(y_true, y_pred, 1) == 1
        assert iou_score_single(y_true, y_pred, 2) == 1
        assert iou_score_single(y_true, y_pred, 3) == 1
        assert iou_score_single(y_true, y_pred, 4) == 1
        assert iou_score_single(y_true, y_pred, 5) == 1
        assert np.isnan(iou_score_single(y_true, y_pred, 6))  # 6 not present
        assert iou_score_single(y_true, y_pred, 7) == 1

        # average over all k's
        assert iou_score_single(y_true, y_pred) == pytest.approx(1, rel=0, abs=1e-10)

    def test_worst_score(self):
        y_true = np.array(
            [
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 1],
            ],
            dtype=np.int8,
        )
        y_pred = 1 - y_true

        # fixed k
        assert iou_score_single(y_true, y_pred, 0) == 0
        assert iou_score_single(y_true, y_pred, 1) == 0
        assert np.isnan(iou_score_single(y_true, y_pred, 2))  # 2 not present

        # average over all k's
        assert iou_score_single(y_true, y_pred) == pytest.approx(0, rel=0, abs=1e-10)

    def test_symmetric(self):
        y_true = np.array(
            [
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 1],
            ],
            dtype=np.int8,
        )
        y_pred = np.array(
            [
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.int8,
        )

        # fixed k
        assert iou_score_single(y_true, y_pred, 0) == iou_score_single(
            y_pred, y_true, 0
        )
        assert iou_score_single(y_true, y_pred, 1) == iou_score_single(
            y_pred, y_true, 1
        )

        # average over all k's - not symmetric since 0's and 1's have different
        # counts
        assert iou_score_single(y_true, y_pred) != iou_score_single(y_pred, y_true)


class TestIOUScore:
    pass
