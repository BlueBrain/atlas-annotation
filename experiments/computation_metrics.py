"""Computation of different metrics.

Goal: Computing different metrics (image similarities and intersection over union).

The metrics are computed:
- between the reference and the moving images (as baseline)
- between the reference and the warped images (as results)
and saved in metrics.pickle

Regarding the image similarities metrics:
- The metrics are computed on the entire images and saved under `metricname`
- The metrics are computed on the foreground part of the image (mask computed
  thanks the reference images) and saved under `metricname_masked`.

An intersection-over-union is also computed. To that end, a segmentation map
foreground/background is constructed based on every input image and IOU is
computed on those segmentation maps. The results are saved in a dictionary
with structure:
{
metric_1: {
    'moving': [],
    'experiment_1': [],
    ....
},
metric_masked_1: {
    'moving': [],
    'experiment_1': [],
    ....
},
}

Assumptions:
- The input images/volumes should have the same shape.
- If a file `out_file` already exists, it is loaded and extended with the new values.

Steps:
- Loading input images.
- Loading metrics file if there is one.
- Computing image similarities for every slice on the entire images for the
  baseline and the results.
- Computing image similarities for every slice on the foreground images for
  the baseline and the results.
- Computing the IOU score for the foreground/background partition (check if
  the borders are aligned).
"""
import argparse
import logging
import pathlib
import pickle

import numpy as np
from tqdm import tqdm
from warpme import metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    "--img-ref",
    default="data/average_template_25.npy",
    type=str,
    help="The image/volume of reference. It is assuming the format of the "
    "reference image is .npy",
)
parser.add_argument(
    "--img-mov",
    default="data/atlasVolume.npy",
    type=str,
    help="The moving image/volume (=to warp). It is assuming the format of the "
    "moving image is .npy",
)
parser.add_argument(
    "--warped-img",
    default="experiments/results/registration_with_middle_line.npy",
    type=str,
    help="The warped image/volume name. It is assuming the format of the "
    "warped image is .npy.",
)
parser.add_argument(
    "--experiment-name",
    default="antspy_middle_bar",
    type=str,
    help="The name under what the metrics are going to be saved.",
)
parser.add_argument(
    "--out-file",
    default="experiments/results/metrics.pickle",
    type=str,
    help="The name of the file with all the metrics values. The algorithm is "
    "assuming that the file is lying under results folder "
    "and has .pickle as extension.",
)
args = parser.parse_args()

logger = logging.getLogger("Computation Metrics")


def iou_score(mask_true, mask_pred):
    """Compute the IOU score.

    Parameters
    ----------
    mask_true : np.ndarray
        The true mask, has to be of boolean type.
    mask_pred : np.ndarray
        The predicted mask, has to be of boolean type.

    Returns
    -------
    iou : float
        The IOU.
    """
    intersection = np.logical_and(mask_true, mask_pred)
    union = np.logical_or(mask_true, mask_pred)

    iou = (intersection.sum() / union.sum()) if not np.all(union == 0) else np.nan
    return iou


def main():
    """Compute the different metrics."""
    logger.info("Loading Images...")
    logger.info(f"Reference Image: {args.img_ref}")
    logger.info(f"Moving Image: {args.img_mov}")
    logger.info(f"Warped Image: {args.warped_img}")

    img_ref = np.load(pathlib.Path(args.img_ref)).astype(np.float32)
    img_mov = np.load(pathlib.Path(args.img_mov)).astype(np.float32)
    warped_img = np.load(pathlib.Path(args.warped_img)).astype(np.float32)

    logger.info(f"Shape of the reference image: {img_ref.shape}")
    logger.info(f"Shape of the moving image: {img_mov.shape}")
    logger.info(f"Shape of the warped image: {warped_img.shape}")

    logger.info("Normalizing pixel values between [0, 1]")
    img_ref = img_ref / img_ref.max()
    img_mov = img_mov / img_mov.max()
    warped_img = warped_img / warped_img.max()

    if len(img_ref.shape) == 2:
        img_ref = np.expand_dims(img_ref, axis=0)
        img_mov = np.expand_dims(img_mov, axis=0)
        warped_img = np.expand_dims(warped_img, axis=0)

    out_path = pathlib.Path(args.out_file)
    if out_path and out_path.exists():
        with open(out_path, "rb") as f:
            all_metrics = pickle.load(f)
    else:
        all_metrics = {}

    logger.info("Computation of the different metrics.")
    metric_fns = [
        "mse_img",
        "mae_img",
        "psnr_img",
        "demons_img",
        "cross_correlation_img",
        "ssmi_img",
        # 'mi_img', Seems to have an issue when running it
        "perceptual_loss_img",  # Notice that perceptual loss is taking a lot of time
    ]

    for metric_name in tqdm(metric_fns):
        logger.info(f"Computation of {metric_name}.")
        key, *_ = metric_name.rpartition("_")
        if key not in all_metrics.keys():
            all_metrics[key] = {}
        metric_fn = getattr(metrics, metric_name)
        all_metrics[key]["moving"] = metric_fn(img_ref, img_mov)
        all_metrics[key][args.experiment_name] = metric_fn(img_ref, warped_img)
        logger.info(
            f"Baseline {metric_name} mean: {np.mean(all_metrics[key]['moving'])}."
        )
        logger.info(
            f"Results {metric_name} mean: "
            f"{np.mean(all_metrics[key][args.experiment_name])}."
        )

    logger.info(
        "Computation of the different metrics after applying "
        "a mask defining the reference foreground."
    )
    # Remove the metrics that are not applicable to masked images.
    metric_fns.remove("ssmi_img")
    metric_fns.remove("perceptual_loss_img")
    # Computation of the mask on the reference image.
    threshold = 0.04
    mask_ref = img_ref > threshold

    for metric_name in tqdm(metric_fns):
        key, *_ = metric_name.rpartition("_")
        key = key + "_masked"
        logger.info(f"Computation of {key}.")
        if key not in all_metrics.keys():
            all_metrics[key] = {}
        metric_fn = getattr(metrics, metric_name)

        moving, values = [], []

        for i in range(img_ref.shape[0]):
            if np.sum(mask_ref[i]) == 0:
                moving.append(np.nan)
                values.append(np.nan)
            else:
                moving.append(metric_fn(img_ref[i], img_mov[i], mask=mask_ref[i]))
                values.append(metric_fn(img_ref[i], warped_img[i], mask=mask_ref[i]))

        all_metrics[key]["moving"] = moving
        all_metrics[key][args.experiment_name] = values
        logger.info(f"Baseline {key} mean: {np.nanmean(all_metrics[key]['moving'])}.")
        logger.info(
            f"Results {key} mean: {np.nanmean(all_metrics[key][args.experiment_name])}."
        )

    logger.info(
        "Computation of IOU scores after applying a mask "
        "defining the reference foreground."
    )

    iou_baseline, iou_results = [], []
    if "iou" not in all_metrics.keys():
        all_metrics["iou"] = {}

    for i in range(img_ref.shape[0]):
        iou_baseline.append(iou_score(img_ref[i] > threshold, img_mov[i] > threshold))
        iou_results.append(iou_score(img_ref[i] > threshold, warped_img[i] > threshold))
    all_metrics["iou"]["moving"] = iou_baseline
    all_metrics["iou"][args.experiment_name] = iou_results
    logger.info(f"Baseline IOU mean: {np.mean(all_metrics['iou']['moving'])}.")
    logger.info(
        f"Results IOU mean: {np.mean(all_metrics['iou'][args.experiment_name])}."
    )

    logger.info("Saving the results...")
    if not out_path.parent.is_dir():
        pathlib.Path.mkdir(out_path.parent, parents=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_metrics, f)

    logger.info("DONE")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s || %(levelname)s || %(name)s || %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
