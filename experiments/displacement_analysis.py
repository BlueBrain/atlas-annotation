"""Goal: Analyze a specific displacement field.

Assumptions:
- The only input needed is a displacement field.
- The axis dimension has to be the last one, for example:
  [n_slices, height, width, 2] or [height, width, 2] or [n_slices, height, width, 3]

Steps:
- Loading the displacement field specified.
- If a moving image/volume is specified, this one is also loaded.
- Creation of a figure with an histogram of every axis of the displacement field.
- If a moving image/volume is specified, creation of a second figure with a
  histogram of every axis of the displacement field for the foreground part
  of the moving image/volume.
"""
import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import nrrd
import numpy as np

logger = logging.getLogger("Displacement Analysis")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--df",
    default="experiments/results/df_registration_changing_labels.npy",
    type=str,
    help="The displacement to analyze.",
)
parser.add_argument(
    "--img-mov",
    default="data/annotation_atlas/annotations_ccf_v2_merged.nrrd",
    type=str,
    help="The annotation image/volume to warp.",
)
args = parser.parse_args()


def main():
    """Analysis of a specific displacement field."""
    logger.info("Loading DF...")
    logger.info(f"Displacement field: {args.df}")
    path_df = pathlib.Path(args.df)
    df = np.load(path_df)
    df = np.squeeze(df)
    logger.info(f"Shape of DF: {df.shape}")

    if args.img_mov:
        logger.info("Loading Moving Image...")
        logger.info(f"Moving Image: {args.img_mov}")
        path_mov = pathlib.Path(args.img_mov)
        img_mov, _ = nrrd.read(path_mov)
        logger.info(f"Shape of moving image: {img_mov.shape}")

    logger.info("Analysis...")
    num_axis = df.shape[-1]
    fig, axs = plt.subplots(1, num_axis, figsize=(7 * num_axis, 5))
    for i in range(num_axis):
        axs[i].hist(df[:, :, :, i].flatten(), bins=50)
        axs[i].set_title(
            f"Histogram along {i} axis: min/max"
            f" {np.min(df[:, :, :, i].flatten()):6.2f}"
            f" /{np.max(df[:, :, :, i].flatten()):6.2f}"
        )

    logger.info("Saving Picture 1...")
    df_name = path_df.stem
    out_path = pathlib.Path("experiments/results/")
    if not out_path.exists():
        pathlib.Path.mkdir(out_path, parents=True)
    fig.savefig(out_path / f"{df_name}_analysis.png")

    if args.img_mov:
        fg = img_mov != 0
        fig2, axs = plt.subplots(1, num_axis, figsize=(7 * num_axis, 5))
        for i in range(num_axis):
            values = df[:, :, :, i][fg].flatten()
            axs[i].hist(values, bins=50)
            axs[i].set_title(
                f"Histogram along {i} axis (w/o background): min/max"
                f" {np.min(values):6.2f}"
                f" /{np.max(values):6.2f}"
            )
        logger.info("Saving Picture 2...")
        fig2.savefig(out_path / f"{df_name}_analysis2.png")

    logger.info("DONE")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s || %(levelname)s || %(name)s || %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
