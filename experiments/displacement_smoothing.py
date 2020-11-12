"""Goal: Smoothing displacement field along the sagittal axis.

Assumptions:
- Reference and Moving input have to be volumes.
- The input volumes have the same shape (n_slices, height, width).
- A displacement field path has to be specified. (from 2d registration).
It needs to have (n_slices, 2, height, width) as dimension.

Steps:
- Loading input volumes.
- Definition of the DF: a path to a displacement field needs to specified.
- Extraction of the deltas from the displacement fields.
- Apply the averaging on the deltas along the sagittal axis (between 3 slices:
  the one before, the current one and the one after). For further information,
  you can check scipy documentation.
- Transform the resulting deltas back to displacement fields.
- Apply the averaged displacement fields to moving images to produce registered
  images.
- Saving the results.
"""
import argparse
import logging
import pathlib

import nrrd
import numpy as np
from scipy import ndimage

from deal import deltas_to_dfs, safe_dfs

logger = logging.getLogger("Displacement Smoothing")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--img-mov",
    default="data/atlasVolume.npy",
    type=str,
    help="The annotation image/volume to warp.",
)
parser.add_argument(
    "--df",
    default="experiments/results/registration_machine_learning_df.npy",
    type=str,
    help="The name of the file with the displacement field.",
)
args = parser.parse_args()


def main():
    """Smooths displacement field along sagittal axis."""
    logger.info("Loading Images...")
    path_mov = pathlib.Path(args.img_mov)
    if path_mov.suffix == ".nrrd":
        img_mov, _ = nrrd.read(path_mov)
    else:
        img_mov = np.load(path_mov)

    logger.info(f"Moving image: {args.img_mov}")
    logger.info(f"Shape of the moving image: {img_mov.shape}")

    logger.info("Normalizing images and change type to float 32...")
    img_mov = (img_mov / img_mov.max()).astype("float32")

    logger.info("Loading displacement field...")
    dfs = np.load(pathlib.Path(args.df))
    logger.info(f"Shape of the DF: {dfs.shape}")

    logger.info(
        "Extract the deltas (=transformation vector fields) for the registration"
    )
    deltas = np.stack([[df[:, :, 1], df[:, :, 0]] for df in dfs])

    logger.info("Apply the averaging on the deltas along the sagittal axis")
    deltas_uniform = ndimage.uniform_filter1d(deltas, 3, axis=0)
    deltas_gaussian = ndimage.gaussian_filter1d(deltas, 1, axis=0)
    deltas_correlate = ndimage.correlate1d(deltas, [1 / 4, 1 / 2, 1 / 4], axis=0)

    logger.info("Transform the resulting deltas back to displacement fields")
    dfs_uniform = deltas_to_dfs(deltas_uniform)
    dfs_gaussian = deltas_to_dfs(deltas_gaussian)
    dfs_correlate = deltas_to_dfs(deltas_correlate)

    logger.info(
        "Apply the averaged displacement fields to moving images to produce registered "
        "images"
    )
    img_uniform = np.stack([df.warp(img) for df, img in zip(dfs_uniform, img_mov)])
    img_gaussian = np.stack([df.warp(img) for df, img in zip(dfs_gaussian, img_mov)])
    img_correlate = np.stack([df.warp(img) for df, img in zip(dfs_correlate, img_mov)])

    logger.info("Saving registered images")
    out_path = pathlib.Path("experiments/results/")
    if not out_path.exists():
        pathlib.Path.mkdir(out_path, parents=True)
    np.save(out_path / f"{path_mov.stem}_uniform_smoothing.npy", img_uniform)
    np.save(out_path / f"{path_mov.stem}_gaussian_smoothing.npy", img_gaussian)
    np.save(out_path / f"{path_mov.stem}_correlate_smoothing.npy", img_correlate)
    safe_dfs(out_path / f"{path_mov.stem}_uniform_smoothing_df.npy", dfs_uniform)
    safe_dfs(out_path / f"{path_mov.stem}_gaussian_smoothing_df.npy", dfs_gaussian)
    safe_dfs(out_path / f"{path_mov.stem}_correlate_smoothing_df.npy", dfs_correlate)
    logger.info("DONE")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s || %(levelname)s || %(name)s || %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
