#!/usr/bin/env python
"""2D Avg Brain/Nissl registration script."""
import logging
import sys

import numpy as np
import utils

from deal import load_volume
from deal.ants import register, stack_2d_transforms, transform

# Parameters
description = """\
2D ANTsPy registration with Avg Brain/Nissl:
fixed = Avg Brain
moving = Nissl
"""
experiment_name = utils.get_script_file_name()
v2_atlas_path = utils.get_v2_atlas_fine_path()
nissl_path = utils.get_nissl_path()
avg_path = utils.get_avg_brain_path()

# Initialize the logger
logger = logging.getLogger(experiment_name)


def main():
    """2D Avg Brain/Nissl registration."""
    # Paths
    output_dir = utils.get_results_dir() / experiment_name
    if not utils.can_write_to_dir(output_dir):
        print("Cannot write to output directory. Stopping")
        return 1

    # Load data
    logger.info("Loading data")
    avg_volume = load_volume(avg_path)
    nissl_volume = load_volume(nissl_path)
    v2_atlas = load_volume(v2_atlas_path, normalize=False)

    # Preprocess data
    logger.info("Preprocessing data")
    avg_pre, nissl_pre = preprocess_volumes(
        avg_volume,
        nissl_volume,
    )

    # Registration
    logger.info("Starting registration")
    dfs = []
    for avg_slice, nissl_slice in zip(avg_pre, nissl_pre):
        df = register(fixed=avg_slice, moving=nissl_slice)
        dfs.append(df)
    df_3d = stack_2d_transforms(dfs)

    # Warping
    logger.info("Warping volumes")
    warped_atlas = transform(
        v2_atlas.astype(np.float32),
        df_3d,
        interpolator="genericLabel",
    )
    warped_atlas = warped_atlas.astype(v2_atlas.dtype)
    warped_nissl = transform(nissl_volume, df_3d)

    # Write output
    logger.info("Saving results")
    # metadata
    with open(output_dir / "description.txt", "w") as fp:
        fp.write(description)
    with open(output_dir / "fixed_path.txt", "w") as fp:
        fp.write(str(avg_path) + "\n")
    with open(output_dir / "moving_path.txt", "w") as fp:
        fp.write(str(nissl_path) + "\n")
    with open(output_dir / "v2_atlas_path.txt", "w") as fp:
        fp.write(str(v2_atlas_path) + "\n")
    # volumes
    np.save(output_dir / "warped_atlas", warped_atlas)
    np.save(output_dir / "warped_nissl", warped_nissl)
    np.save(output_dir / "df", df_3d)
    logger.info(f"Finished. The results were saved to {output_dir}")


def preprocess_volumes(*volumes):
    """Preprocess volumes.

    Parameters
    ----------
    volumes : Iterable of np.ndarray
        All volumes to preprocess.

    Returns
    -------
    new_volumes : Iterable of np.ndarray
        Preprocessed volumes.
    """
    return [volume.astype(np.float32) for volume in volumes]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
