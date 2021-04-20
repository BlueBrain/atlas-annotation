#!/usr/bin/env python
"""3D fine atlases registration script."""
import logging
import sys

import numpy as np
import utils

from deal import load_volume
from deal.ants import register, transform

# Parameters
description = """\
3D ANTsPy registration with atlases:
fixed = v3 atlas
moving = v2 atlas
"""
experiment_name = utils.get_script_file_name()
v2_atlas_path = utils.get_v2_atlas_fine_path()
v3_atlas_path = utils.get_v3_atlas_fine_path()
nissl_path = utils.get_nissl_path()


# Initialize the logger
logger = logging.getLogger(experiment_name)


def main():
    """3D fine atlases registration."""
    # Paths
    output_dir = utils.get_results_dir() / experiment_name
    if not utils.can_write_to_dir(output_dir):
        print("Cannot write to output directory. Stopping")
        return 1

    # Load data
    logger.info("Loading data")
    v2_atlas = load_volume(v2_atlas_path, normalize=False)
    v3_atlas = load_volume(v3_atlas_path, normalize=False)
    nissl_volume = load_volume(nissl_path)

    # Preprocess data
    logger.info("Preprocessing data")
    v2_atlas_pre, v3_atlas_pre = preprocess_atlases(v2_atlas, v3_atlas)

    # Registration
    logger.info("Starting registration")
    df = register(fixed=v3_atlas_pre, moving=v2_atlas_pre)

    # Warping
    logger.info("Warping volumes")
    warped_atlas = transform(
        v2_atlas.astype(np.float32),
        df,
        interpolator="genericLabel",
    )
    warped_atlas = warped_atlas.astype(v2_atlas.dtype)
    warped_nissl = transform(nissl_volume, df)

    # Write output
    logger.info("Saving results")
    # metadata
    with open(output_dir / "description.txt", "w") as fp:
        fp.write(description)
    with open(output_dir / "fixed_path.txt", "w") as fp:
        fp.write(str(v3_atlas_path) + "\n")
    with open(output_dir / "moving_path.txt", "w") as fp:
        fp.write(str(v2_atlas_path) + "\n")
    with open(output_dir / "nissl_path.txt", "w") as fp:
        fp.write(str(nissl_path) + "\n")
    # volumes
    np.save(output_dir / "warped_atlas", warped_atlas)
    np.save(output_dir / "warped_nissl", warped_nissl)
    np.save(output_dir / "df", df)
    logger.info(f"Finished. The results were saved to {output_dir}")


def preprocess_atlases(*atlases):
    """Preprocess atlases.

    Parameters
    ----------
    atlases : Iterable of np.ndarray
        All atlases to preprocess.

    Returns
    -------
    new_atlases : Iterable of np.ndarray
        Preprocessed atlases
    """
    return [atlas.astype(np.float32) for atlas in atlases]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
