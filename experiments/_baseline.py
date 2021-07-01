#!/usr/bin/env python
"""Baseline script."""
import logging
import sys

import numpy as np
import utils

from atlannot import load_volume

# Parameters
description = """\
The baseline without any warping = zero displacement field
"""
experiment_name = utils.get_script_file_name()
v2_atlas_path = utils.get_v2_atlas_fine_path()
nissl_path = utils.get_nissl_path()


# Initialize the logger
logger = logging.getLogger(experiment_name)


def main():
    """Baseline."""
    # Paths
    output_dir = utils.get_results_dir() / experiment_name
    if not utils.can_write_to_dir(output_dir):
        print("Cannot write to output directory. Stopping")
        return 1

    # Load data
    logger.info("Loading data")
    v2_atlas = load_volume(v2_atlas_path, normalize=False)
    nissl_volume = load_volume(nissl_path)

    # Registration
    logger.info("Starting (fake) registration")
    df_3d_shape = nissl_volume.shape + (1, 3)
    df_3d = np.zeros(shape=df_3d_shape)

    # Write output
    logger.info("Saving results")
    # metadata
    with open(output_dir / "description.txt", "w") as fp:
        fp.write(description)
    with open(output_dir / "moving_path.txt", "w") as fp:
        fp.write(str(v2_atlas_path) + "\n")
    with open(output_dir / "nissl_path.txt", "w") as fp:
        fp.write(str(nissl_path) + "\n")
    # volumes
    np.save(output_dir / "warped_atlas", v2_atlas)
    np.save(output_dir / "warped_nissl", nissl_volume)
    np.save(output_dir / "df", df_3d)
    logger.info("Finished")
    logger.info(f"The results were saved to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
