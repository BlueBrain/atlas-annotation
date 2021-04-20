#!/usr/bin/env python
"""3D atlases registration (after remapping labels) script."""
import logging
import sys

import numpy as np
import utils

from deal import load_volume
from deal.ants import register, transform
from deal.utils import remap_labels

# Parameters
description = """\
3D ANTsPy registration with atlases (after remapping labels values):
fixed = v3 atlas
moving = v2 atlas
"""
experiment_name = utils.get_script_file_name()
v2_atlas_path = utils.get_v2_atlas_fine_path()
v3_atlas_path = utils.get_v3_atlas_fine_path()
nissl_path = utils.get_nissl_path()
seed = 2  # (can also be None)


# Initialize the logger
logger = logging.getLogger(experiment_name)

script_info = """
Goal: Computing the registration between two images/volumes after switching
randomly the labels.

Assumptions:
- The input images/volumes have the same shape.
- The input images/volumes are considered as label images.
- The registration is computed on the entire input images at once. Which means
  that if volumes are specified, the registration is a 3D registration. If 2D
  images are specified, this is a 2D registration.

Steps:
- Loading of the images
- Creation of union list containing all the labels appearing at least in one
  of the two input images/volumes.
- The conversion previous labels/new labels is done by taking as new label
  the position in the list of the previous label. For example:
  Union List: [0, 1002, 6, 9]
  New labels: [0, 1, 2, 3]
  Which means 0 stays 0 in the new volume, 1002 is becoming 1, 6 is
  becoming 2, ... Obviously, there are other strategies to convert previous
  labels to new ones.
- Creation of new images/volumes with corresponding new labels.
- Computation of the ANTsPY registration on the new images/volumes.
- Applying transform found in the previous step at the initial images/volumes.
- Computation of baseline misalignement (between inputs) and the results
  misalignment (between input reference and warped moving image).
"""


def main():
    """3D atlases registration (after remapping labels)."""
    # Paths
    output_dir = utils.get_results_dir() / experiment_name
    if not utils.can_write_to_dir(output_dir):
        print("Cannot write to output directory. Stopping")
        return 1

    # Load data
    logger.info("Loading data")
    v3_atlas = load_volume(v3_atlas_path, normalize=False)
    v2_atlas = load_volume(v2_atlas_path, normalize=False)
    nissl_volume = load_volume(nissl_path)

    # Preprocess data
    logger.info("Preprocessing data")
    v3_atlas_pre, v2_atlas_pre = preprocess_atlases(
        v3_atlas,
        v2_atlas,
    )

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
        fp.write(str(v2_atlas_path) + "\n")
    with open(output_dir / "moving_path.txt", "w") as fp:
        fp.write(str(v3_atlas_path) + "\n")
    with open(output_dir / "nissl_path.txt", "w") as fp:
        fp.write(str(nissl_path) + "\n")
    # volumes
    np.save(output_dir / "warped_atlas", warped_atlas)
    np.save(output_dir / "warped_nissl", warped_nissl)
    np.save(output_dir / "df", df)
    logger.info(f"Finished. The results were saved to {output_dir}")


def preprocess_atlases(*atlases, seed=None):
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
    atlases_pre = remap_labels(atlases, seed=seed)
    return [atlas.astype(np.float32) for atlas in atlases_pre]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
