#!/usr/bin/env python
"""Machine Learning Registration Nissl/Avg Brain."""
import logging
import sys

import numpy as np
import utils
from warpme.ml_utils import load_model, merge_global_local

from atlannot import load_volume
from atlannot.ants import stack_2d_transforms, transform

# Parameters
description = """\
Machine Learning registration with Nissl/Avg Brain:
fixed = Avg Brain
moving = Nissl
"""
experiment_name = utils.get_script_file_name()
v2_atlas_path = utils.get_v2_atlas_fine_path()
nissl_path = utils.get_nissl_path()
avg_path = utils.get_avg_brain_path()
thickness = 2


# Initialize the logger
logger = logging.getLogger(experiment_name)

script_info = """
Goal: computing machine learning 2D registration between the reference and
moving images/volumes.

Assumptions:
- The input images/volumes have to have the same shape.

Steps:
- Loading input images/volumes.
- For each 2D images, machine learning 2D registration.
- Applying resulting transformations to the original moving image.
- Saving results.
"""


def main():
    """Machine Learning Registration Nissl/Avg Brain."""
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
    ml_input = preprocess_volumes(
        avg_volume,
        nissl_volume,
    )

    logger.info("Loading ML models...")
    model_merged = machine_learning_model()

    # Registration
    logger.info("Starting registration")
    dfs = []
    for dim, x in enumerate(ml_input):
        img_reg, delta_xy = model_merged.predict(
            np.expand_dims(x, axis=0), batch_size=1
        )
        dfs.append(delta_xy)
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
    volumes_pre = []
    for volume in volumes:
        if len(volume.shape) == 2:
            volume = np.expand_dims(volume, axis=0)
        if len(volume.shape) == 3:
            volume = np.expand_dims(volume, axis=3)
        volumes_pre.append(volume)

    ml_input = np.concatenate(volumes_pre, axis=3)
    return ml_input


def machine_learning_model():
    """Load machine learning model.

    Returns
    -------
    model_merged:
        Machine Learning model.
    """
    model_global = load_model(
        "/gpfs/bbp.cscs.ch/project/proj101/"
        "pretrained_models/global/calm_camel/calm_camel.h5"
    )
    model_local = load_model(
        "/gpfs/bbp.cscs.ch/project/proj101/"
        "pretrained_models/local/cute_cat/cute_cat.h5"
    )
    model_merged = merge_global_local(model_global, model_local)

    return model_merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
