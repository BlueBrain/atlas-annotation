#!/usr/bin/env python
# Copyright 2021, Blue Brain Project, EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""3D Avg Brain/Nissl registration (with atlas superposition) script."""
import logging
import sys

import numpy as np
import utils

import atlannot.utils
from atlannot.ants import register, transform
from atlannot.utils import load_volume

# Parameters
description = """\
The usual 2D ANTsPy registration with
fixed = v3 average brain template
moving = v2 nissl
plus the outline of the brain regions imprinted on the stain volumes
"""
experiment_name = utils.get_script_file_name()
v2_atlas_path = utils.get_v2_atlas_fine_path()
v3_atlas_path = utils.get_v3_atlas_fine_path()
nissl_path = utils.get_nissl_path()
avg_brain_path = utils.get_avg_brain_path()


# Initialize the logger
logger = logging.getLogger(experiment_name)


def main():
    """3D Avg Brain/Nissl registration (with atlas superposition) ."""
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
    average_brain = load_volume(avg_brain_path)

    # Preprocess data
    logger.info("Preprocessing data")
    nissl_volume_augmented = augment_volume(nissl_volume, v2_atlas)
    average_brain_augmented = augment_volume(average_brain, v3_atlas)

    # Registration
    logger.info("Starting registration")
    df_3d = register(fixed=average_brain_augmented, moving=nissl_volume_augmented)

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
        fp.write(str(v3_atlas_path) + "\n")
    with open(output_dir / "moving_path.txt", "w") as fp:
        fp.write(str(v2_atlas_path) + "\n")
    with open(output_dir / "nissl_path.txt", "w") as fp:
        fp.write(str(nissl_path) + "\n")
    with open(output_dir / "avg_brain_path.txt", "w") as fp:
        fp.write(str(avg_brain_path) + "\n")
    # volumes
    np.save(output_dir / "warped_atlas", warped_atlas)
    np.save(output_dir / "warped_nissl", warped_nissl)
    np.save(output_dir / "df", df_3d)
    logger.info(f"Finished. The results were saved to {output_dir}")


def augment_volume(volume, atlas):
    """Augment volume and atlas.

    Parameters
    ----------
    volume : np.ndarray
        Volume to preprocess.
    atlas : np.ndarray
        Atlas to preprocess.

    Returns
    -------
    enhanced_volume : np.ndarray
        Resulting volumes.
    """
    augmented_slices = []
    for volume_slice, atlas_slice in zip(volume, atlas):
        edge = atlannot.utils.edge_laplacian_thick(atlas_slice)
        augmented_slice = atlannot.utils.merge(volume_slice, edge)
        augmented_slices.append(augmented_slice)
    enhanced_volume = np.stack(augmented_slices)

    return enhanced_volume


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
