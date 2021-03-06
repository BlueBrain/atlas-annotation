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
"""2D Avg Brain/Nissl registration (with middle line) script."""
import logging
import sys

import numpy as np
import utils

from atlannot.ants import register, stack_2d_transforms, transform
from atlannot.utils import add_middle_line, load_volume

# Parameters
description = """\
2D ANTsPy registration with Nissl/Avg Brain (with middle line):
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
Goal: Computing ANTsPY registration after creating a middle line on input images.

Assumptions:
- The input images/volumes have to have the same shape.

Steps:
- Loading input images/volumes.
- For each 2D images, creation of a middle bar and registration of the resulting images.
- Applying resulting transformations to the original moving image.
- Saving results.
"""


def main():
    """2D Avg Brain/Nissl registration (with middle line)."""
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
    avg_volume_pre, nissl_volume_pre = preprocess_volumes(
        avg_volume, nissl_volume, thickness=thickness
    )

    # Registration
    logger.info("Starting registration")
    dfs = []
    for avg_slice, nissl_slice in zip(avg_volume_pre, nissl_volume_pre):
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


def preprocess_volumes(*volumes, thickness=0):
    """Preprocess volumes.

    Parameters
    ----------
    volumes : Iterable of np.ndarray
        All volumes to preprocess.
    thickness : int
        Thickness of the middle line.

    Returns
    -------
    new_volumes : Iterable of np.ndarray
        Preprocessed volumes.
    """
    volumes_pre = []
    for volume in volumes:
        volume_pre = add_middle_line(volume, axis=2, thickness=thickness, value=1)
        volumes_pre.append(volume_pre)
    return [volume.astype(np.float32) for volume in volumes_pre]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
