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
"""2D coarse atlases registration script."""
import logging
import sys

import numpy as np
import utils
from tqdm import tqdm

from atlannot import load_volume
from atlannot.ants import register, stack_2d_transforms, transform

# Parameters
description = """\
2D ANTsPy registration with atlases:
fixed = v3 atlas, coarsely merged version
moving = v2 atlas, coarsely merged version
"""
experiment_name = utils.get_script_file_name()
v2_atlas_path = utils.get_v2_atlas_coarse_path()
v3_atlas_path = utils.get_v3_atlas_coarse_path()
nissl_path = utils.get_nissl_path()


# Initialize the logger
logger = logging.getLogger(experiment_name)


def main():
    """2D coarse atlases registration."""
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
    dfs = []
    total = len(v2_atlas_pre)
    for v3_slice, v2_slice in tqdm(zip(v2_atlas_pre, v3_atlas_pre), total=total):
        df = register(fixed=v3_slice, moving=v2_slice)
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
        fp.write(str(v3_atlas_path) + "\n")
    with open(output_dir / "moving_path.txt", "w") as fp:
        fp.write(str(v2_atlas_path) + "\n")
    with open(output_dir / "nissl_path.txt", "w") as fp:
        fp.write(str(nissl_path) + "\n")
    # volumes
    np.save(output_dir / "warped_atlas", warped_atlas)
    np.save(output_dir / "warped_nissl", warped_nissl)
    np.save(output_dir / "df", df_3d)
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
    sys.exit(main())
