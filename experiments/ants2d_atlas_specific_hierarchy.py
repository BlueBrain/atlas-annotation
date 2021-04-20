#!/usr/bin/env python
"""2D fine atlases registration (at specific hierarchy) script."""
import logging
import sys

import numpy as np
import utils

from deal import load_volume
from deal.ants import register, stack_2d_transforms, transform
from deal.atlas import unfurl_regions
from deal.utils import load_region_meta

# Parameters
description = """\
2D ANTsPy registration with atlases (at a chosen hierarchy level):
fixed = v3 atlas
moving = v2 atlas
"""
experiment_name = utils.get_script_file_name()
v2_atlas_path = utils.get_v2_atlas_fine_path()
v3_atlas_path = utils.get_v3_atlas_fine_path()
nissl_path = utils.get_nissl_path()
brain_region_path = utils.get_data_dir() / "brain_regions.json"
hierarchy = 6


# Initialize the logger
logger = logging.getLogger(experiment_name)

script_info = """
Goal: Computing registration at an intermediate level of hierarchy.

Assumptions:
- Input images/volumes have to have the same shape.
- Input images/volumes have to be label images with a hierarchy for the labels.
- This hierarchy has te be specified by a json file (brain_regions).
- The hierarchy can be defined thanks to a number. This ones has to be
  between 0 and the number of level of hierarchy. The hierarchy 0 is the
  most detailed one, the last level should be the distinction
  foreground/background of the images.

Steps:
- Loading input images/volumes.
- Reading brain hierarchy and parsing it.
- Creating volumes for every level of the hierarchy by unfurling the different
  regions of the brain.
- Computation of ANTsPY registration between images/volumes at the chosen
  level of hierarchy
- Applying transformation to the input volumes at the highest level of the hierarchy.
- Saving the results.
- Computation and printing the baseline misalignment (between input
  images/volumes) and the results misalignment (between input reference and
  warping moving) at every level of the hierarchy.
"""


def main():
    """2D fine atlases registration (at specific hierarchy)."""
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
    region_meta = load_region_meta(brain_region_path)

    # Preprocess data
    logger.info("Preprocessing data")
    v2_atlas_pre, v3_atlas_pre = preprocess_atlases(
        v2_atlas, v3_atlas, region_meta=region_meta, hierarchy_level=hierarchy
    )

    # Registration
    logger.info("Starting registration")
    dfs = []
    for v3_slice, v2_slice in zip(v2_atlas_pre, v3_atlas_pre):
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


def preprocess_atlases(*atlases, region_meta=None, hierarchy_level=10):
    """Preprocess atlases.

    Parameters
    ----------
    atlases : Iterable of np.ndarray
        All atlases to preprocess.
    region_meta : deal.RegionMeta
        Object containing brain regions hierarchy.
    hierarchy_level: int
        Hierarchy level to keep for the registration

    Returns
    -------
    new_atlases : Iterable of np.ndarray
        Preprocessed atlases
    """
    atlases_pre = []
    for atlas in atlases:
        atlas_pre = unfurl_regions(atlas, region_meta)
        atlases_pre.append(atlas_pre[hierarchy_level])
    return [atlas.astype(np.float32) for atlas in atlases_pre]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
