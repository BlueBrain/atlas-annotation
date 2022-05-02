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
"""Script that maps Nissl to CCFv3."""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from pathlib import Path

import numpy as np
from utils import get_results_dir

from atlannot import load_volume
from atlannot.ants import register, transform

# Initialize the logger
logger = logging.getLogger("nissl-to-ccfv3")
DATA_FOLDER = pathlib.Path(__file__).resolve().parent.parent / "data"


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nissl_path",
        type=Path,
        default=DATA_FOLDER / "ara_nissl_25.nrrd",
        help="""\
        Path to Nissl Volume.
        """,
    )
    parser.add_argument(
        "--ccfv2_path",
        type=Path,
        default=DATA_FOLDER / "ccfv2_merged.nrrd",
        help="""\
        Path to CCFv2 annotation volume.
        """,
    )
    parser.add_argument(
        "--ccfv3_path",
        type=Path,
        default=DATA_FOLDER / "ccfv3_merged.nrrd",
        help="""\
        Path to CCFv3 annotation volume.
        """,
    )
    return parser.parse_args()


def check_and_load(path: pathlib.Path | str) -> np.array:
    """Load volume if path exists.

    Parameters
    ----------
    path
        File path.

    Returns
    -------
    volume : np.array
        Loaded volume.
    """
    if not path.exists():
        logger.error(f"The specified path {path} does not exist.")
        return 1
    volume = load_volume(path, normalize=False)
    return volume.astype(np.float32)


def slice_registration(
    fixed: np.array, moving: np.array, nissl_slice: np.array | None = None
) -> tuple[np.array, np.array | None]:
    """Compute registration transform between a couple of slices.

    Parameters
    ----------
    fixed
        Fixed slice of type annotation.
    moving
        Moving slice ot type annotation.
    nissl_slice
        If needed, intensity slice in the same coordinate system as moving.

    Returns
    -------
    warped : np.array
        Warped slice of type annotation.
    nissl_slice : np.array | None
        If a nissl slice is specified, the given slice is warped with the
        same transformation.
    """
    nii_data = register(fixed, moving, is_atlas=True)
    warped = transform(moving, nii_data, interpolator="genericLabel")

    if nissl_slice is not None:
        nissl_slice = transform(nissl_slice, nii_data)

    return warped, nissl_slice


def registration(
    reference_volume: np.array, moving_volume: np.array, nissl_volume: np.array
) -> tuple[np.array, np.array]:
    """Compute registration between a moving volume and reference one.

    The registration is computed in two steps:
    1. First between one slice of the moving volume and the neighbouring slice
    (To reduce the jaggedness)
    2. Once the first registration is done, the transformation between the
    resulting slice and the corresponding slice of the reference volume is
    applied. (To make moving and reference more similar)
    The registration is computed from the center of the volume to the two
    extremities.

    Parameters
    ----------
    reference_volume
        Reference volume.
    moving_volume
        Moving volume.
    nissl_volume
        Nissl volume to register. It has to have same shape as reference_volume
        and moving_volume and be from the same coordinate system as the moving
        volume.

    Returns
    -------
    warped_volume
        Moving volume after the registration transformation.
    nissl_warped
        Nissl volume once the registration transformation are applied.
    """
    warped_volume = np.zeros_like(moving_volume)
    nissl_warped = np.zeros_like(nissl_volume)
    total_n_iterations = moving_volume.shape[0]
    n_iterations = int(total_n_iterations / 2)

    fixed = moving_volume[n_iterations, :, :]
    for i in range(n_iterations):  # 0 -> 263
        warped = moving_volume[n_iterations - (i + 1), :, :]
        fixed_ref = reference_volume[n_iterations - (i + 1), :, :]
        nissl_slice = nissl_volume[n_iterations - (i + 1), :, :]

        warped, nissl_slice = slice_registration(fixed, warped, nissl_slice)
        warped, nissl_slice = slice_registration(fixed_ref, warped, nissl_slice)

        fixed = warped
        warped_volume[n_iterations - (i + 1), :, :] = warped
        nissl_warped[n_iterations - (i + 1), :, :] = nissl_slice

        if i % 5 == 0:
            logger.info(f"{i + 1} / {total_n_iterations} registrations done")

    fixed = moving_volume[n_iterations - 1, :, :]
    for i in range(n_iterations - 1, total_n_iterations - 1):
        warped = moving_volume[i + 1, :, :]
        fixed_ref = reference_volume[i + 1, :, :]
        nissl_slice = nissl_volume[i + 1, :, :]

        warped, nissl_slice = slice_registration(fixed, warped, nissl_slice)
        warped, nissl_slice = slice_registration(fixed_ref, warped, nissl_slice)

        fixed = warped
        warped_volume[i + 1, :, :] = warped  # 264 -> 527
        nissl_warped[i + 1, :, :] = nissl_slice

        if i % 5 == 0:
            logger.info(f"{i + 1} / {total_n_iterations} registrations done")

    return warped_volume, nissl_warped


def main():
    """Implement main function."""
    args = parse_args()

    logger.info("Loading volumes")
    nissl = check_and_load(args.nissl_path)
    ccfv2 = check_and_load(args.ccfv2_path)
    ccfv3 = check_and_load(args.ccfv3_path)

    logger.info("Start registration...")
    warped_atlas, warped_nissl = registration(ccfv3, ccfv2, nissl)

    logger.info("Saving results...")
    output_dir = get_results_dir() / "nissl-to-ccfv3"
    output_dir.mkdir(parents=True)
    np.save(output_dir / "warped_ccfv2", warped_atlas)
    np.save(output_dir / "warped_nissl", warped_nissl)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
