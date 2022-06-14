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
"""Evaluation pipeline."""
import argparse
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
import utils
from skimage.metrics import structural_similarity as ssim
from warpme.metrics import iou_score

from atlannot.atlas.align import get_misalignment
from atlannot.utils import atlas_symmetry_score, load_volume, stain_symmetry_score


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="""\
        If provided only the given experiment will be evaluated,
        otherwise all experiments will be processed.
        """,
    )
    return parser.parse_args()


def main():
    """Evaluate."""
    args = parse_args()
    if args.experiment:
        experiments = [pathlib.Path(args.experiment)]
    else:
        experiments = list(utils.get_results_dir().iterdir())

    print("Checking for existing results")
    existing_metrics = []
    for experiment_path in sorted(experiments):
        metrics_path = experiment_path / "metrics.csv"
        if metrics_path.exists():
            existing_metrics.append(metrics_path)
    if len(existing_metrics) > 0:
        print("WARNING: The following metrics already exist:")
        for metrics_path in existing_metrics:
            print(f"> {metrics_path}")
        print("WARNING: metrics won't be recomputed - the existing ones will be used")
        print("WARNING: if you want to recompute then delete the corresponding files")
        input("Press any key to continue")

    print("Loading volumes")
    volumes = load_volumes()

    print("Evaluation")
    for experiment_path in sorted(experiments):
        evaluate(experiment_path, volumes)


def load_volumes():
    """Load reference volumes.

    Returns
    -------
    volumes : dict
        Dictionary containing reference volumes.
    """
    volumes = {
        "avg": load_volume(utils.get_avg_brain_path()),
        "nissl": load_volume(utils.get_nissl_path()),
        "atl v2": load_volume(utils.get_v2_atlas_fine_path()),
        "atl v3": load_volume(utils.get_v3_atlas_fine_path()),
    }
    return volumes


def evaluate(directory, volumes):
    """Evaluate the results.

    Parameters
    ----------
    directory : pathlib.Path
        Diretory of the experiment to evaluate.
    volumes : dict
        Dictionary containing reference volumes.
    """
    metrics_path = directory / "metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path, index_col=0)
    else:
        warped = load_volume(directory / "warped_nissl.npy")
        warped_atl = load_volume(directory / "warped_atlas.npy")
        metrics_df = compute_metrics(
            volumes["avg"], warped, volumes["atl v3"], warped_atl
        )
        metrics_df.to_csv(metrics_path)

    print_scores(directory.name, metrics_df)


def compute_metrics(fixed, warped, fixed_atl, warped_atl):
    """Compute different evaluation metrics.

    Parameters
    ----------
    fixed : np.ndarray
        Fixed input.
    warped : np.ndarray
        Registered volume.
    fixed_atl : np.ndarray
        Fixed atlas.
    warped_atl: np.ndarray
        Registered atlas.

    Returns
    -------
    all_metrics : pd.DataFrame
        DataFrame containing the resulting metrics.
    """
    ssim_score = ssim_many(fixed, warped)
    misalignment = [
        get_misalignment(fixed_atl_slice, warped_atl_slice)
        for fixed_atl_slice, warped_atl_slice in zip(fixed_atl, warped_atl)
    ]
    stain_sym = [stain_symmetry_score(img) for img in warped]
    atlas_sym = [atlas_symmetry_score(img) for img in warped_atl]
    iou_average, iou_per_sample = iou_score(
        fixed_atl, warped_atl, k=None, excluded_labels=[0]
    )

    all_metrics = pd.DataFrame(
        {
            "SSIM": ssim_score,
            "MISL": misalignment,
            "SYMS": stain_sym,
            "SYMA": atlas_sym,
            "IOU": iou_per_sample,
        }
    )

    return all_metrics


def print_scores(name, metrics_df):
    """Print resulting scores.

    Parameters
    ----------
    name: str
        Name of the experiment.
    metrics_df : pd.DataFrame
        DataFrame containing all the resulting metrics.
    """
    if metrics_df is None:
        metric_strs = ["no metrics_df found"]
    else:
        metric_strs = [
            f"{metric_name} {metrics_df[metric_name].mean():.2f}"
            for metric_name in metrics_df.columns
        ]
    print(" | ".join([f"{name.ljust(50, '.')}"] + metric_strs))


def ssim_many(imgs1, imgs2):
    """Compute SSMI for several pairs of images.

    Parameters
    ----------
    imgs1 : np.ndarray
        First input.
    imgs2 : np.ndarray
        Second input. Should be of the same shape as imgs1.

    Returns
    -------
    scores : np.ndarray
        Scores for each pair of images.
    """
    scores = [ssim(img1, img2) for img1, img2 in zip(imgs1, imgs2)]
    return np.array(scores)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
