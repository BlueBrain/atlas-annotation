"""Script that maps Gene Expression to Nissl."""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from pathlib import Path

import numpy as np
from skimage.color import rgb2gray
from utils import get_results_dir

from atlannot import load_volume
from atlannot.ants import register, transform

# Initialize the logger
logger = logging.getLogger("gene-to-nissl")
DATA_FOLDER = pathlib.Path(__file__).resolve().parent.parent / "data"


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gene_path",
        type=Path,
        help="""\
        Path to Gene Expression.
        """,
    )
    parser.add_argument(
        "--section_numbers",
        type=Path,
        help="""\
        Path to json containing section numbers of gene expression.
        """,
    )
    parser.add_argument(
        "--nissl_path",
        type=Path,
        default=DATA_FOLDER / "ara_nissl_25.nrrd",
        help="""\
        Path to Nissl Volume.
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
        raise ValueError(f"The specified path {path} does not exist.")

    volume = load_volume(path, normalize=False)
    return volume.astype(np.float32)


def registration(
    nissl_volume: np.array, gene_volume: np.array, section_numbers: np.array
) -> np.array:
    """Compute registration transform between a couple of volumes.

    Parameters
    ----------
    nissl_volume
        Nissl volume (fixed volume during registration).
    gene_volume
        Gene to register.(moving volume during registration).
    section_numbers
        Section numbers of every gene slice in gene volume.

    Returns
    -------
    warped_genes : np.array
        Warped slice.
    """
    rgb = False
    if gene_volume.ndim == 4:
        rgb = True

    warped_genes = []
    for i, (section_number, gene_slice) in enumerate(zip(section_numbers, gene_volume)):
        try:
            nissl_slice = nissl_volume[int(section_number)]
        except IndexError:
            continue

        if rgb:
            gene_slice_rgb = gene_slice.copy()
            gene_slice = rgb2gray(gene_slice)

        nii_data = register(nissl_slice, gene_slice, is_atlas=False)

        if rgb:
            warped = np.zeros_like(gene_slice_rgb)
            warped[:, :, 0] = transform(gene_slice_rgb[:, :, 0], nii_data)
            warped[:, :, 1] = transform(gene_slice_rgb[:, :, 1], nii_data)
            warped[:, :, 2] = transform(gene_slice_rgb[:, :, 2], nii_data)
        else:
            warped = transform(gene_slice, nii_data)

        warped_genes.append(warped)

        if (i + 1) % 5 == 0:
            logger.info(f" {i + 1} / {gene_volume.shape[0]} registrations done")

    return np.array(warped_genes)


def main():
    """Implement main function."""
    args = parse_args()

    logger.info("Loading volumes")
    nissl = check_and_load(args.nissl_path)
    genes = check_and_load(args.gene_path)
    gene_experiment = args.gene_path.stem
    gene_name = args.gene_path.parent.stem

    with open(args.section_numbers) as f:
        json_dict = json.load(f)

    section_numbers = json_dict["section_numbers"]
    axis = json_dict["axis"]

    if axis == "sagittal":
        nissl = np.transpose(nissl, (2, 0, 1))

    if nissl.shape[1:] != genes.shape[1:3]:
        raise ValueError(
            f"It seems the nissl ({nissl.shape}) and genes ({genes.shape}) "
            "do not have the same shape !"
        )

    if genes.shape[0] != len(section_numbers):
        raise ValueError(
            f"The length of the list of the section numbers ({len(section_numbers)})"
            f" has to be consistent to the genes shape ({genes.shape[0]})"
        )

    logger.info("Start registration...")

    warped_genes = registration(nissl, genes, section_numbers)

    logger.info("Saving results...")
    output_dir = get_results_dir() / "gene-to-nissl" / gene_name
    output_dir.mkdir(parents=True)
    np.save(output_dir / f"{gene_experiment}_warped_gene", warped_genes)

    with open(output_dir / f"{gene_experiment}_section_numbers", "w") as f:
        json.dumps(json_dict)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
