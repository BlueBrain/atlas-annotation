"""Script that maps Gene Expression to Nissl."""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from pathlib import Path

import numpy as np
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
        logger.error(f"The specified path {path} does not exist.")
        return 1
    volume = load_volume(path, normalize=False)
    return volume.astype(np.float32)


def slice_registration(
    fixed: np.array, moving: np.array
) -> tuple[np.array, np.array | None]:
    """Compute registration transform between a couple of slices.

    Parameters
    ----------
    fixed
        Fixed slice.
    moving
        Moving slice.

    Returns
    -------
    warped : np.array
        Warped slice.
    """
    nii_data = register(fixed, moving, is_atlas=False)
    warped = transform(moving, nii_data)

    return warped


def main():
    """Implement main function."""
    args = parse_args()

    logger.info("Loading volumes")
    nissl = check_and_load(args.nissl_path)
    genes = check_and_load(args.gene_path)
    gene_experiment = args.gene_path.stem

    with open(args.section_numbers) as f:
        json_dict = json.load(f)

    section_numbers = json_dict["section_numbers"]

    logger.info("Start registration...")

    warped_genes = []
    for section_number, gene_slice in zip(section_numbers, genes):
        try:
            nissl_slice = nissl[section_number]
        except ValueError:
            continue
        warped_genes.append(slice_registration(nissl_slice, gene_slice))

    warped_genes = np.array(warped_genes)

    logger.info("Saving results...")
    output_dir = get_results_dir() / "gene-to-nissl"
    output_dir.mkdir(parents=True)
    np.save(output_dir / f"{gene_experiment}_warped_gene", warped_genes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
