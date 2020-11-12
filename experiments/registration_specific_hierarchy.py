"""Goal: Computing registration at an intermediate level of hierarchy.

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
import argparse
import json
import logging
import pathlib

import nrrd
import numpy as np

from deal.ants import register, transform
from deal.atlas import RegionMeta, unfurl_regions
from deal.notebook import print_misalignments

logger = logging.getLogger("Registration with specific hierarchy")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--img-ref",
    default="data/annotation_atlas/annotations_ccf_v3_merged.nrrd",
    type=str,
    help="The annotation image/volume of reference.",
)
parser.add_argument(
    "--img-mov",
    default="data/annotation_atlas/annotations_ccf_v2_merged.nrrd",
    type=str,
    help="The annotation image/volume to warp.",
)
parser.add_argument(
    "--brain-regions",
    default="data/annotation_atlas/brain_regions.json",
    type=str,
    help="Path to file containing brain regions hierarchy.",
)
parser.add_argument(
    "--hierarchy",
    default=0,
    type=int,
    help="Hierarchy at which to apply the registration algorithm. 0 meaning the "
    "most detailed brain regions decomposition, 10 the distinction "
    "background/foreground.",
)
parser.add_argument(
    "--out-file",
    default="experiments/results/registration_specific_hierarchy.npy",
    type=str,
    help="The name of the file of the resulting warped image.",
)
args = parser.parse_args()


def main():
    """Compute registration at specific hierarchy."""
    logger.info("Loading Images...")
    img_ref, _ = nrrd.read(pathlib.Path(args.img_ref))
    img_mov, _ = nrrd.read(pathlib.Path(args.img_mov))

    logger.info(f"Reference image: {args.img_ref}")
    logger.info(f"Moving image: {args.img_mov}")
    logger.info(f"Shape of the reference image: {img_ref.shape}")
    logger.info(f"Shape of the moving image: {img_mov.shape}")

    logger.info("Read brain regions...")
    with open(args.brain_regions, "r") as f:
        brain_regions = json.load(f)
    region_meta = RegionMeta.from_root_region(brain_regions["msg"][0])

    logger.info("Unfurl regions...")
    img_ref_hierarchy = unfurl_regions(img_ref, region_meta)
    img_mov_hierarchy = unfurl_regions(img_mov, region_meta)

    img_ref_hierarchy = img_ref_hierarchy.astype("float32")
    img_mov_hierarchy = img_mov_hierarchy.astype("float32")

    logger.info("ANTsPY registration...")
    df = register(img_ref_hierarchy[args.hierarchy], img_mov_hierarchy[args.hierarchy])

    logger.info("ANTsPY apply transform...")
    img_reg = transform(img_mov.astype("float32"), df, interpolator="genericLabel")
    img_reg = img_reg.astype("int")

    out_path = pathlib.Path(args.out_file)
    if not out_path.parent.exists():
        pathlib.Path.mkdir(out_path.parent, parents=True)
    img_path = (
        out_path.parent / f"{out_path.stem}_hierarchy{args.hierarchy}{out_path.suffix}"
    )
    df_path = (
        out_path.parent
        / f"{out_path.stem}_hierarchy{args.hierarchy}_df{out_path.suffix}"
    )
    logger.info(f"Saving Results at {img_path} and {df_path}...")
    np.save(img_path, img_reg)
    np.save(df_path, df)

    logger.info("Analysis of the results...")
    logger.info("Baseline misalignment....")
    print_misalignments(img_ref_hierarchy, img_mov_hierarchy)
    logger.info("Results misalignment...")
    img_reg_hierarchy = unfurl_regions(img_reg, region_meta)
    print_misalignments(img_ref_hierarchy, img_reg_hierarchy)
    logger.info("DONE")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s || %(levelname)s || %(name)s || %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
