"""Compute registration at intermediate hierarchy levels."""
import argparse
import logging
import pathlib
import sys

name = "Registration with specific hierarchy"
logger = logging.getLogger(name)

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


def main(argv=None):
    """Run the main script.

    Parameters
    ----------
    argv : sequence or None
        The argument vector. If None then the arguments are parsed from
        the command line directly.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=script_info,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--img-ref",
        default="results/annotation_atlas/ccf_v3_merged.nrrd",
        type=str,
        help="The annotation image/volume of reference.",
    )
    parser.add_argument(
        "--img-mov",
        default="results/annotation_atlas/ccf_v2_merged.nrrd",
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
        help=(
            "Hierarchy at which to apply the registration algorithm. "
            "0 meaning the most detailed brain region decomposition, "
            "the so-called leaf regions, 10 the distinction background/foreground."
        ),
    )
    parser.add_argument(
        "--out-file",
        default="results/registration_specific_hierarchy.npy",
        type=str,
        help="The name of the file of the resulting warped image.",
    )
    args = parser.parse_args(argv)

    logger.info("Loading libraries")

    import deal
    from deal.ants import register, transform
    from deal.atlas import unfurl_regions
    from deal.notebook import print_misalignments
    from deal.utils import create_description, load_region_meta, saving_results

    logger.info("Loading Images...")
    img_ref = deal.load_volume(pathlib.Path(args.img_ref))
    img_mov = deal.load_volume(pathlib.Path(args.img_mov))

    logger.info(f"Reference image: {args.img_ref}")
    logger.info(f"Moving image: {args.img_mov}")
    logger.info(f"Shape of the reference image: {img_ref.shape}")
    logger.info(f"Shape of the moving image: {img_mov.shape}")

    logger.info("Read brain regions...")
    region_meta = load_region_meta(args.brain_regions)

    logger.info("Unfurl regions...")
    img_ref_hierarchy = unfurl_regions(img_ref, region_meta)
    img_mov_hierarchy = unfurl_regions(img_mov, region_meta)

    logger.info("ANTsPY registration...")
    df = register(
        img_ref_hierarchy[args.hierarchy].astype("float32"),
        img_mov_hierarchy[args.hierarchy].astype("float32"),
    )

    logger.info("ANTsPY apply transform...")
    img_reg = transform(img_mov.astype("float32"), df, interpolator="genericLabel")
    img_reg = img_reg.astype("int")

    out_path = pathlib.Path(args.out_file)
    logger.info(f"Saving Results at {out_path}...")
    description = create_description(name, args)
    saving_results(out_path, img_reg=img_reg, df=df, description=description)

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
    sys.exit(main())
