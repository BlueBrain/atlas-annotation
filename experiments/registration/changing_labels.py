"""Registration with alternative region labels."""
import argparse
import logging
import pathlib
import sys

name = "Registration with changed labels"
logger = logging.getLogger(name)

script_info = """
Goal: Computing the registration between two images/volumes after switching
randomly the labels.

Assumptions:
- The input images/volumes have the same shape.
- The input images/volumes are considered as label images.
- The registration is computed on the entire input images at once. Which means
  that if volumes are specified, the registration is a 3D registration. If 2D
  images are specified, this is a 2D registration.

Steps:
- Loading of the images
- Creation of union list containing all the labels appearing at least in one
  of the two input images/volumes.
- The conversion previous labels/new labels is done by taking as new label
  the position in the list of the previous label. For example:
  Union List: [0, 1002, 6, 9]
  New labels: [0, 1, 2, 3]
  Which means 0 stays 0 in the new volume, 1002 is becoming 1, 6 is
  becoming 2, ... Obviously, there are other strategies to convert previous
  labels to new ones.
- Creation of new images/volumes with corresponding new labels.
- Computation of the ANTsPY registration on the new images/volumes.
- Applying transform found in the previous step at the initial images/volumes.
- Computation of baseline misalignement (between inputs) and the results
  misalignment (between input reference and warped moving image).
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
        "--seed", default=1234, type=int, help="The seed for the change of labels."
    )
    parser.add_argument(
        "--out-file",
        default="results/registration_changing_labels.npy",
        type=str,
        help="The name of the file of the resulting warped image.",
    )
    args = parser.parse_args(argv)

    logger.info("Loading libraries")
    import deal
    from deal.ants import register, transform
    from deal.atlas import get_misalignment
    from deal.utils import create_description, remap_labels, saving_results

    logger.info("Loading Images...")
    img_ref = deal.load_volume(pathlib.Path(args.img_ref))
    img_mov = deal.load_volume(pathlib.Path(args.img_mov))

    logger.info(f"Reference image: {args.img_ref}")
    logger.info(f"Moving image: {args.img_mov}")
    logger.info(f"Shape of the reference image: {img_ref.shape}")
    logger.info(f"Shape of the moving image: {img_mov.shape}")

    logger.info("Change randomly the labels of the images...")
    imgs_changed = remap_labels([img_ref, img_mov], seed=args.seed)

    logger.info("Intensity-based registration with ANTsPY...")
    df = register(imgs_changed[0].astype("float32"), imgs_changed[1].astype("float32"))
    logger.info("Apply transform to the Moving Image ...")
    img_reg = transform(img_mov.astype("float32"), df, interpolator="genericLabel")
    img_reg = img_reg.astype("int")

    out_path = pathlib.Path(args.out_file)
    logger.info(f"Saving Results at {args.out_file}...")
    description = create_description(name, args)
    saving_results(
        output_dir=out_path,
        img_ref=img_ref,
        img_mov=img_mov,
        img_reg=img_reg,
        df=df,
        description=description,
    )

    logger.info("Analysis of the results...")
    base_mis = get_misalignment(img_ref, img_mov) * 100
    base_mis_fg = get_misalignment(img_ref, img_mov, fg_only=True) * 100
    logger.info(
        f"Baseline misalignement full(fg): {base_mis:6.2f}% ({base_mis_fg:6.2f}%) "
    )

    new_mis = get_misalignment(img_ref, img_reg) * 100
    new_mis_fg = get_misalignment(img_ref, img_reg, fg_only=True) * 100
    logger.info(f"New misalignement full(fg): {new_mis:6.2f}% ({new_mis_fg:6.2f}%) ")
    logger.info("DONE")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s || %(levelname)s || %(name)s || %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
