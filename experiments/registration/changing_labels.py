"""Registration with alternative region labels."""
import argparse
import logging
import pathlib
import sys

logger = logging.getLogger("Registration with changed labels")

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
    import nrrd
    import numpy as np
    from tqdm import tqdm

    from deal.ants import register, transform
    from deal.atlas import get_misalignment

    logger.info("Loading Images...")
    if pathlib.Path(args.img_ref).suffix == ".nrrd":
        img_ref, _ = nrrd.read(pathlib.Path(args.img_ref))
        img_mov, _ = nrrd.read(pathlib.Path(args.img_mov))
    else:
        img_ref = np.load(pathlib.Path(args.img_ref))
        img_mov = np.load(pathlib.Path(args.img_mov))

    logger.info(f"Reference image: {args.img_ref}")
    logger.info(f"Moving image: {args.img_mov}")
    logger.info(f"Shape of the reference image: {img_ref.shape}")
    logger.info(f"Shape of the moving image: {img_mov.shape}")

    logger.info("Change randomly the labels of the images...")
    values_ref = np.unique(img_ref)
    values_mov = np.unique(img_mov)
    union_list = list(set(values_mov).union(set(values_ref)))
    logger.info(f"Number of labels: {len(union_list)}")
    logger.info("Decide the new labels...")
    np.random.seed(seed=args.seed)
    new_labels = np.arange(len(union_list))
    np.random.shuffle(new_labels)

    # Create new numpy containing the new labels
    img_ref_changed = np.zeros_like(img_ref)
    img_mov_changed = np.zeros_like(img_mov)

    for new, old in tqdm(zip(new_labels, union_list)):
        img_mov_changed[img_mov == old] = new
        img_ref_changed[img_ref == old] = new

    img_mov_changed = img_mov_changed.astype("float32")
    img_ref_changed = img_ref_changed.astype("float32")

    logger.info("Intensity-based registration with ANTsPY...")
    df = register(img_ref_changed, img_mov_changed)
    logger.info("Apply transform to the Moving Image ...")
    img_reg = transform(img_mov.astype("float32"), df, interpolator="genericLabel")
    img_reg = img_reg.astype("int")

    out_path = pathlib.Path(args.out_file)
    if not out_path.parent.exists():
        pathlib.Path.mkdir(out_path.parent, parents=True)
    df_path = out_path.parent / f"{out_path.stem}_df{out_path.suffix}"
    logger.info(f"Saving Results at {args.out_file} and {df_path}...")
    np.save(out_path, img_reg)
    np.save(df_path, df)

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
