"""Run registration after creating a middle line on input images."""
import argparse
import logging
import pathlib
import sys

name = "Registration with middle line"
logger = logging.getLogger(name)

script_info = """
Goal: Computing ANTsPY registration after creating a middle line on input images.

Assumptions:
- The input images/volumes have to have the same shape.

Steps:
- Loading input images/volumes.
- For each 2D images, creation of a middle bar and registration of the resulting images.
- Applying resulting transformations to the original moving image.
- Saving results.
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
        default="data/average_template_25.npy",
        type=str,
        help="The image/volume of reference.",
    )
    parser.add_argument(
        "--img-mov",
        default="data/atlasVolume.npy",
        type=str,
        help="The moving image/volume (=to warp).",
    )
    parser.add_argument(
        "--line-thickness",
        default=2,
        type=int,
        help="The middle line thickness.",
    )
    parser.add_argument(
        "--out-file",
        default="results/registration_with_middle_line.npy",
        type=str,
        help="The name of the file of the resulting warped image.",
    )
    args = parser.parse_args(argv)

    logger.info("Loading libraries")
    import numpy as np
    from tqdm import tqdm
    from warpme.base import DisplacementField

    import deal
    from deal.ants import register
    from deal.utils import add_middle_line, create_description, saving_results

    logger.info("Loading Images...")
    img_ref = deal.load_volume(pathlib.Path(args.img_ref))
    img_mov = deal.load_volume(pathlib.Path(args.img_mov))

    logger.info(f"Reference image: {args.img_ref}")
    logger.info(f"Moving image: {args.img_mov}")
    logger.info(f"Shape of the reference image: {img_ref.shape}")
    logger.info(f"Shape of the moving image: {img_mov.shape}")

    if len(img_ref.shape) == 2:
        img_ref = np.expand_dims(img_ref, axis=0)
        img_mov = np.expand_dims(img_mov, axis=0)

    logger.info("Creation of a middle bar and registration of the images")
    registered_antspy_middle_bar = np.zeros_like(img_ref, dtype=np.float32)
    dfs = []
    n_img, *_ = img_ref.shape
    for i, (current_ref, current_mov) in tqdm(
        enumerate(zip(img_ref, img_mov)), total=n_img
    ):
        fixed_img_bar = add_middle_line(current_ref, axis=0, thickness=2)
        moving_img_bar = add_middle_line(current_mov, axis=0, thickness=2)

        df = register(fixed_img_bar, moving_img_bar)
        df = np.squeeze(df)
        registered_antspy_middle_bar[i, :, :] = DisplacementField(
            df[:, :, 1], df[:, :, 0]
        ).warp(current_mov)
        dfs.append(df)

    out_path = pathlib.Path(args.out_file)
    logger.info(f"Saving the results at {out_path}")
    description = create_description(name, args)
    saving_results(
        out_path,
        img_reg=registered_antspy_middle_bar,
        df=np.stack(dfs),
        description=description,
    )
    logger.info("DONE")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s || %(levelname)s || %(name)s || %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
