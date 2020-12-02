"""Run registration after creating a middle line on input images."""
import argparse
import logging
import pathlib
import sys

logger = logging.getLogger("Registration with middle line")

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
        "--out-file",
        default="results/registration_with_middle_line.npy",
        type=str,
        help="The name of the file of the resulting warped image.",
    )
    args = parser.parse_args(argv)

    logger.info("Loading libraries")
    import nrrd
    import numpy as np
    from tqdm import tqdm
    from warpme.base import DisplacementField

    from deal.ants import register

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
        h, w = current_ref.shape

        fixed_img_bar = current_ref.copy().astype(np.float32)
        moving_img_bar = current_mov.copy().astype(np.float32)

        fixed_img_bar[..., w // 2] = 1
        moving_img_bar[..., w // 2] = 1

        df = register(fixed_img_bar, moving_img_bar)
        df = np.squeeze(df)
        registered_antspy_middle_bar[i, :, :] = DisplacementField(
            df[:, :, 1], df[:, :, 0]
        ).warp(current_mov)
        dfs.append(df)

    out_path = pathlib.Path(args.out_file)
    if not out_path.parent.exists():
        pathlib.Path.mkdir(out_path.parent, parents=True)
    df_path = out_path.parent / f"{out_path.stem}_df{out_path.suffix}"
    logger.info(f"Saving the results at {out_path} and {df_path}")
    np.save(out_path, registered_antspy_middle_bar)
    np.save(df_path, np.stack(dfs))
    logger.info("DONE")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s || %(levelname)s || %(name)s || %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
