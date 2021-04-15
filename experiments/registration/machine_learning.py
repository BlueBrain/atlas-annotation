"""Machine learning registration."""
import argparse
import logging
import pathlib
import sys

name = "Registration with Machine Learning"
logger = logging.getLogger(name)

script_info = """
Goal: computing machine learning 2D registration between the reference and
moving images/volumes.

Assumptions:
- The input images/volumes have to have the same shape.

Steps:
- Loading input images/volumes.
- For each 2D images, machine learning 2D registration.
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
        default="results/registration_machine_learning.npy",
        type=str,
        help="The name of the file of the resulting warped image.",
    )
    args = parser.parse_args(argv)

    logger.info("Loading libraries")
    import numpy as np
    from tqdm import tqdm
    from warpme.ml_utils import load_model, merge_global_local

    import deal
    from deal.utils import create_description, saving_results

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

    logger.info("Loading ML models...")
    model_global = load_model(
        "/gpfs/bbp.cscs.ch/project/proj101/"
        "pretrained_models/global/calm_camel/calm_camel.h5"
    )
    model_local = load_model(
        "/gpfs/bbp.cscs.ch/project/proj101/"
        "pretrained_models/local/cute_cat/cute_cat.h5"
    )
    model_merged = merge_global_local(model_global, model_local)

    logger.info("Registration...")
    img_ref = np.expand_dims(img_ref, axis=3)
    img_mov = np.expand_dims(img_mov, axis=3)
    ml_input = np.concatenate((img_ref, img_mov), axis=3)
    img_regs, delta_xys = [], []
    logger.info(f"Shape of the ML input: {ml_input.shape}")
    for dim, x in tqdm(enumerate(ml_input)):
        img_reg, delta_xy = model_merged.predict(
            np.expand_dims(x, axis=0), batch_size=1
        )
        img_regs.append(img_reg)
        delta_xys.append(delta_xy)

    out_path = pathlib.Path(args.out_file)
    logger.info(f"Saving Results at {out_path}...")
    description = create_description(name, args)
    saving_results(
        out_path,
        img_reg=np.squeeze(np.stack(img_regs)),
        df=np.squeeze(np.stack(delta_xys)),
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
