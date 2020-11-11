"""Goal: computing machine learning 2D registration between the reference and
moving images/volumes.

Assumptions:
- The input images/volumes have to have the same shape.

Steps:
- Loading input images/volumes.
- For each 2D images, machine learning 2D registration.
- Applying resulting transformations to the original moving image.
- Saving results.
"""
import argparse
import logging
import pathlib

import nrrd
import numpy as np
from tqdm import tqdm
from warpme.ml_utils import load_model, merge_global_local

logger = logging.getLogger("Registration with Machine Learning")

parser = argparse.ArgumentParser()
parser.add_argument("--img-ref",
                    default="data/average_template_25.npy",
                    type=str,
                    help="The image/volume of reference.")
parser.add_argument("--img-mov",
                    default="data/atlasVolume.npy",
                    type=str,
                    help="The moving image/volume (=to warp).")
parser.add_argument("--out-file",
                    default="experiments/results/registration_machine_learning.npy",
                    type=str,
                    help="The name of the file of the resulting warped image.")
args = parser.parse_args()


def main():
    """Computes machine learning registration. """
    logger.info("Loading Images...")
    if pathlib.Path(args.img_ref).suffix == '.nrrd':
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

    logger.info("Loading ML models...")
    model_global = load_model('/gpfs/bbp.cscs.ch/project/proj101/'
                              'pretrained_models/global/calm_camel/calm_camel.h5')
    model_local = load_model('/gpfs/bbp.cscs.ch/project/proj101/'
                             'pretrained_models/local/cute_cat/cute_cat.h5')
    model_merged = merge_global_local(model_global, model_local)

    logger.info("Registration...")
    img_ref = np.expand_dims(img_ref, axis=3)
    img_mov = np.expand_dims(img_mov, axis=3)
    ml_input = np.concatenate((img_ref, img_mov), axis=3)
    img_regs, delta_xys = [], []
    logger.info(f"Shape of the ML input: {ml_input.shape}")
    for dim, x in tqdm(enumerate(ml_input)):
        img_reg, delta_xy = model_merged.predict(np.expand_dims(x, axis=0), batch_size=1)
        img_regs.append(img_reg)
        delta_xys.append(delta_xy)

    out_path = pathlib.Path(args.out_file)
    if not out_path.parent.exists():
        pathlib.Path.mkdir(out_path.parent, parents=True)
    df_path = out_path.parent / f"{out_path.stem}_df{out_path.suffix}"
    logger.info(f"Saving Results at {args.out_file} and {df_path}...")
    np.save(out_path, np.squeeze(np.stack(img_regs)))
    np.save(df_path, np.squeeze(np.stack(delta_xys)))

    logger.info("DONE")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s || %(levelname)s || %(name)s || %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
