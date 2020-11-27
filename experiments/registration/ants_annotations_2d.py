"""Perform the 2D registration on CCF v2/v3 atlases."""
import argparse
import logging
import pathlib
import sys

logger = logging.getLogger("ants-annotations-2d")

script_info = """
This script performs a 2D registration of the merged CCF v2/v3 atlases.

We load the merged CCF v2/v3 atlases that were produced by the
ccf_v2_v3_merge.py script. If the files are not available then that script
needs to be run manually first.

The registration produces displacement fields which are saved under the
paths specified in the config.toml file. The warping of the atlases is
managed by a different script (warp_atlases.py), which can be run after
this script.
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
    parser.parse_args(argv)

    logger.info("Loading libraries")
    import nrrd
    import numpy as np
    import toml
    from tqdm import tqdm

    import deal.ants

    logger.info("Loading configuration")
    config_file = pathlib.Path("experiments/config.toml")
    experiment_name = pathlib.Path(__file__).stem
    with config_file.open() as f:
        config = toml.load(f)
    ccf_v2_merged_path = config["ccf_v2_v3_merge"]["ccf_v2_merged"]
    ccf_v3_merged_path = config["ccf_v2_v3_merge"]["ccf_v3_merged"]
    nii_output = pathlib.Path(config["registration"][experiment_name]["nii_output"])

    logger.info(f"Reading CCFv2 atlas from {ccf_v2_merged_path}")
    ccf_v2_merged, header_v2 = nrrd.read(ccf_v2_merged_path)
    logger.info(f"Reading CCFv3 atlas from {ccf_v3_merged_path}")
    ccf_v3_merged, header_v3 = nrrd.read(ccf_v3_merged_path)

    logger.info("Registering 2D slices")
    nii_2d_slices = []
    n_slices = len(ccf_v2_merged)
    for img_v2, img_v3 in tqdm(zip(ccf_v2_merged, ccf_v3_merged), total=n_slices):
        nii = deal.ants.register(
            fixed=img_v3.astype(np.float32),
            moving=img_v2.astype(np.float32),
        )
        nii_2d_slices.append(nii)

    logger.info("Stacking 2d slices")
    nii_2d = deal.ants.stack_2d_transforms(nii_2d_slices)

    logger.info(f"Writing {nii_output}")
    nii_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(nii_output, nii_2d)

    logger.info("Finished")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
