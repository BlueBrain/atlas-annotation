"""Perform the 3D registration on CCF v2/v3 atlases."""
import argparse
import logging
import pathlib
import sys

logger = logging.getLogger("ants-annotations-3d")

script_info = """
This script performs a 3D registration of the merged CCF v2/v3 atlases.

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
    import numpy as np
    import toml

    import deal.ants

    logger.info("Loading configuration")
    config_file = pathlib.Path("experiments/config.toml")
    experiment_name = pathlib.Path(__file__).stem
    with config_file.open() as f:
        config = toml.load(f)
    ccf_v2_merged_path = config["ccf_v2_v3_merge"]["ccf_v2_merged"]
    ccf_v3_merged_path = config["ccf_v2_v3_merge"]["ccf_v3_merged"]
    nii_output = pathlib.Path(config["registration"][experiment_name]["nii_output"])

    logger.info(
        f"Reading CCFv2 atlas from {ccf_v2_merged_path} "
        f"and CCFv3 atlas from {ccf_v3_merged_path}"
    )
    ccf_v2_merged = deal.load_volume(ccf_v2_merged_path)
    ccf_v3_merged = deal.load_volume(ccf_v3_merged_path)

    logger.info("Registering")
    nii_3d = deal.ants.register(
        fixed=ccf_v3_merged.astype(np.float32),
        moving=ccf_v2_merged.astype(np.float32),
    )

    logger.info(f"Writing {nii_output}")
    nii_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(nii_output, nii_3d)

    logger.info("Finished")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
