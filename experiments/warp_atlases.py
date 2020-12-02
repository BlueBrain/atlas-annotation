"""Warp the CCFv2 atlas with existing displacement fields."""
import argparse
import logging
import pathlib
import sys

logger = logging.getLogger("warp-atlases")

script_info = """
Warp merged CCF v2/v3 atlases with all available displacement fields.

We load the merged CCF v2/v3 atlases that were produced by the
ccf_v2_v3_merge.py script. If the files are not available then that script
needs to be run manually first.

Next we search for all displacement fields that were produced by the
corresponding registration scripts. If you haven't run any registration
scripts yet, then you need to do it first. The registered displacement
fields are usually stored in the results/registration folder, but check
in the config.toml file under the section corresponding to the script's
name if it is still the correct path.

For each found displacement field the CCFv2 atlas is warped and the resulting
atlas is saved in the output directory. The output directory path can also be
found in the config.toml file.
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

    import deal.ants

    logger.info("Loading configuration")
    config_file = pathlib.Path("experiments/config.toml")
    experiment_name = pathlib.Path(__file__).stem
    with config_file.open() as f:
        config = toml.load(f)
    ccf_v2_merged_path = pathlib.Path(config["ccf_v2_v3_merge"]["ccf_v2_merged"])
    registration_dir = pathlib.Path(config[experiment_name]["registration_dir"])
    output_dir = pathlib.Path(config[experiment_name]["output_dir"])

    logger.info(f"Reading CCFv2 atlas from {ccf_v2_merged_path}")
    ccf_v2_merged, header_v2 = nrrd.read(ccf_v2_merged_path)

    logger.info("Applying all available transformations")
    output_dir.mkdir(parents=True, exist_ok=True)
    for nii_file in pathlib.Path(registration_dir).glob("*.npy"):
        filename_v2 = f"{ccf_v2_merged_path.stem}_{nii_file.stem}.nrrd"
        output_path_v2 = output_dir / filename_v2

        if output_path_v2.exists():
            logger.warning(f"{output_path_v2} already exists, skipping")
            continue

        logger.info(f"Loading {nii_file}")
        nii_data = np.load(str(nii_file))

        logger.info(f"Applying {nii_file} to {ccf_v2_merged_path.stem}")
        ccf_v2_warped = deal.ants.transform(
            image=ccf_v2_merged.astype(np.float32),
            nii_data=nii_data,
            interpolator="genericLabel",
        )

        logger.info(f"Writing {output_path_v2}")
        nrrd.write(
            filename=str(output_path_v2),
            data=ccf_v2_warped.astype(np.int),
            header=header_v2,
        )

    logger.info("Finished")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
