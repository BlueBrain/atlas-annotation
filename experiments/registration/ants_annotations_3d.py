"""Perform the 3D registration on CCF v2/v3 atlases."""
import logging
import pathlib
import sys

logger = logging.getLogger("ants-annotations-3d")


def main():
    """Run the script."""
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
    ccf_v2_merged_path = config["ccfv2_v3_merge"]["ccf_v2_merged"]
    ccf_v3_merged_path = config["ccfv2_v3_merge"]["ccf_v3_merged"]
    nii_output = pathlib.Path(config["registration"][experiment_name]["nii_output"])

    logger.info(f"Reading CCFv2 atlas from {ccf_v2_merged_path}")
    ccf_v2_merged, header_v2 = nrrd.read(ccf_v2_merged_path)
    logger.info(f"Reading CCFv3 atlas from {ccf_v3_merged_path}")
    ccf_v3_merged, header_v3 = nrrd.read(ccf_v3_merged_path)

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
