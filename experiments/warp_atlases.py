"""Warp the CCFv2 atlas with existing displacement fields."""
import logging
import pathlib
import sys

logger = logging.getLogger("warp-atlases")


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
    ccf_v2_merged_path = pathlib.Path(config["ccfv2_v3_merge"]["ccf_v2_merged"])
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
