"""Perform the 3D registration on average brain / Nissl."""
import logging
import pathlib
import sys

logger = logging.getLogger("ants-nissl-3d")


def main():
    """Run the script."""
    logger.info("Loading libraries")
    import numpy as np
    import toml

    import deal.ants

    logger.info("Loading configuration")
    config_file = pathlib.Path("experiments/config.toml")
    experiment_name = pathlib.Path(__file__).stem
    with config_file.open() as f:
        config = toml.load(f)
    average_brain_path = pathlib.Path(config["data"]["average_brain"])
    nissl_path = pathlib.Path(config["data"]["nissl"])
    nii_output = pathlib.Path(config["registration"][experiment_name]["nii_output"])

    logger.info(f"Loading {average_brain_path}")
    average_brain = deal.load_volume(average_brain_path)
    logger.info(f"Loading {nissl_path}")
    nissl = deal.load_volume(nissl_path)

    logger.info("Registering")
    nii_3d = deal.ants.register(fixed=average_brain, moving=nissl)

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
