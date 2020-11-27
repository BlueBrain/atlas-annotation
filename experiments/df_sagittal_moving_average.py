"""Apply moving average to stacks of displacement fields."""
import argparse
import logging
import pathlib
import sys

logger = logging.getLogger("sagittal-avg")

script_info = """
Apply a moving average to all registered displacement fields.

This script will load all displacement fields registered in the
displacement field output folder (usually results/registration)
and apply the three slightly different strategies of a moving average
along the sagittal axis:
- Flat/uniform average
- Weighted average with Gaussian weights
- Weighted average with custom weights

The resulting averaged displacement fields are stored in the same location
as the input displacement fields with the file name extended by the
averaging method used.
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
    from scipy import ndimage

    logger.info("Loading configuration")
    config_file = pathlib.Path("experiments/config.toml")
    with config_file.open() as f:
        config = toml.load(f)

    logger.info("Starting the averaging")
    for experiment_name in config["registration"]:
        nii_file = config["registration"][experiment_name].get("nii_output")
        if nii_file is None:
            logger.warning(
                f'The registration experiment "{experiment_name}" has no '
                '"nii_file" output property, skipping'
            )
            continue
        else:
            nii_file = pathlib.Path(nii_file)

        logger.info(f"Loading displacements for {experiment_name}")
        nii_data = np.load(nii_file)

        logger.info("Applying the moving average")
        nii_avg = {
            "uniform": ndimage.uniform_filter1d(nii_data, 3, axis=0),
            "gaussian": ndimage.gaussian_filter1d(nii_data, 1, axis=0),
            "correlate": ndimage.correlate1d(nii_data, [1 / 4, 1 / 2, 1 / 4], axis=0),
        }

        logger.info("Saving the averaged displacements")
        for name, nii_data_avg in nii_avg.items():
            out_path = nii_file.with_name(f"{nii_file.stem}_{name}.npy")
            if out_path.exists():
                logger.warning(f"File {out_path} already exists, skipping")
                continue

            logger.info(f"Writing {out_path}")
            np.save(out_path, nii_data_avg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
