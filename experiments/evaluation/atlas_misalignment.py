"""Compute misalignment for all registered volumes."""
import argparse
import logging
import pathlib
import sys

logger = logging.getLogger("compute-misalignment")

script_info = """
Compute misalignment for all available warped atlases.

First we load the CCFv3 atlas that was produced by the ccf_v2_v3_merge.py
script. If you haven't run that script before then you need to do it.

Then all warped CCFv2 atlases that are produced by the warp_atlases.py
are located. The warped atlases are usually found in results/warped_atlas
but check the warp_atlases.py script for any changes.

For each warped CCFv2 atlas we compute the misalignment against the CCFv3
atlas at each level of the hierarchy level. The results are stored in text
files in the results/misalignment directory.
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
    import json

    import nrrd
    import toml
    from tqdm import tqdm

    import deal.atlas
    from deal.atlas import RegionMeta
    from deal.notebook import print_misalignments

    logger.info("Loading configuration")
    config_file = pathlib.Path("experiments/config.toml")
    experiment_name = pathlib.Path(__file__).stem
    with config_file.open() as f:
        config = toml.load(f)
    brain_regions_path = pathlib.Path(config["data"]["brain_regions"])
    ccf_v3_merged_path = pathlib.Path(config["ccf_v2_v3_merge"]["ccf_v3_merged"])
    warped_dir = pathlib.Path(config["evaluation"][experiment_name]["warped_dir"])
    output_dir = pathlib.Path(config["evaluation"][experiment_name]["output_dir"])

    logger.info(f"Reading brain regions from {brain_regions_path}")
    with open(brain_regions_path, "r") as f:
        brain_regions_json = json.load(f)
    region_meta = RegionMeta.from_root_region(brain_regions_json["msg"][0])

    logger.info(f"Reading CCFv3 atlas from {ccf_v3_merged_path}")
    ccf_v3_merged, header_v3 = nrrd.read(ccf_v3_merged_path)

    logger.info("Unfurling CCFv3")
    ccf_v3_unfurled = deal.atlas.unfurl_regions(ccf_v3_merged, region_meta, tqdm)

    logger.info("Computing all misalignments")
    output_dir.mkdir(parents=True, exist_ok=True)
    for warped_file in pathlib.Path(warped_dir).glob("*.nrrd"):
        output_file = output_dir / f"{warped_file.stem}.txt"
        if output_file.exists():
            logger.info(f"The file {output_file} already exists, skipping")
            continue

        logger.info(f"Loading {warped_file}")
        warped_atlas, header_warped = nrrd.read(str(warped_file))

        logger.info(f"Unfurling {warped_file}")
        warped_atlas_unfurled = deal.atlas.unfurl_regions(
            warped_atlas, region_meta, tqdm
        )

        with output_file.open("w") as f:
            print_misalignments(ccf_v3_unfurled, warped_atlas_unfurled, file=f)

    logger.info("Finished")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
