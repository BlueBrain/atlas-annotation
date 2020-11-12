"""Dimitri's atlas merge script."""
import argparse
import json
import logging
import pathlib
import sys

import nrrd
import numpy as np
import toml

import deal.atlas.merge as utils

logger = logging.getLogger("v2_v3_merge")


def main(argv=None):
    """Run the main script.

    Parameters
    ----------
    argv : sequence or None
        The argument vector. If None then the arguments are parsed from
        the command line directly.
    """
    # Parse arguments and load the configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", default="config.toml")
    parser.add_argument(
        "-e",
        "--experiment-name",
        help=(
            "The name of the experiment. Usually this should be"
            "left empty, in which case the name of the script "
            "file will be used as the experiment name. This "
            "should normally automatically match the correct "
            "section in the config.toml file, unless the script "
            "file and/or the configuration section was renamed."
        ),
    )
    args = parser.parse_args(argv)
    config_file = args.config_file
    if args.experiment_name is None:
        experiment_name = pathlib.Path(__file__).stem
    else:
        experiment_name = args.experiment_name

    with open(config_file) as f:
        config_all = toml.load(f)
        config = config_all[experiment_name]

    brain_regions_path = config["brain_regions"]
    ccf_v2_path = config["atlas_v2"]
    ccf_v3_path = config["atlas_v3"]
    ccf_v2_merged_path = config["atlas_v2_merged"]
    ccf_v3_merged_path = config["atlas_v3_merged"]

    logger.info(f"Reading brain regions from {brain_regions_path}")
    with open(brain_regions_path, "r") as f:
        brain_regions_json = json.load(f)

    logger.info(f"Reading CCFv2 atlas from {ccf_v2_path}")
    atlas_v2, header_v2 = nrrd.read(ccf_v2_path)
    logger.info(f"Reading CCFv3 atlas from {ccf_v3_path}")
    atlas_v3, header_v3 = nrrd.read(ccf_v3_path)

    logger.info("Start merging atlases")
    ccfv2_corrected, ccfv3_corrected = merge_atlases(
        brain_regions=brain_regions_json["msg"][0],
        atlas_v2=atlas_v2,
        atlas_v3=atlas_v3,
    )

    logger.info(f"Writing merged v2 atlas to {ccf_v2_merged_path}")
    nrrd.write(
        filename=str(ccf_v2_merged_path),
        data=ccfv2_corrected,
        header=header_v2,
    )
    logger.info(f"Writing merged v3 atlas to {ccf_v3_merged_path}")
    nrrd.write(
        filename=str(ccf_v3_merged_path),
        data=ccfv3_corrected,
        header=header_v3,
    )


def merge_atlases(brain_regions, atlas_v2, atlas_v3):
    """Merge two atlases based on the region hierarchy.

    Parameters
    ----------
    brain_regions : dict
        The brain regions. The format should be following that of
        `http://api.brain-map.org/api/v2/structure_graph_download/1.json`
        in the "msg" field.
    atlas_v2 : np.ndarray
        The CCFv2 atlas volume.
    atlas_v3 : np.ndarray
        The CCFv3 atlas volume.

    Returns
    -------
    atlas_v2_corrected : np.ndarray
        The merged version of the CCFv2 atlas.
    atlas_v3_corrected : np.ndarray
        The merged version of the CCFv3 atlas.
    """
    logger.info("search_children")
    utils.search_children(brain_regions)

    logger.info("find_unique_regions, v2")
    uniques_v2 = utils.find_unique_regions(
        atlas_v2,
        utils.id_to_region_dictionary_ALLNAME,
        utils.region_dictionary_to_id_ALLNAME,
        utils.region_dictionary_to_id_ALLNAME_parent,
        utils.name2allname,
        top_region_name="root",
    )
    logger.info("find_children, v2")
    children_v2, _ = utils.find_children(
        uniques_v2,
        utils.id_to_region_dictionary_ALLNAME,
        utils.is_leaf,
        utils.region_dictionary_to_id_ALLNAME_parent,
        utils.region_dictionary_to_id_ALLNAME,
    )
    logger.info("find_unique_regions, v3")
    uniques_v3 = utils.find_unique_regions(
        atlas_v3,
        utils.id_to_region_dictionary_ALLNAME,
        utils.region_dictionary_to_id_ALLNAME,
        utils.region_dictionary_to_id_ALLNAME_parent,
        utils.name2allname,
        top_region_name="root",
    )
    logger.info("find_children, v3")
    children_v3, _ = utils.find_children(
        uniques_v3,
        utils.id_to_region_dictionary_ALLNAME,
        utils.is_leaf,
        utils.region_dictionary_to_id_ALLNAME_parent,
        utils.region_dictionary_to_id_ALLNAME,
    )

    logger.info("Initialize corrected atlases")
    atlas_v2_corrected = np.copy(atlas_v2)
    atlas_v3_corrected = np.copy(atlas_v3)
    ids_v2 = np.unique(atlas_v2_corrected)
    ids_v3 = np.unique(atlas_v3_corrected)
    ids_to_correct = ids_v2[np.isin(ids_v2, ids_v3, invert=True)]

    logger.info("Correction loop over ids_to_correct")
    for id_reg in ids_to_correct:
        allname = utils.id_to_region_dictionary_ALLNAME[id_reg]
        if (
            utils.is_leaf[allname]
            and id_reg not in ids_v3
            and utils.region_dictionary_to_id[
                utils.region_dictionary_to_id_parent[
                    utils.id_to_region_dictionary[id_reg]
                ]
            ]
            in ids_v3
        ):
            atlas_v2_corrected[
                atlas_v2_corrected == id_reg
            ] = utils.region_dictionary_to_id[
                utils.region_dictionary_to_id_parent[
                    utils.id_to_region_dictionary[id_reg]
                ]
            ]
        elif utils.is_leaf[allname] and (
            "Medial amygdalar nucleus" in allname
            or "Subiculum" in allname
            or "Bed nuclei of the stria terminalis" in allname
        ):
            atlas_v2_corrected[
                atlas_v2_corrected == id_reg
            ] = utils.region_dictionary_to_id[
                utils.region_dictionary_to_id_parent[
                    utils.region_dictionary_to_id_parent[
                        utils.id_to_region_dictionary[id_reg]
                    ]
                ]
            ]
        elif "Paraventricular hypothalamic nucleus" in allname:
            atlas_v2_corrected[atlas_v2_corrected == id_reg] = 38

    logger.info("Manual region mapping")
    # Entorhinal area, lateral part
    atlas_v2_corrected[np.where(atlas_v2_corrected == 60)] = 28  # L6b -> L6a
    atlas_v2_corrected[
        np.where(atlas_v2_corrected == 999)
    ] = 20  # L2/3 -> L2 # double check?
    atlas_v2_corrected[np.where(atlas_v2_corrected == 715)] = 20  # L2a -> L2
    atlas_v2_corrected[np.where(atlas_v2_corrected == 764)] = 20  # L2b -> L2
    atlas_v2_corrected[np.where(atlas_v2_corrected == 92)] = 139  # L4 -> L5
    atlas_v2_corrected[np.where(atlas_v2_corrected == 312)] = 139  # L4/5 -> L5

    # Entorhinal area, medial part, dorsal zone
    atlas_v2_corrected[np.where(atlas_v2_corrected == 468)] = 543  # L2a -> L2
    atlas_v2_corrected[np.where(atlas_v2_corrected == 508)] = 543  # L2b -> L2
    atlas_v2_corrected[
        np.where(atlas_v2_corrected == 712)
    ] = 727  # L4 -> L5 # double check?

    atlas_v2_corrected[np.where(atlas_v2_corrected == 195)] = 304  # L2 -> L2/3
    atlas_v2_corrected[np.where(atlas_v2_corrected == 524)] = 582  # L2 -> L2/3
    atlas_v2_corrected[np.where(atlas_v2_corrected == 606)] = 430  # L2 -> L2/3
    atlas_v2_corrected[np.where(atlas_v2_corrected == 747)] = 556  # L2 -> L2/3

    # subreg of Cochlear nuclei -> Cochlear nuclei
    atlas_v2_corrected[np.where(atlas_v2_corrected == 96)] = 607
    atlas_v2_corrected[np.where(atlas_v2_corrected == 101)] = 607
    atlas_v2_corrected[np.where(atlas_v2_corrected == 112)] = 607
    atlas_v2_corrected[np.where(atlas_v2_corrected == 560)] = 607
    atlas_v3_corrected[np.where(atlas_v3_corrected == 96)] = 607
    atlas_v3_corrected[np.where(atlas_v3_corrected == 101)] = 607
    # subreg of Nucleus ambiguus -> Nucleus ambiguus
    atlas_v2_corrected[np.where(atlas_v2_corrected == 143)] = 135
    atlas_v2_corrected[np.where(atlas_v2_corrected == 939)] = 135
    atlas_v3_corrected[np.where(atlas_v3_corrected == 143)] = 135
    atlas_v3_corrected[np.where(atlas_v3_corrected == 939)] = 135
    # subreg of Accessory olfactory bulb -> Accessory olfactory bulb
    atlas_v2_corrected[np.where(atlas_v2_corrected == 188)] = 151
    atlas_v2_corrected[np.where(atlas_v2_corrected == 196)] = 151
    atlas_v2_corrected[np.where(atlas_v2_corrected == 204)] = 151
    atlas_v3_corrected[np.where(atlas_v3_corrected == 188)] = 151
    atlas_v3_corrected[np.where(atlas_v3_corrected == 196)] = 151
    atlas_v3_corrected[np.where(atlas_v3_corrected == 204)] = 151
    # subreg of Medial mammillary nucleus -> Medial mammillary nucleus
    atlas_v2_corrected[np.where(atlas_v2_corrected == 798)] = 491
    atlas_v3_corrected[np.where(atlas_v3_corrected == 798)] = 491
    atlas_v3_corrected[np.where(atlas_v3_corrected == 606826647)] = 491
    atlas_v3_corrected[np.where(atlas_v3_corrected == 606826651)] = 491
    atlas_v3_corrected[np.where(atlas_v3_corrected == 606826655)] = 491
    atlas_v3_corrected[np.where(atlas_v3_corrected == 606826659)] = 491
    # Subreg to Dorsal part of the lateral geniculate complex
    atlas_v3_corrected[np.where(atlas_v3_corrected == 496345664)] = 170
    atlas_v3_corrected[np.where(atlas_v3_corrected == 496345668)] = 170
    atlas_v3_corrected[np.where(atlas_v3_corrected == 496345672)] = 170
    # Subreg to Lateral reticular nucleus
    atlas_v2_corrected[np.where(atlas_v2_corrected == 955)] = 235
    atlas_v2_corrected[np.where(atlas_v2_corrected == 963)] = 235
    atlas_v3_corrected[np.where(atlas_v3_corrected == 955)] = 235
    atlas_v3_corrected[np.where(atlas_v3_corrected == 963)] = 235

    # subreg of Posterior parietal association areas combined layer by layer
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782550)] = 532
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782604)] = 532
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782554)] = 241
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782608)] = 241
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782558)] = 635
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782612)] = 635
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782562)] = 683
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782616)] = 683
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782566)] = 308
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782620)] = 308
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782570)] = 340
    atlas_v3_corrected[np.where(atlas_v3_corrected == 312782624)] = 340

    # subreg to Parabrachial nucleus
    atlas_v2_corrected[np.where(atlas_v2_corrected == 123)] = 867
    atlas_v2_corrected[np.where(atlas_v2_corrected == 860)] = 867
    atlas_v2_corrected[np.where(atlas_v2_corrected == 868)] = 867
    atlas_v2_corrected[np.where(atlas_v2_corrected == 875)] = 867
    atlas_v2_corrected[np.where(atlas_v2_corrected == 883)] = 867
    atlas_v2_corrected[np.where(atlas_v2_corrected == 891)] = 867
    atlas_v2_corrected[np.where(atlas_v2_corrected == 899)] = 867
    atlas_v2_corrected[np.where(atlas_v2_corrected == 915)] = 867
    atlas_v3_corrected[np.where(atlas_v3_corrected == 123)] = 867

    logger.info("Some replacements by layer")
    for id_reg in np.unique(np.concatenate((ids_v2, ids_v3)))[1:]:
        allname = utils.id_to_region_dictionary_ALLNAME[id_reg]
        if "Visual areas" in allname:
            if "ayer 1" in allname:
                atlas_v3_corrected[np.where(atlas_v3_corrected == id_reg)] = 801
                atlas_v2_corrected[np.where(atlas_v2_corrected == id_reg)] = 801
            elif "ayer 2/3" in allname:
                atlas_v3_corrected[np.where(atlas_v3_corrected == id_reg)] = 561
                atlas_v2_corrected[np.where(atlas_v2_corrected == id_reg)] = 561
            elif "ayer 4" in allname:
                atlas_v3_corrected[np.where(atlas_v3_corrected == id_reg)] = 913
                atlas_v2_corrected[np.where(atlas_v2_corrected == id_reg)] = 913
            elif "ayer 5" in allname:
                atlas_v3_corrected[np.where(atlas_v3_corrected == id_reg)] = 937
                atlas_v2_corrected[np.where(atlas_v2_corrected == id_reg)] = 937
            elif "ayer 6a" in allname:
                atlas_v3_corrected[np.where(atlas_v3_corrected == id_reg)] = 457
                atlas_v2_corrected[np.where(atlas_v2_corrected == id_reg)] = 457
            elif "ayer 6b" in allname:
                atlas_v3_corrected[np.where(atlas_v3_corrected == id_reg)] = 497
                atlas_v2_corrected[np.where(atlas_v2_corrected == id_reg)] = 497

    # subreg of Prosubiculum to subiculum
    atlas_v3_corrected[np.where(atlas_v3_corrected == 484682470)] = 502
    # Orbital area, medial part, layer 6b -> 6a
    atlas_v3_corrected[np.where(atlas_v3_corrected == 527696977)] = 910
    atlas_v3_corrected[np.where(atlas_v3_corrected == 355)] = 314

    logger.info("Correction loop over ids_v3")
    for id_reg in ids_v3[1:]:
        if (
            (
                "fiber tracts" in utils.id_to_region_dictionary_ALLNAME[id_reg]
                or "Interpeduncular nucleus"
                in utils.id_to_region_dictionary_ALLNAME[id_reg]
            )
            and id_reg not in ids_v2
            and utils.region_dictionary_to_id[
                utils.region_dictionary_to_id_parent[
                    utils.id_to_region_dictionary[id_reg]
                ]
            ]
            in ids_v2
        ):
            atlas_v3_corrected[
                np.where(atlas_v3_corrected == id_reg)
            ] = utils.region_dictionary_to_id[
                utils.region_dictionary_to_id_parent[
                    utils.id_to_region_dictionary[id_reg]
                ]
            ]
        if (
            "Frontal pole, cerebral cortex"
            in utils.id_to_region_dictionary_ALLNAME[id_reg]
        ):
            atlas_v3_corrected[np.where(atlas_v3_corrected == id_reg)] = 184
            atlas_v2_corrected[np.where(atlas_v2_corrected == id_reg)] = 184

    ids_v2 = np.unique(atlas_v2_corrected)
    ids_v3 = np.unique(atlas_v3_corrected)
    ids_to_correct = ids_v2[np.isin(ids_v2, ids_v3, invert=True)]
    ids_to_correct = ids_v3[np.isin(ids_v3, ids_v2, invert=True)]
    ids_to_correct = ids_to_correct[ids_to_correct != 997]
    ids_to_correct = ids_to_correct[ids_to_correct != 8]

    logger.info("Exhaust ids_to_correct, round 1")
    while len(ids_to_correct) > 0:
        logger.info(f"{len(ids_to_correct)} left...")
        parent = ids_to_correct[0]
        while parent not in uniques_v2:
            parent = utils.region_dictionary_to_id[
                utils.region_dictionary_to_id_parent[
                    utils.id_to_region_dictionary[parent]
                ]
            ]
        for child in children_v3[utils.id_to_region_dictionary_ALLNAME[parent]]:
            atlas_v3_corrected[np.where(atlas_v3_corrected == child)] = parent
        for child in children_v2[utils.id_to_region_dictionary_ALLNAME[parent]]:
            atlas_v2_corrected[np.where(atlas_v2_corrected == child)] = parent
        ids_v2 = np.unique(atlas_v2_corrected)
        ids_v3 = np.unique(atlas_v3_corrected)
        ids_to_correct = ids_v3[np.isin(ids_v3, ids_v2, invert=True)]
        ids_to_correct = ids_to_correct[ids_to_correct != 997]
        ids_to_correct = ids_to_correct[ids_to_correct != 8]

    ids_to_correct = ids_v2[np.isin(ids_v2, ids_v3, invert=True)]
    ids_to_correct = ids_to_correct[ids_to_correct != 997]
    ids_to_correct = ids_to_correct[ids_to_correct != 8]

    logger.info("Exhaust ids_to_correct, round 2")
    while len(ids_to_correct) > 0:
        logger.info(f"{len(ids_to_correct)} left...")
        parent = ids_to_correct[0]
        while parent not in uniques_v3:
            parent = utils.region_dictionary_to_id[
                utils.region_dictionary_to_id_parent[
                    utils.id_to_region_dictionary[parent]
                ]
            ]
        for child in children_v3[utils.id_to_region_dictionary_ALLNAME[parent]]:
            atlas_v3_corrected[np.where(atlas_v3_corrected == child)] = parent
        for child in children_v2[utils.id_to_region_dictionary_ALLNAME[parent]]:
            atlas_v2_corrected[np.where(atlas_v2_corrected == child)] = parent
        ids_v2 = np.unique(atlas_v2_corrected)
        ids_v3 = np.unique(atlas_v3_corrected)
        ids_to_correct = ids_v2[np.isin(ids_v2, ids_v3, invert=True)]
        ids_to_correct = ids_to_correct[ids_to_correct != 997]
        ids_to_correct = ids_to_correct[ids_to_correct != 8]

    return atlas_v2_corrected, atlas_v3_corrected


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
