"""The coarse merging of the annotation atlases."""
import numpy as np

from .JSONread import RegionData


def coarse_merge(annotation, annotation2, brain_regions):
    """Perform the coarse atlas merging.

    Parameters
    ----------
    annotation : np.ndarray
        The first atlas to merge, usually CCFv2.
    annotation2 : np.ndarray
        The second atlas to merge, usually CCFv3.
    brain_regions : dict
        The brain regions dictionary. Can be obtained from the "msg" key of
        the `brain_regions.json` (`1.json`) file.

    Returns
    -------
    ccfv2_corrected : np.ndarray
        The merged CCFv2 atlas.
    ccfv3_corrected : np.ndarray
        The merged CCFv3 atlas.
    """
    region_data = RegionData(brain_regions)

    uniques = region_data.find_unique_regions(annotation, top_region_name="root")
    children, _ = region_data.find_children(uniques)
    uniques2 = region_data.find_unique_regions(annotation2, top_region_name="root")
    children2, _ = region_data.find_children(uniques2)

    ccfv2_corrected = np.copy(annotation)
    ccfv3_corrected = np.copy(annotation2)
    ids = np.unique(ccfv2_corrected)
    ids2 = np.unique(ccfv3_corrected)
    ids_to_correct = ids[np.in1d(ids, ids2, invert=True)]

    for id_reg in ids_to_correct:
        allname = region_data.id_to_region_dictionary_ALLNAME[id_reg]
        if (
            region_data.is_leaf[allname]
            and id_reg not in ids2
            and region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[id_reg]
                ]
            ]
            in ids2
        ):
            ccfv2_corrected[
                ccfv2_corrected == id_reg
            ] = region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[id_reg]
                ]
            ]
        elif region_data.is_leaf[allname] and (
            "Medial amygdalar nucleus" in allname
            or "Subiculum" in allname
            or "Bed nuclei of the stria terminalis" in allname
        ):
            ccfv2_corrected[
                ccfv2_corrected == id_reg
            ] = region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.region_dictionary_to_id_parent[
                        region_data.id_to_region_dictionary[id_reg]
                    ]
                ]
            ]
        elif "Paraventricular hypothalamic nucleus" in allname:
            ccfv2_corrected[ccfv2_corrected == id_reg] = 38

    # Entorhinal area, lateral part
    ccfv2_corrected[np.where(ccfv2_corrected == 60)] = 28  # L6b -> L6a
    ccfv2_corrected[np.where(ccfv2_corrected == 999)] = 20  # L2/3 -> L2 # double check?
    ccfv2_corrected[np.where(ccfv2_corrected == 715)] = 20  # L2a -> L2
    ccfv2_corrected[np.where(ccfv2_corrected == 764)] = 20  # L2b -> L2
    ccfv2_corrected[np.where(ccfv2_corrected == 92)] = 139  # L4 -> L5
    ccfv2_corrected[np.where(ccfv2_corrected == 312)] = 139  # L4/5 -> L5

    # Entorhinal area, medial part, dorsal zone
    ccfv2_corrected[np.where(ccfv2_corrected == 468)] = 543  # L2a -> L2
    ccfv2_corrected[np.where(ccfv2_corrected == 508)] = 543  # L2b -> L2
    ccfv2_corrected[np.where(ccfv2_corrected == 712)] = 727  # L4 -> L5 # double check?

    ccfv2_corrected[np.where(ccfv2_corrected == 195)] = 304  # L2 -> L2/3
    ccfv2_corrected[np.where(ccfv2_corrected == 524)] = 582  # L2 -> L2/3
    ccfv2_corrected[np.where(ccfv2_corrected == 606)] = 430  # L2 -> L2/3
    ccfv2_corrected[np.where(ccfv2_corrected == 747)] = 556  # L2 -> L2/3

    # subreg of Cochlear nuclei -> Cochlear nuclei
    ccfv2_corrected[np.where(ccfv2_corrected == 96)] = 607
    ccfv2_corrected[np.where(ccfv2_corrected == 101)] = 607
    ccfv2_corrected[np.where(ccfv2_corrected == 112)] = 607
    ccfv2_corrected[np.where(ccfv2_corrected == 560)] = 607
    ccfv3_corrected[np.where(ccfv3_corrected == 96)] = 607
    ccfv3_corrected[np.where(ccfv3_corrected == 101)] = 607
    # subreg of Nucleus ambiguus -> Nucleus ambiguus
    ccfv2_corrected[np.where(ccfv2_corrected == 143)] = 135
    ccfv2_corrected[np.where(ccfv2_corrected == 939)] = 135
    ccfv3_corrected[np.where(ccfv3_corrected == 143)] = 135
    ccfv3_corrected[np.where(ccfv3_corrected == 939)] = 135
    # subreg of Accessory olfactory bulb -> Accessory olfactory bulb
    ccfv2_corrected[np.where(ccfv2_corrected == 188)] = 151
    ccfv2_corrected[np.where(ccfv2_corrected == 196)] = 151
    ccfv2_corrected[np.where(ccfv2_corrected == 204)] = 151
    ccfv3_corrected[np.where(ccfv3_corrected == 188)] = 151
    ccfv3_corrected[np.where(ccfv3_corrected == 196)] = 151
    ccfv3_corrected[np.where(ccfv3_corrected == 204)] = 151
    # subreg of Medial mammillary nucleus -> Medial mammillary nucleus
    ccfv2_corrected[np.where(ccfv2_corrected == 798)] = 491
    ccfv3_corrected[np.where(ccfv3_corrected == 798)] = 491
    ccfv3_corrected[np.where(ccfv3_corrected == 606826647)] = 491
    ccfv3_corrected[np.where(ccfv3_corrected == 606826651)] = 491
    ccfv3_corrected[np.where(ccfv3_corrected == 606826655)] = 491
    ccfv3_corrected[np.where(ccfv3_corrected == 606826659)] = 491
    # Subreg to Dorsal part of the lateral geniculate complex
    ccfv3_corrected[np.where(ccfv3_corrected == 496345664)] = 170
    ccfv3_corrected[np.where(ccfv3_corrected == 496345668)] = 170
    ccfv3_corrected[np.where(ccfv3_corrected == 496345672)] = 170
    # Subreg to Lateral reticular nucleus
    ccfv2_corrected[np.where(ccfv2_corrected == 955)] = 235
    ccfv2_corrected[np.where(ccfv2_corrected == 963)] = 235
    ccfv3_corrected[np.where(ccfv3_corrected == 955)] = 235
    ccfv3_corrected[np.where(ccfv3_corrected == 963)] = 235

    # subreg of Posterior parietal association areas combined layer by layer
    ccfv3_corrected[np.where(ccfv3_corrected == 312782550)] = 532
    ccfv3_corrected[np.where(ccfv3_corrected == 312782604)] = 532
    ccfv3_corrected[np.where(ccfv3_corrected == 312782554)] = 241
    ccfv3_corrected[np.where(ccfv3_corrected == 312782608)] = 241
    ccfv3_corrected[np.where(ccfv3_corrected == 312782558)] = 635
    ccfv3_corrected[np.where(ccfv3_corrected == 312782612)] = 635
    ccfv3_corrected[np.where(ccfv3_corrected == 312782562)] = 683
    ccfv3_corrected[np.where(ccfv3_corrected == 312782616)] = 683
    ccfv3_corrected[np.where(ccfv3_corrected == 312782566)] = 308
    ccfv3_corrected[np.where(ccfv3_corrected == 312782620)] = 308
    ccfv3_corrected[np.where(ccfv3_corrected == 312782570)] = 340
    ccfv3_corrected[np.where(ccfv3_corrected == 312782624)] = 340

    # subreg to Parabrachial nucleus
    ccfv2_corrected[np.where(ccfv2_corrected == 123)] = 867
    ccfv2_corrected[np.where(ccfv2_corrected == 860)] = 867
    ccfv2_corrected[np.where(ccfv2_corrected == 868)] = 867
    ccfv2_corrected[np.where(ccfv2_corrected == 875)] = 867
    ccfv2_corrected[np.where(ccfv2_corrected == 883)] = 867
    ccfv2_corrected[np.where(ccfv2_corrected == 891)] = 867
    ccfv2_corrected[np.where(ccfv2_corrected == 899)] = 867
    ccfv2_corrected[np.where(ccfv2_corrected == 915)] = 867
    ccfv3_corrected[np.where(ccfv3_corrected == 123)] = 867

    for id_reg in np.unique(np.concatenate((ids, ids2)))[1:]:
        allname = region_data.id_to_region_dictionary_ALLNAME[id_reg]
        if "Visual areas" in allname:
            if "ayer 1" in allname:
                ccfv3_corrected[np.where(ccfv3_corrected == id_reg)] = 801
                ccfv2_corrected[np.where(ccfv2_corrected == id_reg)] = 801
            elif "ayer 2/3" in allname:
                ccfv3_corrected[np.where(ccfv3_corrected == id_reg)] = 561
                ccfv2_corrected[np.where(ccfv2_corrected == id_reg)] = 561
            elif "ayer 4" in allname:
                ccfv3_corrected[np.where(ccfv3_corrected == id_reg)] = 913
                ccfv2_corrected[np.where(ccfv2_corrected == id_reg)] = 913
            elif "ayer 5" in allname:
                ccfv3_corrected[np.where(ccfv3_corrected == id_reg)] = 937
                ccfv2_corrected[np.where(ccfv2_corrected == id_reg)] = 937
            elif "ayer 6a" in allname:
                ccfv3_corrected[np.where(ccfv3_corrected == id_reg)] = 457
                ccfv2_corrected[np.where(ccfv2_corrected == id_reg)] = 457
            elif "ayer 6b" in allname:
                ccfv3_corrected[np.where(ccfv3_corrected == id_reg)] = 497
                ccfv2_corrected[np.where(ccfv2_corrected == id_reg)] = 497

    # subreg of Prosubiculum to subiculum
    ccfv3_corrected[np.where(ccfv3_corrected == 484682470)] = 502
    # Orbital area, medial part, layer 6b -> 6a
    ccfv3_corrected[np.where(ccfv3_corrected == 527696977)] = 910
    ccfv3_corrected[np.where(ccfv3_corrected == 355)] = 314

    for id_reg in ids2[1:]:
        if (
            (
                "fiber tracts" in region_data.id_to_region_dictionary_ALLNAME[id_reg]
                or "Interpeduncular nucleus"
                in region_data.id_to_region_dictionary_ALLNAME[id_reg]
            )
            and id_reg not in ids
            and region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[id_reg]
                ]
            ]
            in ids
        ):
            ccfv3_corrected[
                np.where(ccfv3_corrected == id_reg)
            ] = region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[id_reg]
                ]
            ]
        if (
            "Frontal pole, cerebral cortex"
            in region_data.id_to_region_dictionary_ALLNAME[id_reg]
        ):
            ccfv3_corrected[np.where(ccfv3_corrected == id_reg)] = 184
            ccfv2_corrected[np.where(ccfv2_corrected == id_reg)] = 184

    ids = np.unique(ccfv2_corrected)
    ids2 = np.unique(ccfv3_corrected)
    ids_to_correct = ids[np.in1d(ids, ids2, invert=True)]

    ids_to_correct = ids2[np.in1d(ids2, ids, invert=True)]

    ids_to_correct = ids_to_correct[ids_to_correct != 997]
    ids_to_correct = ids_to_correct[ids_to_correct != 8]

    while len(ids_to_correct) > 0:
        parent = ids_to_correct[0]
        while parent not in uniques:
            parent = region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[parent]
                ]
            ]
        for child in children2[region_data.id_to_region_dictionary_ALLNAME[parent]]:
            ccfv3_corrected[np.where(ccfv3_corrected == child)] = parent
        for child in children[region_data.id_to_region_dictionary_ALLNAME[parent]]:
            ccfv2_corrected[np.where(ccfv2_corrected == child)] = parent
        ids = np.unique(ccfv2_corrected)
        ids2 = np.unique(ccfv3_corrected)
        ids_to_correct = ids2[np.in1d(ids2, ids, invert=True)]
        ids_to_correct = ids_to_correct[ids_to_correct != 997]
        ids_to_correct = ids_to_correct[ids_to_correct != 8]

    ids_to_correct = ids[np.in1d(ids, ids2, invert=True)]
    ids_to_correct = ids_to_correct[ids_to_correct != 997]
    ids_to_correct = ids_to_correct[ids_to_correct != 8]
    while len(ids_to_correct) > 0:
        parent = ids_to_correct[0]
        while parent not in uniques2:
            parent = region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[parent]
                ]
            ]
        for child in children2[region_data.id_to_region_dictionary_ALLNAME[parent]]:
            ccfv3_corrected[np.where(ccfv3_corrected == child)] = parent
        for child in children[region_data.id_to_region_dictionary_ALLNAME[parent]]:
            ccfv2_corrected[np.where(ccfv2_corrected == child)] = parent
        ids = np.unique(ccfv2_corrected)
        ids2 = np.unique(ccfv3_corrected)
        ids_to_correct = ids[np.in1d(ids, ids2, invert=True)]
        ids_to_correct = ids_to_correct[ids_to_correct != 997]
        ids_to_correct = ids_to_correct[ids_to_correct != 8]

    return ccfv2_corrected, ccfv3_corrected
