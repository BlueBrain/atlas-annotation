"""The fine merging of the annotation atlases."""
import numpy as np

from .JSONread import RegionData


def explore_voxel(origin, data, count=-1):
    """Explore a given voxel.

    Ask Dimitri for more details.

    Parameters
    ----------
    origin : sequence
        A triplet with the (x, y, z) coordinates of the origin voxel.
    data : np.ndarray
        A 3D array with the volume data.
    count : int
        Maximal number of iterations.

    Returns
    -------
    value : int
        The value of some voxel in the data volume.
    """
    origin_value = data[origin[0], origin[1], origin[2]]
    explored = np.zeros(data.shape, dtype=bool)
    explored[origin[0], origin[1], origin[2]] = True
    to_explore = [origin]
    maxx = len(explored)
    maxy = len(explored[0])
    maxz = len(explored[0][0])
    while len(to_explore) > 0 and count != 0:
        current_voxel = to_explore[0]
        current_value = data[current_voxel[0], current_voxel[1], current_voxel[2]]
        if current_value != origin_value and current_value != 0:
            return current_value
        to_explore = to_explore[1:]
        for (x, y, z) in [
            (-1, 0, 0),
            (0, -1, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]:
            new_vox = [current_voxel[0] + x, current_voxel[1] + y, current_voxel[2] + z]
            if (
                0 <= new_vox[0] < maxx
                and 0 <= new_vox[1] < maxy
                and 0 <= new_vox[2] < maxz
                and not explored[new_vox[0], new_vox[1], new_vox[2]]
            ):
                explored[new_vox[0], new_vox[1], new_vox[2]] = True
                to_explore.append(new_vox)
        count -= 1
    # print("Error", origin)
    return origin_value


def fine_merge(annotation, annotation2, brain_regions):
    """Perform the coarse atlas merging.

    Parameters
    ----------
    annotation : np.ndarray
        The first atlas to merge, usually CCFv2
    annotation2 : np.ndarray
        The second atlas to merge, usually CCFv3
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

    # Hippocampus Field CA2 is strongly different -> merge it with CA1
    ccfv2_corrected[ccfv2_corrected == 423] = 382
    ccfv3_corrected[ccfv3_corrected == 423] = 382

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

    # Frontal pole children to their parent
    ccfv3_corrected[ccfv3_corrected == 68] = 184
    ccfv3_corrected[ccfv3_corrected == 667] = 184
    ccfv3_corrected[ccfv3_corrected == 526157192] = 184
    ccfv3_corrected[ccfv3_corrected == 526157196] = 184
    ccfv3_corrected[ccfv3_corrected == 526322264] = 184
    ccfv2_corrected[ccfv2_corrected == 68] = 184
    ccfv2_corrected[ccfv2_corrected == 667] = 184

    # Every region ventral to cortex transition area is merge because
    # Entorhinal area, medial part, ventral zone is not in ccfv3
    # ie 259,  324,  371, 1133, 655, 663, 780 -> 663
    ccfv2_corrected[np.where(ccfv2_corrected == 259)] = 663
    ccfv2_corrected[np.where(ccfv2_corrected == 324)] = 663
    ccfv2_corrected[np.where(ccfv2_corrected == 371)] = 663
    ccfv2_corrected[np.where(ccfv2_corrected == 1133)] = 663

    ccfv2_corrected[np.where(ccfv2_corrected == 655)] = 663
    ccfv2_corrected[np.where(ccfv2_corrected == 780)] = 663
    ccfv3_corrected[np.where(ccfv3_corrected == 655)] = 663
    ccfv3_corrected[np.where(ccfv3_corrected == 780)] = 663

    # Medial terminal nucleus of the accessory optic tract -> Ventral tegmental area

    # Correct annotation edge for ccfv2 and ccfv3
    # no limit for striatum
    id_reg = 278
    filter_ = region_data.filter_region(
        ccfv2_corrected,
        region_data.id_to_region_dictionary_ALLNAME[id_reg],
        children,
    )
    copy_filt = np.copy(ccfv2_corrected)
    copy_filt[~filter_] = 0
    error_voxel = np.where(copy_filt == id_reg)
    for (x, y, z) in zip(*error_voxel):
        ccfv2_corrected[x, y, z] = explore_voxel([x, y, z], copy_filt, -1)

    for id_reg in (803, 477):
        filter_ = region_data.filter_region(
            ccfv3_corrected,
            region_data.id_to_region_dictionary_ALLNAME[id_reg],
            children,
        )
        copy_filt = np.copy(ccfv3_corrected)
        copy_filt[~filter_] = 0
        error_voxel = np.where(copy_filt == id_reg)
        for (x, y, z) in zip(*error_voxel):
            ccfv3_corrected[x, y, z] = explore_voxel([x, y, z], copy_filt, -1)

    # Correct ccfv2 annotation edge Cerebral cortex, Basic Cell group and
    # regions and root  1089, 688, 8, 997
    for id_reg in (688, 8, 997):
        filter_ = region_data.filter_region(
            ccfv2_corrected,
            region_data.id_to_region_dictionary_ALLNAME[id_reg],
            children,
        )
        copy_filt = np.copy(ccfv2_corrected)
        copy_filt[~filter_] = 0
        error_voxel = np.where(copy_filt == id_reg)
        for (x, y, z) in zip(*error_voxel):
            ccfv2_corrected[x, y, z] = explore_voxel([x, y, z], copy_filt, 3)

    # Correct ccfv3 annotation edge for Hippocampal formation, Cortical subplate
    for id_reg in (1089, 703):
        filter_ = region_data.filter_region(
            ccfv3_corrected,
            region_data.id_to_region_dictionary_ALLNAME[id_reg],
            children,
        )
        copy_filt = np.copy(ccfv3_corrected)
        copy_filt[~filter_] = 0
        error_voxel = np.where(copy_filt == id_reg)
        for (x, y, z) in zip(*error_voxel):
            ccfv3_corrected[x, y, z] = explore_voxel([x, y, z], copy_filt, 3)

    for id_main in [795]:
        for id_reg in children[region_data.id_to_region_dictionary_ALLNAME[id_main]]:
            if id_reg in uniques:
                ccfv2_corrected[ccfv2_corrected == id_reg] = id_main
            if id_reg in uniques2:
                ccfv3_corrected[ccfv3_corrected == id_reg] = id_main

    ids = np.unique(ccfv2_corrected)
    ids2 = np.unique(ccfv3_corrected)
    for id_reg in ids[1:]:
        allname = region_data.id_to_region_dictionary_ALLNAME[id_reg]
        if "fiber tracts" in allname:
            ccfv2_corrected[ccfv2_corrected == id_reg] = 1009
        elif "ventricular systems" in allname:
            ccfv2_corrected[ccfv2_corrected == id_reg] = 997
    for id_reg in ids2[1:]:
        allname = region_data.id_to_region_dictionary_ALLNAME[id_reg]
        if "fiber tracts" in allname:
            ccfv3_corrected[ccfv3_corrected == id_reg] = 1009
        elif "ventricular systems" in allname:
            ccfv3_corrected[ccfv3_corrected == id_reg] = 997

    ids = np.unique(ccfv2_corrected)
    ids2 = np.unique(ccfv3_corrected)
    ids_to_correct = ids[np.in1d(ids, ids2, invert=True)]

    ids_to_correct = ids2[np.in1d(ids2, ids, invert=True)]

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

    return ccfv2_corrected, ccfv3_corrected
