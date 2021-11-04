# Copyright 2021, Blue Brain Project, EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The coarse merging of the annotation atlases."""
import logging

import numpy as np

from atlannot.atlas.region_meta import RegionMeta
from atlannot.atlas_merge.JSONread import RegionData

logger = logging.getLogger(__name__)


def replace_label(atlas, old_value, new_value):
    atlas[atlas == old_value] = new_value


def manual_relabel(ccfv2_corrected, ccfv3_corrected):
    # Entorhinal area, lateral part
    replace_label(ccfv2_corrected, 60, 28)  # L6b -> L6a
    replace_label(ccfv2_corrected, 999, 20)  # L2/3 -> L2 # double check?
    replace_label(ccfv2_corrected, 715, 20)  # L2a -> L2
    replace_label(ccfv2_corrected, 764, 20)  # L2b -> L2
    replace_label(ccfv2_corrected, 92, 139)  # L4 -> L5
    replace_label(ccfv2_corrected, 312, 139)  # L4/5 -> L5

    # Entorhinal area, medial part, dorsal zone
    replace_label(ccfv2_corrected, 468, 543)  # L2a -> L2
    replace_label(ccfv2_corrected, 508, 543)  # L2b -> L2
    replace_label(ccfv2_corrected, 712, 727)  # L4 -> L5 # double check?

    replace_label(ccfv2_corrected, 195, 304)  # L2 -> L2/3
    replace_label(ccfv2_corrected, 524, 582)  # L2 -> L2/3
    replace_label(ccfv2_corrected, 606, 430)  # L2 -> L2/3
    replace_label(ccfv2_corrected, 747, 556)  # L2 -> L2/3

    # subreg of Cochlear nuclei -> Cochlear nuclei
    replace_label(ccfv2_corrected, 96, 607)
    replace_label(ccfv2_corrected, 101, 607)
    replace_label(ccfv2_corrected, 112, 607)
    replace_label(ccfv2_corrected, 560, 607)
    replace_label(ccfv3_corrected, 96, 607)
    replace_label(ccfv3_corrected, 101, 607)
    # subreg of Nucleus ambiguus -> Nucleus ambiguus
    replace_label(ccfv2_corrected, 143, 135)
    replace_label(ccfv2_corrected, 939, 135)
    replace_label(ccfv3_corrected, 143, 135)
    replace_label(ccfv3_corrected, 939, 135)
    # subreg of Accessory olfactory bulb -> Accessory olfactory bulb
    replace_label(ccfv2_corrected, 188, 151)
    replace_label(ccfv2_corrected, 196, 151)
    replace_label(ccfv2_corrected, 204, 151)
    replace_label(ccfv3_corrected, 188, 151)
    replace_label(ccfv3_corrected, 196, 151)
    replace_label(ccfv3_corrected, 204, 151)
    # subreg of Medial mammillary nucleus -> Medial mammillary nucleus
    replace_label(ccfv2_corrected, 798, 491)
    replace_label(ccfv3_corrected, 798, 491)
    replace_label(ccfv3_corrected, 606826647, 491)
    replace_label(ccfv3_corrected, 606826651, 491)
    replace_label(ccfv3_corrected, 606826655, 491)
    replace_label(ccfv3_corrected, 606826659, 491)
    # Subreg to Dorsal part of the lateral geniculate complex
    replace_label(ccfv3_corrected, 496345664, 170)
    replace_label(ccfv3_corrected, 496345668, 170)
    replace_label(ccfv3_corrected, 496345672, 170)
    # Subreg to Lateral reticular nucleus
    replace_label(ccfv2_corrected, 955, 235)
    replace_label(ccfv2_corrected, 963, 235)
    replace_label(ccfv3_corrected, 955, 235)
    replace_label(ccfv3_corrected, 963, 235)

    # subreg of Posterior parietal association areas combined layer by layer
    replace_label(ccfv3_corrected, 312782550, 532)
    replace_label(ccfv3_corrected, 312782604, 532)
    replace_label(ccfv3_corrected, 312782554, 241)
    replace_label(ccfv3_corrected, 312782608, 241)
    replace_label(ccfv3_corrected, 312782558, 635)
    replace_label(ccfv3_corrected, 312782612, 635)
    replace_label(ccfv3_corrected, 312782562, 683)
    replace_label(ccfv3_corrected, 312782616, 683)
    replace_label(ccfv3_corrected, 312782566, 308)
    replace_label(ccfv3_corrected, 312782620, 308)
    replace_label(ccfv3_corrected, 312782570, 340)
    replace_label(ccfv3_corrected, 312782624, 340)

    # subreg to Parabrachial nucleus
    replace_label(ccfv2_corrected, 123, 867)
    replace_label(ccfv2_corrected, 860, 867)
    replace_label(ccfv2_corrected, 868, 867)
    replace_label(ccfv2_corrected, 875, 867)
    replace_label(ccfv2_corrected, 883, 867)
    replace_label(ccfv2_corrected, 891, 867)
    replace_label(ccfv2_corrected, 899, 867)
    replace_label(ccfv2_corrected, 915, 867)
    replace_label(ccfv3_corrected, 123, 867)


def merge(ccfv2, ccfv3, brain_regions):
    """Perform the coarse atlas merging.

    Parameters
    ----------
    ccfv2 : np.ndarray
        The first atlas to merge, usually CCFv2.
    ccfv3 : np.ndarray
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
    logger.info("Preparing region metadata")
    region_data = RegionData(brain_regions)
    region_meta = RegionMeta.from_root_region(brain_regions)

    logger.info("Prepping new arrays")
    ccfv2 = ccfv2.copy()
    ccfv3 = ccfv3.copy()

    logger.info("Computing unique leaf region IDs")
    ids_v2 = set(np.unique(ccfv2))
    ids_v3 = set(np.unique(ccfv3))

    logger.info("First loop")
    ids_to_correct = ids_v2 - ids_v3
    for id_reg in ids_to_correct:
        allname = region_data.id_to_region_dictionary_ALLNAME[id_reg]
        if (
            region_data.is_leaf[allname]
            and id_reg not in ids_v3
            and region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[id_reg]
                ]
            ]
            in ids_v3
        ):
            ccfv2[
                ccfv2 == id_reg
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
            ccfv2[
                ccfv2 == id_reg
            ] = region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.region_dictionary_to_id_parent[
                        region_data.id_to_region_dictionary[id_reg]
                    ]
                ]
            ]
        elif "Paraventricular hypothalamic nucleus" in allname:
            replace_label(ccfv3, id_reg, 38)

    logger.info("Manual replacements")
    manual_relabel(ccfv2, ccfv3)

    logger.info("Second loop")
    for id_reg in (ids_v2 | ids_v3) - {0}:
        allname = region_data.id_to_region_dictionary_ALLNAME[id_reg]
        if "Visual areas" in allname:
            if "ayer 1" in allname:
                replace_label(ccfv3, id_reg, 801)
                replace_label(ccfv2, id_reg, 801)
            elif "ayer 2/3" in allname:
                replace_label(ccfv3, id_reg, 561)
                replace_label(ccfv2, id_reg, 561)
            elif "ayer 4" in allname:
                replace_label(ccfv3, id_reg, 913)
                replace_label(ccfv2, id_reg, 913)
            elif "ayer 5" in allname:
                replace_label(ccfv3, id_reg, 937)
                replace_label(ccfv2, id_reg, 937)
            elif "ayer 6a" in allname:
                replace_label(ccfv3, id_reg, 457)
                replace_label(ccfv2, id_reg, 457)
            elif "ayer 6b" in allname:
                replace_label(ccfv3, id_reg, 497)
                replace_label(ccfv2, id_reg, 497)

    logger.info("Manual replacements #2")
    # subreg of Prosubiculum to subiculum
    replace_label(ccfv3, 484682470, 502)
    # Orbital area, medial part, layer 6b -> 6a
    replace_label(ccfv3, 527696977, 910)
    replace_label(ccfv3, 355, 314)

    logger.info("Third loop")
    for id_reg in ids_v3 - {0}:
        if (
            (
                "fiber tracts" in region_data.id_to_region_dictionary_ALLNAME[id_reg]
                or "Interpeduncular nucleus"
                in region_data.id_to_region_dictionary_ALLNAME[id_reg]
            )
            and id_reg not in ids_v2
            and region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[id_reg]
                ]
            ]
            in ids_v2
        ):
            new_id = region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[id_reg]
                ]
            ]
            replace_label(ccfv3, id_reg, new_id)
        if (
            "Frontal pole, cerebral cortex"
            in region_data.id_to_region_dictionary_ALLNAME[id_reg]
        ):
            replace_label(ccfv3, id_reg, 184)
            replace_label(ccfv2, id_reg, 184)

    logger.info("Some manual stuff again")
    ids_v2 = set(np.unique(ccfv2))
    ids_v3 = set(np.unique(ccfv3))

    logger.info("Computing unique region IDs")
    uniques_v2 = region_meta.collect_ancestors(ids_v2)
    uniques_v3 = region_meta.collect_ancestors(ids_v3)

    logger.info("Computing children")
    children_v2, _ = region_data.find_children(np.array(sorted(uniques_v2)))
    children_v3, _ = region_data.find_children(np.array(sorted(uniques_v3)))

    logger.info("While loop 1")
    ids_to_correct = ids_v3 - ids_v2 - {8, 997}
    while len(ids_to_correct) > 0:
        parent = ids_to_correct.pop()
        while parent not in uniques_v2:
            parent = region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[parent]
                ]
            ]
        for child in children_v3[region_data.id_to_region_dictionary_ALLNAME[parent]]:
            replace_label(ccfv3, child, parent)
        for child in children_v2[region_data.id_to_region_dictionary_ALLNAME[parent]]:
            replace_label(ccfv2, child, parent)
        ids_v2 = set(np.unique(ccfv2))
        ids_v3 = set(np.unique(ccfv3))
        ids_to_correct = ids_v3 - ids_v2 - {8, 997}

    logger.info("While loop 2")
    ids_to_correct = ids_v2 - ids_v3 - {8, 997}
    while len(ids_to_correct) > 0:
        parent = ids_to_correct.pop()
        while parent not in uniques_v3:
            parent = region_data.region_dictionary_to_id[
                region_data.region_dictionary_to_id_parent[
                    region_data.id_to_region_dictionary[parent]
                ]
            ]
        for child in children_v3[region_data.id_to_region_dictionary_ALLNAME[parent]]:
            replace_label(ccfv3, child, parent)
        for child in children_v2[region_data.id_to_region_dictionary_ALLNAME[parent]]:
            replace_label(ccfv2, child, parent)
        ids_v2 = set(np.unique(ccfv2))
        ids_v3 = set(np.unique(ccfv3))
        ids_to_correct = ids_v2 - ids_v3 - {8, 997}

    return ccfv2, ccfv3
