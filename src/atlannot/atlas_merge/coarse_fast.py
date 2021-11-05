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

logger = logging.getLogger(__name__)


def replace(ids, old_id, new_id):
    ids[ids == old_id] = new_id


def manual_relabel(ids_v2, ids_v3):
    # Entorhinal area, lateral part
    replace(ids_v2, 60, 28)  # L6b -> L6a
    replace(ids_v2, 999, 20)  # L2/3 -> L2 # double check?
    replace(ids_v2, 715, 20)  # L2a -> L2
    replace(ids_v2, 764, 20)  # L2b -> L2
    replace(ids_v2, 92, 139)  # L4 -> L5
    replace(ids_v2, 312, 139)  # L4/5 -> L5

    # Entorhinal area, medial part, dorsal zone
    replace(ids_v2, 468, 543)  # L2a -> L2
    replace(ids_v2, 508, 543)  # L2b -> L2
    replace(ids_v2, 712, 727)  # L4 -> L5 # double check?

    replace(ids_v2, 195, 304)  # L2 -> L2/3
    replace(ids_v2, 524, 582)  # L2 -> L2/3
    replace(ids_v2, 606, 430)  # L2 -> L2/3
    replace(ids_v2, 747, 556)  # L2 -> L2/3

    # subreg of Cochlear nuclei -> Cochlear nuclei
    replace(ids_v2, 96, 607)
    replace(ids_v2, 101, 607)
    replace(ids_v2, 112, 607)
    replace(ids_v2, 560, 607)
    replace(ids_v3, 96, 607)
    replace(ids_v3, 101, 607)
    # subreg of Nucleus ambiguus -> Nucleus ambiguus
    replace(ids_v2, 143, 135)
    replace(ids_v2, 939, 135)
    replace(ids_v3, 143, 135)
    replace(ids_v3, 939, 135)
    # subreg of Accessory olfactory bulb -> Accessory olfactory bulb
    replace(ids_v2, 188, 151)
    replace(ids_v2, 196, 151)
    replace(ids_v2, 204, 151)
    replace(ids_v3, 188, 151)
    replace(ids_v3, 196, 151)
    replace(ids_v3, 204, 151)
    # subreg of Medial mammillary nucleus -> Medial mammillary nucleus
    replace(ids_v2, 798, 491)
    replace(ids_v3, 798, 491)
    replace(ids_v3, 606826647, 491)
    replace(ids_v3, 606826651, 491)
    replace(ids_v3, 606826655, 491)
    replace(ids_v3, 606826659, 491)
    # Subreg to Dorsal part of the lateral geniculate complex
    replace(ids_v3, 496345664, 170)
    replace(ids_v3, 496345668, 170)
    replace(ids_v3, 496345672, 170)
    # Subreg to Lateral reticular nucleus
    replace(ids_v2, 955, 235)
    replace(ids_v2, 963, 235)
    replace(ids_v3, 955, 235)
    replace(ids_v3, 963, 235)

    # subreg of Posterior parietal association areas combined layer by layer
    replace(ids_v3, 312782550, 532)
    replace(ids_v3, 312782604, 532)
    replace(ids_v3, 312782554, 241)
    replace(ids_v3, 312782608, 241)
    replace(ids_v3, 312782558, 635)
    replace(ids_v3, 312782612, 635)
    replace(ids_v3, 312782562, 683)
    replace(ids_v3, 312782616, 683)
    replace(ids_v3, 312782566, 308)
    replace(ids_v3, 312782620, 308)
    replace(ids_v3, 312782570, 340)
    replace(ids_v3, 312782624, 340)

    # subreg to Parabrachial nucleus
    replace(ids_v2, 123, 867)
    replace(ids_v2, 860, 867)
    replace(ids_v2, 868, 867)
    replace(ids_v2, 875, 867)
    replace(ids_v2, 883, 867)
    replace(ids_v2, 891, 867)
    replace(ids_v2, 899, 867)
    replace(ids_v2, 915, 867)
    replace(ids_v3, 123, 867)


def atlas_remap(atlas, values_from, values_to):
    """Remap atlas values fast.

    This only works if

    * ``values_from`` contains all unique values in ``atlas``,
    * ``values_from`` is sorted.

    In other words, it must be that ``values_from = np.unique(atlas)``.

    Source: https://stackoverflow.com/a/35464758/2804645

    Parameters
    ----------
    atlas : np.ndarray
        The atlas volume to remap. Can be of any shape.
    values_from : np.ndarray
        The values to map from. It must be that
        ``values_from = np.unique(atlas)``.
    values_to : np.ndarray
        The values to map to. Must have the same shape as ``values_from``.

    Returns
    -------
    np.ndarray
        The remapped atlas.
    """
    idx = np.searchsorted(values_from, atlas.ravel())
    new_atlas = values_to[idx].reshape(atlas.shape)

    return new_atlas


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
    region_meta = RegionMeta.from_root_region(brain_regions)

    def get_allname(region_id):
        x = ""
        while region_id != region_meta.background_id:
            x = f"|{region_meta.name[region_id]}" + x
            region_id = region_meta.parent_id[region_id]
        return x

    def is_leaf(region_id):
        # leaf = not parent of anyone
        return region_id not in region_meta.parent_id.values()

    def parent(region_id):
        """Get the parent region ID of a region."""
        return region_meta.parent_id.get(region_id)

    def children(region_id):
        """Get all child region IDs of a given region."""
        for child_id, parent_id in region_meta.parent_id.items():
            if parent_id == region_id:
                yield child_id

    def descendants(region_id, allowed_ids):
        """Get all filtered descendant IDs of a given region ID.

        A descendant is only accepted if it's in ``allowed_ids`` or is a
        leaf region.

        This is mimicking Dimitri's algorithm, I'm not sure about why this must
        be that way.
        """
        all_descendants = set()
        for child_id in children(region_id):
            if child_id in allowed_ids or is_leaf(child_id):
                all_descendants.add(child_id)

            all_descendants |= descendants(child_id, allowed_ids)

        return all_descendants

    logger.info("Preparing region ID maps")
    v2_from = np.unique(ccfv2)
    v2_to = v2_from.copy()
    v3_from = np.unique(ccfv3)
    v3_to = v3_from.copy()

    logger.info("First loop")
    ids_v2 = set(v2_to)
    ids_v3 = set(v3_to)
    ids_to_correct = ids_v2 - ids_v3
    for id_reg in ids_to_correct:
        allname = get_allname(id_reg)
        if is_leaf(id_reg) and id_reg not in ids_v3 and parent(id_reg) in ids_v3:
            replace(v2_to, id_reg, parent(id_reg))
        elif is_leaf(id_reg) and (
            "Medial amygdalar nucleus" in allname
            or "Subiculum" in allname
            or "Bed nuclei of the stria terminalis" in allname
        ):
            replace(v2_to, id_reg, parent(parent(id_reg)))
        elif "Paraventricular hypothalamic nucleus" in allname:
            replace(v3_to, id_reg, 38)

    logger.info("Manual replacements")
    manual_relabel(v2_to, v3_to)

    logger.info("Second loop")
    for id_reg in (ids_v2 | ids_v3) - {0}:
        allname = get_allname(id_reg)
        if "Visual areas" in allname:
            if "ayer 1" in allname:
                replace(v3_to, id_reg, 801)
                replace(v2_to, id_reg, 801)
            elif "ayer 2/3" in allname:
                replace(v3_to, id_reg, 561)
                replace(v2_to, id_reg, 561)
            elif "ayer 4" in allname:
                replace(v3_to, id_reg, 913)
                replace(v2_to, id_reg, 913)
            elif "ayer 5" in allname:
                replace(v3_to, id_reg, 937)
                replace(v2_to, id_reg, 937)
            elif "ayer 6a" in allname:
                replace(v3_to, id_reg, 457)
                replace(v2_to, id_reg, 457)
            elif "ayer 6b" in allname:
                replace(v3_to, id_reg, 497)
                replace(v2_to, id_reg, 497)

    logger.info("Manual replacements #2")
    # subreg of Prosubiculum to subiculum
    replace(v3_to, 484682470, 502)
    # Orbital area, medial part, layer 6b -> 6a
    replace(v3_to, 527696977, 910)
    replace(v3_to, 355, 314)

    logger.info("Third loop")
    for id_reg in ids_v3 - {0}:
        if (
            (
                "fiber tracts" in get_allname(id_reg)
                or "Interpeduncular nucleus" in get_allname(id_reg)
            )
            and id_reg not in ids_v2
            and parent(id_reg) in ids_v2
        ):
            replace(v3_to, id_reg, parent(id_reg))
        if "Frontal pole, cerebral cortex" in get_allname(id_reg):
            replace(v3_to, id_reg, 184)
            replace(v2_to, id_reg, 184)

    logger.info("Some manual stuff again")
    ids_v2 = set(v2_to)
    ids_v3 = set(v3_to)

    logger.info("Computing unique region IDs")
    uniques_v2 = region_meta.collect_ancestors(ids_v2)
    uniques_v3 = region_meta.collect_ancestors(ids_v3)

    logger.info("While loop 1")
    ids_to_correct = ids_v3 - ids_v2 - {8, 997}
    while len(ids_to_correct) > 0:
        id_ = ids_to_correct.pop()
        while id_ not in uniques_v2:
            id_ = parent(id_)
        for child in descendants(id_, uniques_v3):
            replace(v3_to, child, id_)
        for child in descendants(id_, uniques_v2):
            replace(v2_to, child, id_)
        ids_v2 = set(v2_to)
        ids_v3 = set(v3_to)
        ids_to_correct = ids_v3 - ids_v2 - {8, 997}

    logger.info("While loop 2")
    ids_to_correct = ids_v2 - ids_v3 - {8, 997}
    while len(ids_to_correct) > 0:
        id_ = ids_to_correct.pop()
        while id_ not in uniques_v3:
            id_ = parent(id_)
        for child in descendants(id_, uniques_v3):
            replace(v3_to, child, id_)
        for child in descendants(id_, uniques_v2):
            replace(v2_to, child, id_)
        ids_v2 = set(v2_to)
        ids_v3 = set(v3_to)
        ids_to_correct = ids_v2 - ids_v3 - {8, 997}

    logger.info("Applying replacements")
    ccfv2_new = atlas_remap(ccfv2, v2_from, v2_to)
    ccfv3_new = atlas_remap(ccfv3, v3_from, v3_to)

    return ccfv2_new, ccfv3_new
