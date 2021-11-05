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
"""The fine merging of the annotation atlases."""
import logging

import numpy as np
from numpy import ma

from atlannot.atlas.region_meta import RegionMeta
from atlannot.atlas_merge.common import atlas_remap, replace

logger = logging.getLogger(__name__)


def explore_voxel(start_pos, masked_atlas, count=-1):
    """Explore a given voxel.

    Ask Dimitri for more details.

    Parameters
    ----------
    start_pos : tuple
        A triplet with the (x, y, z) coordinates of the origin voxel.
    masked_atlas : ma.MaskedArray
        A 3D array with the volume data.
    count : int
        Maximal number of iterations.

    Returns
    -------
    value : int
        The value of some voxel in the data volume.
    """
    logger.debug("exploring voxel %s", start_pos)
    if not isinstance(start_pos, tuple):
        raise ValueError("The 'origin parameter must be a tuple (got {type(origin)})")

    start_value = masked_atlas[start_pos]
    seen = {start_pos}
    queue = [start_pos]
    while len(queue) > 0 and count != 0:
        pos = queue.pop(0)
        value = masked_atlas[pos]
        if value != start_value and value:  # "and value" means not masked
            return value
        for dx, dy, dz in [
            (-1, 0, 0),
            (0, -1, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]:
            new_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            if (0, 0, 0) <= new_pos < masked_atlas.shape and new_pos not in seen:
                seen.add(new_pos)
                queue.append(new_pos)
        count -= 1

    return start_value


def manual_relabel_1(ids_v2: np.ndarray, ids_v3: np.ndarray) -> None:
    """Perform a manual re-labeling step on the CCFv2 and CCFv3 atlases.

    The replacements were compiled by Dimitri Rodarie.

    Parameters
    ----------
    ids_v2
        The (unique) region IDs of the CCFv2 atlas.
    ids_v3
        The (unique) region IDs of the CCFv3 atlas.
    """
    # Hippocampus Field CA2 is strongly different -> merge it with CA1
    replace(ids_v2, 423, 382)
    replace(ids_v3, 423, 382)

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


def manual_relabel_2(ids_v2: np.ndarray, ids_v3: np.ndarray) -> None:
    """Perform a manual re-labeling step on the CCFv2 and CCFv3 atlases.

    The replacements were compiled by Dimitri Rodarie.

    Parameters
    ----------
    ids_v2
        The (unique) region IDs of the CCFv2 atlas.
    ids_v3
        The (unique) region IDs of the CCFv3 atlas.
    """
    # subreg of Prosubiculum to subiculum
    replace(ids_v3, 484682470, 502)
    # Orbital area, medial part, layer 6b -> 6a
    replace(ids_v3, 527696977, 910)
    replace(ids_v3, 355, 314)

    # Frontal pole children to their parent
    replace(ids_v3, 68, 184)
    replace(ids_v3, 667, 184)
    replace(ids_v3, 526157192, 184)
    replace(ids_v3, 526157196, 184)
    replace(ids_v3, 526322264, 184)
    replace(ids_v2, 68, 184)
    replace(ids_v2, 667, 184)

    # Every region ventral to cortex transition area is merge because
    # Entorhinal area, medial part, ventral zone is not in CCFv3
    # ie 259,  324,  371, 1133, 655, 663, 780 -> 663
    replace(ids_v2, 259, 663)
    replace(ids_v2, 324, 663)
    replace(ids_v2, 371, 663)
    replace(ids_v2, 1133, 663)

    replace(ids_v2, 655, 663)
    replace(ids_v2, 780, 663)
    replace(ids_v3, 655, 663)
    replace(ids_v3, 780, 663)


def merge(ccfv2, ccfv3, region_meta):
    """Perform the coarse atlas merging.

    Parameters
    ----------
    ccfv2 : np.ndarray
        The first atlas to merge, usually CCFv2
    ccfv3 : np.ndarray
        The second atlas to merge, usually CCFv3
    region_meta : RegionMeta
        The brain region metadata. Usually constructed as follows:
        ``RegionMeta.from_root_region(brain_regions)``, where ``brain_regions``
        can be obtained from the "msg" key of the `brain_regions.json`
        (`1.json`) file.

    Returns
    -------
    ccfv2_corrected : np.ndarray
        The merged CCFv2 atlas.
    ccfv3_corrected : np.ndarray
        The merged CCFv3 atlas.
    """
    logger.info("Preparing region ID maps")
    v2_from = np.unique(ccfv2)
    v2_to = v2_from.copy()
    v3_from = np.unique(ccfv3)
    v3_to = v3_from.copy()
    allowed_v2 = region_meta.collect_ancestors(v2_to)
    allowed_v3 = region_meta.collect_ancestors(v3_to)

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

    def in_region_like(name_part, region_id):
        """Check if region belongs to a region with a given name part."""
        while region_id != region_meta.background_id:
            if name_part in region_meta.name[region_id]:
                return True
            region_id = parent(region_id)

    logger.info("First for-loop correction")
    unique_v2 = set(v2_to)
    unique_v3 = set(v3_to)
    ids_to_correct = unique_v2 - unique_v3
    for id_reg in ids_to_correct:
        if (
            is_leaf(id_reg)
            and id_reg not in unique_v3
            and parent(id_reg) in unique_v3
        ):
            replace(v2_to, id_reg, parent(id_reg))
        elif is_leaf(id_reg) and (
            in_region_like("Medial amygdalar nucleus", id_reg)
            or in_region_like("Subiculum", id_reg)
            or in_region_like("Bed nuclei of the stria terminalis", id_reg)
        ):
            replace(v2_to, id_reg, parent(parent(id_reg)))
        elif in_region_like("Paraventricular hypothalamic nucleus", id_reg):
            replace(v2_to, id_reg, 38)

    logger.info("Manual relabeling #1")
    manual_relabel_1(v2_to, v3_to)

    logger.info("Second for loop correction")
    for id_reg in (unique_v2 | unique_v3) - {0}:
        if in_region_like("Visual areas", id_reg):
            if in_region_like("ayer 1", id_reg):
                replace(v3_to, id_reg, 801)
                replace(v2_to, id_reg, 801)
            elif in_region_like("ayer 2/3", id_reg):
                replace(v3_to, id_reg, 561)
                replace(v2_to, id_reg, 561)
            elif in_region_like("ayer 4", id_reg):
                replace(v3_to, id_reg, 913)
                replace(v2_to, id_reg, 913)
            elif in_region_like("ayer 5", id_reg):
                replace(v3_to, id_reg, 937)
                replace(v2_to, id_reg, 937)
            elif in_region_like("ayer 6a", id_reg):
                replace(v3_to, id_reg, 457)
                replace(v2_to, id_reg, 457)
            elif in_region_like("ayer 6b", id_reg):
                replace(v3_to, id_reg, 497)
                replace(v2_to, id_reg, 497)

    logger.info("Manual relabeling #2")
    manual_relabel_2(v2_to, v3_to)

    logger.info("Ramapping atlases")
    ccfv2_corrected = atlas_remap(ccfv2, v2_from, v2_to)
    ccfv3_corrected = atlas_remap(ccfv3, v3_from, v3_to)

    # Medial terminal nucleus of the accessory optic tract -> Ventral tegmental area

    def run_filter(atlas, region_id, count):
        keep_ids = [region_id, *descendants(region_id, allowed_v2)]
        hide_mask = np.isin(atlas, keep_ids, invert=True)
        masked_atlas = ma.masked_array(atlas, hide_mask)

        error_voxel = np.where(atlas == region_id)
        if len(error_voxel) == 0:
            logger.info(f"No voxels to explore", len(error_voxel[0]))
            return
        logger.info(f"Exploring %d voxels", len(error_voxel[0]))
        new_values = [
            explore_voxel(xyz, masked_atlas, count)
            for xyz in zip(*error_voxel)
        ]
        atlas[error_voxel] = new_values

    # Correct annotation edge for CCFv2 and CCFv3
    # no limit for striatum
    logger.info("First filter")
    run_filter(ccfv2_corrected, 278, -1)

    logger.info("Second filter")
    for id_reg in [803, 477]:
        run_filter(ccfv3_corrected, id_reg, -1)

    # Correct CCFv2 annotation edge Cerebral cortex, Basic Cell group and
    # regions and root  1089, 688, 8, 997
    logger.info("Third filter")
    for id_reg in [688, 8, 997]:
        run_filter(ccfv2_corrected, id_reg, 3)

    # Correct CCFv3 annotation edge for Hippocampal formation, Cortical subplate
    logger.info("Fourth filter")
    for id_reg in [1089, 703]:
        run_filter(ccfv3_corrected, id_reg, 3)

    logger.info("Preparing region ID maps")
    v2_from = np.unique(ccfv2_corrected)
    v2_to = v2_from.copy()
    v3_from = np.unique(ccfv3_corrected)
    v3_to = v3_from.copy()

    logger.info("Some more manual replacement of descendants")
    for id_main in [795]:
        for id_reg in descendants(id_main, allowed_v2):
            if id_reg in allowed_v2:
                replace(v2_to, id_reg, id_main)
            if id_reg in allowed_v3:
                replace(v3_to, id_reg, id_main)

    logger.info("More for-loop corrections")
    unique_v2 = set(v2_to)
    unique_v3 = set(v3_to)
    for id_reg in unique_v2 - {0}:
        if in_region_like("fiber tracts", id_reg):
            replace(v2_to, id_reg, 1009)
        elif in_region_like("ventricular systems", id_reg):
            replace(v2_to, id_reg, 997)
    for id_reg in unique_v3 - {0}:
        if in_region_like("fiber tracts", id_reg):
            replace(v3_to, id_reg, 1009)
        elif in_region_like("ventricular systems", id_reg):
            replace(v3_to, id_reg, 997)

    logger.info("While-loop correction")
    unique_v2 = set(v2_to)
    unique_v3 = set(v3_to)
    ids_to_correct = unique_v3 - unique_v2
    while len(ids_to_correct) > 0:
        id_ = ids_to_correct.pop()
        while id_ not in allowed_v2:
            id_ = parent(id_)
        for child in descendants(id_, allowed_v3):
            replace(v3_to, child, id_)
        for child in descendants(id_, allowed_v2):
            replace(v2_to, child, id_)
        unique_v2 = set(v2_to)
        unique_v3 = set(v3_to)
        ids_to_correct = unique_v3 - unique_v2

    logger.info("Ramapping atlases")
    ccfv2_new = atlas_remap(ccfv2_corrected, v2_from, v2_to)
    ccfv3_new = atlas_remap(ccfv3_corrected, v3_from, v3_to)

    return ccfv2_new, ccfv3_new
