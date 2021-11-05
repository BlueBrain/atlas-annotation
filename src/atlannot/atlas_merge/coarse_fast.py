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
"""The coarse merging of the annotation atlases.

This is the refactored and optimized version of ``coarse::coarse_merge``. It
uses ``RegionMeta`` instead of ``JSONread`` and greatly speeds up the merging
by optimizing a number of steps. The original logic was designed by
Dimitri Rodarie.

The biggest optimization is to not do label replacements directly on the atlases
but on the set of unique labels, which is much smaller than the atlas volume.
The labels in the atlases are remapped at the very end of the whole procedure
using fast vectorized numpy operations, see ``atlas_remap``.
"""
from __future__ import annotations

import logging

import numpy as np

from atlannot.atlas.region_meta import RegionMeta
from atlannot.atlas_merge.common import atlas_remap, replace

logger = logging.getLogger(__name__)


def manual_relabel(ids_v2: np.ndarray, ids_v3: np.ndarray) -> None:
    """Perform a manual re-labeling step on the CCFv2 and CCFv3 atlases.

    The replacements were compiled by Dimitri Rodarie.

    Parameters
    ----------
    ids_v2
        The (unique) region IDs of the CCFv2 atlas.
    ids_v3
        The (unique) region IDs of the CCFv3 atlas.
    """
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


def merge(
    ccfv2: np.ndarray, ccfv3: np.ndarray, brain_regions: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Perform the coarse atlas merging.

    Parameters
    ----------
    ccfv2
        The first atlas to merge, usually CCFv2.
    ccfv3
        The second atlas to merge, usually CCFv3.
    brain_regions
        The brain regions dictionary. Can be obtained from the "msg" key of
        the `brain_regions.json` (`1.json`) file.

    Returns
    -------
    ccfv2_new : np.ndarray
        The merged CCFv2 atlas.
    ccfv3_new : np.ndarray
        The merged CCFv3 atlas.
    """
    logger.info("Preparing region metadata")
    region_meta = RegionMeta.from_root_region(brain_regions)

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

    logger.info("Preparing region ID maps")
    v2_from = np.unique(ccfv2)
    v2_to = v2_from.copy()
    v3_from = np.unique(ccfv3)
    v3_to = v3_from.copy()

    logger.info("First loop")
    unique_v2 = set(v2_to)
    unique_v3 = set(v3_to)
    ids_to_correct = unique_v2 - unique_v3
    for id_ in ids_to_correct:
        if is_leaf(id_) and id_ not in unique_v3 and parent(id_) in unique_v3:
            replace(v2_to, id_, parent(id_))
        elif is_leaf(id_) and (
            in_region_like("Medial amygdalar nucleus", id_)
            or in_region_like("Subiculum", id_)
            or in_region_like("Bed nuclei of the stria terminalis", id_)
        ):
            replace(v2_to, id_, parent(parent(id_)))
        elif in_region_like("Paraventricular hypothalamic nucleus", id_):
            replace(v3_to, id_, 38)

    logger.info("Manual replacements #1")
    manual_relabel(v2_to, v3_to)

    logger.info("Second loop")
    for id_ in (unique_v2 | unique_v3) - {0}:
        if not in_region_like("Visual areas", id_):
            continue

        if in_region_like("ayer 1", id_):
            replace(v3_to, id_, 801)
            replace(v2_to, id_, 801)
        elif in_region_like("ayer 2/3", id_):
            replace(v3_to, id_, 561)
            replace(v2_to, id_, 561)
        elif in_region_like("ayer 4", id_):
            replace(v3_to, id_, 913)
            replace(v2_to, id_, 913)
        elif in_region_like("ayer 5", id_):
            replace(v3_to, id_, 937)
            replace(v2_to, id_, 937)
        elif in_region_like("ayer 6a", id_):
            replace(v3_to, id_, 457)
            replace(v2_to, id_, 457)
        elif in_region_like("ayer 6b", id_):
            replace(v3_to, id_, 497)
            replace(v2_to, id_, 497)

    logger.info("Manual replacements #2")
    # subreg of Prosubiculum to subiculum
    replace(v3_to, 484682470, 502)
    # Orbital area, medial part, layer 6b -> 6a
    replace(v3_to, 527696977, 910)
    replace(v3_to, 355, 314)

    logger.info("Third loop")
    for id_ in unique_v3 - {0}:
        if (
            (
                in_region_like("fiber tracts", id_)
                or in_region_like("Interpeduncular nucleus", id_)
            )
            and id_ not in unique_v2
            and parent(id_) in unique_v2
        ):
            replace(v3_to, id_, parent(id_))
        if in_region_like("Frontal pole, cerebral cortex", id_):
            replace(v3_to, id_, 184)
            replace(v2_to, id_, 184)

    def while_correct(ids_1, ids_2, allowed_1, allowed_2):
        unique_1 = set(ids_1)
        unique_2 = set(ids_2)
        ids_to_correct_ = unique_1 - unique_2 - {8, 997}
        while len(ids_to_correct_) > 0:
            id__ = ids_to_correct_.pop()
            while id__ not in allowed_v2:
                id__ = parent(id__)
            for child in descendants(id__, allowed_1):
                replace(ids_1, child, id__)
            for child in descendants(id__, allowed_2):
                replace(ids_2, child, id__)
            unique_2 = set(ids_2)
            unique_1 = set(ids_1)
            ids_to_correct_ = unique_1 - unique_2 - {8, 997}

    logger.info("While loop corrections")
    allowed_v2 = region_meta.collect_ancestors(v2_to)
    allowed_v3 = region_meta.collect_ancestors(v3_to)
    while_correct(v3_to, v2_to, allowed_v3, allowed_v2)
    while_correct(v2_to, v3_to, allowed_v2, allowed_v3)

    logger.info("Applying replacements")
    ccfv2_new = atlas_remap(ccfv2, v2_from, v2_to)
    ccfv3_new = atlas_remap(ccfv3, v3_from, v3_to)

    return ccfv2_new, ccfv3_new
