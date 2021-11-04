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
"""Implementation of the RegionMeta class."""


class RegionMeta:
    """Class holding the hierarchical region metadata.

    Typically, such information would be parsed from a `brain_regions.json`
    file.

    Parameters
    ----------
    background_id : int, optional
        Override the default ID for the background.
    background_color : str, optional
        Override the default color for the background. Should be a
        string of length 6 with RGB values in hexadecimal form.
    """

    def __init__(self, background_id=0, background_color="000000"):
        self.background_id = background_id

        self.atlas_id = {self.background_id: None}
        self.ontology_id = {self.background_id: None}
        self.acronym = {self.background_id: "bg"}
        self.name = {self.background_id: "background"}
        self.color_hex_triplet = {self.background_id: background_color}
        self.graph_order = {self.background_id: None}
        self.st_level = {self.background_id: None}
        self.hemisphere_id = {self.background_id: None}
        self.parent_id = {self.background_id: None}

        self.level = {self.background_id: 0}

    @property
    def color_map(self):
        """Map region IDs to RGB colors.

        Returns
        -------
        color_map : dict
            The color map. Keys are regions IDs, and values are tuples
            with three integers between 0 and 255 representing the RGB
            colors.
        """
        color_map = {}
        for region_id, color in self.color_hex_triplet.items():
            red, green, blue = color[:2], color[2:4], color[4:]
            color_map[region_id] = tuple(int(c, base=16) for c in (red, green, blue))
        return color_map

    def ids_at_level(self, level):
        """Region IDs at a given hierarchy level.

        Finds all region IDs at the given level and yields them one by one.

        Yields
        ------
        region_id : int
            A region ID at the given level
        """
        for region_id, region_level in self.level.items():
            if region_level == level:
                yield region_id

    def collect_ancestors(self, leaf_ids, top_id=None, remove_background=True):
        """Collect all region IDs between the leaf regions and the top region.

        Leaf regions that don't descent from the given top region are ignored.
        The collection is inclusive, i.e. both the leaf region IDs and the
        top region ID will be collected along with all intermediate regions.

        If the top region is not provided it will default to the background.
        This will effectively collect all parents of all leaves up to the top
        since the background is at the top of the hierarchy.

        Parameters
        ----------
        leaf_ids : iterable of int
            The IDs of the leaf regions
        top_id : int
            The ID of the top region up to which to collect the parents.
        remove_background : bool
            If True it will be guaranteed that the background region ID is
            not included in the result.

        Returns
        -------
        set
            All leaf IDs, the top region ID, and all the intermediate region IDs
            between the leaf and the top regions.
        """
        if top_id is None:
            top_id = self.background_id

        def descends_from_top_id(child_id):
            """Check if the given ID is a descendant of top_id."""
            if child_id == top_id:
                return True
            if child_id is None:
                return False

            return descends_from_top_id(self.parent_id[child_id])

        # The actual work - collect all ancestors up to the top_id
        ids = {top_id}
        for id_ in set(leaf_ids):
            if not descends_from_top_id(id_):
                continue
            while id_ != top_id:
                ids.add(id_)
                id_ = self.parent_id[id_]

        if remove_background:
            ids -= {self.background_id}

        return ids

    def _parse_region_hierarchy(self, region, parent_id=None):
        """Parse and save a region and its children.

        This helper method is usually used to initialize the class
        instance.

        Parameters
        ----------
        region : dict
            Metadata for a region and its children.
        parent_id : int or None, optional
            Override the region parent. Typically used to set the background
            as the parent region of the top-level region (e. g. root).
        """
        region_id = region["id"]
        if parent_id is None:
            parent_id = region["parent_structure_id"]

        self.atlas_id[region_id] = region["atlas_id"]
        self.ontology_id[region_id] = region["ontology_id"]
        self.acronym[region_id] = region["acronym"]
        self.name[region_id] = region["name"]
        self.color_hex_triplet[region_id] = region["color_hex_triplet"]
        self.graph_order[region_id] = region["graph_order"]
        self.st_level[region_id] = region["st_level"]
        self.hemisphere_id[region_id] = region["hemisphere_id"]
        self.parent_id[region_id] = parent_id

        self.level[region_id] = self.level[parent_id] + 1

        for child in region["children"]:
            self._parse_region_hierarchy(child)

    @classmethod
    def from_root_region(cls, root_region):
        """Construct and instance from the top-level region.

        Parameters
        ----------
        root_region : dict
            The metadata of the top-level region.

        Returns
        -------
        meta : RegionMeta
            The initialized instance of this class
        """
        meta = cls()
        meta._parse_region_hierarchy(root_region, parent_id=meta.background_id)

        return meta
