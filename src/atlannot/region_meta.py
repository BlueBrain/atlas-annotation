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
import logging
import numbers

logger = logging.getLogger(__name__)


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
        self.children_ids = {self.background_id: []}

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

    def is_leaf(self, region_id):
        """Check if the given region is a leaf region.

        Parameters
        ----------
        region_id : int
            The region ID in question.

        Returns
        -------
        bool
            Whether or not the given region is a leaf region.
        """
        return len(self.children_ids[region_id]) == 0

    def parent(self, region_id):
        """Get the parent region ID of a region.

        Parameters
        ----------
        region_id
            The region ID in question.

        Returns
        -------
        int or None
            The region ID of the parent. If there's no parent then None is
            returned.
        """
        return self.parent_id.get(region_id)

    def children(self, region_id):
        """Get all child region IDs of a given region.

        Note that by children we mean only the direct children, much like
        by parent we only mean the direct parent. The cumulative quantities
        that span all generations are called ancestors and descendants.

        Parameters
        ----------
        region_id : int
            The region ID in question.

        Yields
        ------
        int
            The region ID of a child region.
        """
        return tuple(self.children_ids[region_id])

    def in_region_like(self, name_part, region_id):
        """Check if region belongs to a region with a given name part.

        Parameters
        ----------
        name_part : str
            Part of the name of a brain region.
        region_id : int
            A region ID.

        Returns
        -------
        bool
            Whether or not the region with the given ID is in a region with
            the given name part. All parent regions are also checked.
        """
        if region_id not in self.name:
            logger.warning("Invalid region ID: %d", region_id)
            return False

        while region_id != self.background_id:
            if name_part in self.name[region_id]:
                return True
            region_id = self.parent(region_id)

        return False

    def ancestors(self, ids, include_background=False):
        """Find all ancestors of given regions.

        The result is inclusive, i.e. the input region IDs will be
        included in the result.

        Parameters
        ----------
        ids : int or iterable of int
            A region ID or a collection of region IDs to collect ancestors for.
        include_background : bool
            If True the background region ID will be included in the result.

        Returns
        -------
        set
            All ancestor region IDs of the given regions, including the input
            regions themselves.
        """
        if isinstance(ids, numbers.Integral):
            unique_ids = {ids}
        else:
            unique_ids = set(ids)

        ancestors = set()
        for id_ in unique_ids:
            while id_ is not None:
                ancestors.add(id_)
                id_ = self.parent(id_)

        if not include_background:
            ancestors.remove(self.background_id)

        return ancestors

    def descendants(self, ids):
        """Find all descendants of given regions.

        The result is inclusive, i.e. the input region IDs will be
        included in the result.

        Parameters
        ----------
        ids : int or iterable of int
            A region ID or a collection of region IDs to collect
            descendants for.

        Returns
        -------
        set
            All descendant region IDs of the given regions, including the input
            regions themselves.
        """
        if isinstance(ids, numbers.Integral):
            unique_ids = {ids}
        else:
            unique_ids = set(ids)

        def iter_descendants(region_id):
            """Iterate over all descendants of a given region ID."""
            yield region_id
            for child in self.children(region_id):
                yield child
                yield from iter_descendants(child)

        descendants = set()
        for id_ in unique_ids:
            descendants |= set(iter_descendants(id_))

        return descendants

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
        self.children_ids[parent_id].append(region_id)

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
        self.children_ids[region_id] = []

        for child in region["children"]:
            self._parse_region_hierarchy(child)

    @classmethod
    def from_dict(cls, region_hierarchy, warn_raw_response=True):
        """Construct an instance from the region hierarchy.

        Parameters
        ----------
        region_hierarchy : dict
            The dictionary of the region hierarchy. Should have the format
            as usually provided by the AIBS.
        warn_raw_response: bool
            If True and a raw AIBS response (containing the "msg" key) is used,
            then a warning will be logged.

        Returns
        -------
        region_meta : RegionMeta
            The initialized instance of this class.
        """
        if "msg" in region_hierarchy:
            if warn_raw_response:
                logger.warning(
                    "Seems like you're trying to use the raw AIBS response as "
                    'input, I gotcha. Next time please use response["msg"][0].'
                )
            region_hierarchy = region_hierarchy["msg"][0]

        self = cls()
        self._parse_region_hierarchy(
            region_hierarchy,
            parent_id=self.background_id,
        )

        return self
