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
import json
import logging
import numbers
import re

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
        self.root_id = None

        self.atlas_id = {self.background_id: None}
        self.ontology_id = {self.background_id: None}
        self.acronym_ = {self.background_id: "bg"}
        self.name_ = {self.background_id: "background"}
        self.color_hex_triplet = {self.background_id: background_color}
        self.graph_order = {self.background_id: None}
        self.st_level = {self.background_id: None}
        self.hemisphere_id = {self.background_id: None}
        self.parent_id = {self.background_id: None}
        self.children_ids = {self.background_id: []}

        self.level = {self.background_id: 0}

    def __repr__(self):
        """Create the repr of the instance."""
        return f"{self.__class__.__qualname__}, {self.size} regions, depth {self.depth}"

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

    def is_valid_id(self, id_):
        """Check whether the given region ID is part of the structure graph.

        Parameters
        ----------
        id_ : int
            The region ID in question.

        Returns
        -------
        bool
            Whether the given region ID is part of the structure graph
        """
        # The parent_id dictionary should have all region IDs as keys
        return id_ in self.parent_id

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

    def name(self, id_):
        """Get the name of a region.

        Parameters
        ----------
        id_
            A region ID.

        Returns
        -------
        str
            The name of the given region.
        """
        if not self.is_valid_id(id_):
            logger.warning(f"Unknown region ID: {id_!r}; no name available.")
            return ""
        else:
            return self.name_[id_]

    def acronym(self, id_):
        """Get the acronym of a region.

        Parameters
        ----------
        id_
            A region ID.

        Returns
        -------
        The acronym a the given region.
        """
        if not self.is_valid_id(id_):
            logger.warning(f"Unknown region ID: {id_!r}; no acronym available.")
            return ""
        else:
            return self.acronym_[id_]

    def find_by_name(self, name):
        """Find the region ID given its name.

        Parameters
        ----------
        name : str
            The name of a region ID

        Returns
        -------
        int or None
            The region ID if a region is found, otherwise None
        """
        for id_, region_name in self.name_.items():
            if name == region_name:
                return id_

        return None

    def find_by_acronym(self, acronym):
        """Find the region ID given its acronym.

        Parameters
        ----------
        acronym : str
            The acronym of a region ID

        Returns
        -------
        int or None
            The region ID if a region is found, otherwise None
        """
        for id_, region_acronym in self.acronym_.items():
            if acronym == region_acronym:
                return id_

        return None

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

    @property
    def depth(self):
        """Find the depth of the region hierarchy.

        The background region is not taken into account.

        Returns
        -------
        int
            The depth of the region hierarchy.
        """
        return max(self.level.values())

    @property
    def size(self):
        """Find the number of regions in the structure graph.

        Returns
        -------
        int
            The number of regions in the structure graph
        """
        # parent_id should have all region IDs as keys. Subtract one to remove
        # the background from the count
        return len(self.parent_id) - 1

    def in_region_like(self, region_name_regex, region_id):
        """Check if region belongs to a region with a given name pattern.

        Note that providing a simple string without any special characters
        is equivalent to a substring test.

        Parameters
        ----------
        region_name_regex : str
            A regex to match the region name.
        region_id : int
            A region ID.

        Returns
        -------
        bool
            Whether or not the region with the given ID is in a region with
            the given name part. All parent regions are also checked.
        """
        if not self.is_valid_id(region_id):
            logger.warning("Invalid region ID: %d", region_id)
            return False

        while region_id != self.background_id:
            if re.search(region_name_regex, self.name(region_id)):
                return True
            region_id = self.parent(region_id)

        return False

    def _parse_region_hierarchy(self, region, is_root=False):
        """Parse and save a region and its children.

        This helper method is usually used to initialize the class
        instance.

        Parameters
        ----------
        region : dict
            Metadata for a region and its children.
        is_root : bool, default False
            If True then it will be assumed that this region is the root
            region. As a consequence it will be attached as the child of
            the background.
        """
        region_id = region["id"]
        if is_root:
            self.root_id = region_id
            parent_id = self.background_id
        else:
            parent_id = region["parent_structure_id"]
        self.children_ids[parent_id].append(region_id)

        self.atlas_id[region_id] = region["atlas_id"]
        self.ontology_id[region_id] = region["ontology_id"]
        self.acronym_[region_id] = region["acronym"]
        self.name_[region_id] = region["name"]
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
        self._parse_region_hierarchy(region_hierarchy, is_root=True)

        return self

    def to_dict(self, root_id=None):
        """Serialise the region structure data to a dictionary.

        This is exactly the inverse of the ``from_dict`` method.

        Parameters
        ----------
        root_id : int or None, optional
            Which region ID to start with. This will be the new top of the
            serialised structure graph. If none is provided then to real
            root region is used and as a consequence the complete structure
            graph is serialised.

        Returns
        -------
        dict
            The serialised region structure data.
        """

        def region_to_dict(id_):
            region_dict = {
                "id": id_,
                "atlas_id": self.atlas_id[id_],
                "ontology_id": self.ontology_id[id_],
                "acronym": self.acronym_[id_],
                "name": self.name_[id_],
                "color_hex_triplet": self.color_hex_triplet[id_],
                "graph_order": self.graph_order[id_],
                "st_level": self.st_level[id_],
                "hemisphere_id": self.hemisphere_id[id_],
                "parent_structure_id": self.parent_id[id_],
                "children": [],
            }
            for child_id in self.children_ids[id_]:
                region_dict["children"].append(region_to_dict(child_id))

            return region_dict

        if root_id is None:
            root_id = self.root_id
        result = region_to_dict(root_id)
        # Detach the root region rest of the rest of structure graph
        result["parent_structure_id"] = None

        return result

    @classmethod
    def load_json(cls, json_path):
        """Load the structure graph from a JSON file and create an instance.

        Parameters
        ----------
        json_path : str or pathlib.Path

        Returns
        -------
        RegionMeta
            The initialized instance of this class.
        """
        with open(json_path) as fh:
            structure_graph = json.load(fh)

        # The JSON file could be either a raw response with the AIBS headers
        # or just the bare structure graph. We don't make any assumptions and
        # support both. No need to warn the user at this point if it's the
        # raw response.

        return cls.from_dict(structure_graph, warn_raw_response=False)
