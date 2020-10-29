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
        meta : object
            The initialized instance of this class
        """
        meta = cls()
        meta._parse_region_hierarchy(root_region, parent_id=meta.background_id)

        return meta
