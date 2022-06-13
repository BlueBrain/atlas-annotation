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
"""Tools to deal with brain hierarchy from the AIBS."""
from __future__ import annotations

import numpy as np


class RegionData:
    """Extract and process brain region data.

    Parameters
    ----------
    brain_regions : dict
        The brain regions dictionary. Can be obtained from the "msg" key of
        the `brain_regions.json` (`1.json`) file.
    """

    def __init__(self, brain_regions):
        # id to region name
        self.id_to_region_dictionary = {}
        # dictionary from region id to region complete name
        self.id_to_region_dictionary_ALLNAME = {}
        # name to id
        self.region_dictionary_to_id = {}
        # dictionary from region complete name to region id
        self.region_dictionary_to_id_ALLNAME = {}

        # dictionary from region complete name to its parent complete name
        self.region_dictionary_to_id_ALLNAME_parent = {}
        self.region_dictionary_to_id_parent = {}  # name to name parent
        self.allname2name = {}  # complete name to name
        # dictionary from region name to region complete name
        self.name2allname = {}
        self.region_keys = []  # list of regions names
        self.regions_ALLNAME_list = []  # list of complete regions names
        # dictionary from region complete name to boolean, True if the region
        # is a leaf region
        self.is_leaf = {}  # full name to int (! if is leaf, else 0)
        self.id_to_color = {}  # region id to color in RGB
        self.region_to_color = {}  # complete name to color in RGB
        self.id_to_abv = {}
        self.region_dictionary_to_abv = {}

        dict_corrections = {}
        old_regions_layer23 = [
            41,
            113,
            163,
            180,
            201,
            211,
            219,
            241,
            251,
            269,
            288,
            296,
            304,
            328,
            346,
            412,
            427,
            430,
            434,
            492,
            556,
            561,
            582,
            600,
            643,
            657,
            667,
            670,
            694,
            755,
            806,
            821,
            838,
            854,
            888,
            905,
            943,
            962,
            965,
            973,
            1053,
            1066,
            1106,
            1127,
            12994,
            182305697,
        ]
        for reg in old_regions_layer23:
            dict_corrections[reg] = [reg + 20000, reg + 30000]

        # Change of id when L2 and L2/3 existed
        dict_corrections[195] = [20304]
        dict_corrections[747] = [20556]
        dict_corrections[524] = [20582]
        dict_corrections[606] = [20430]

        inv_corrections = {}
        for k, v in dict_corrections.items():
            for conv in v:
                inv_corrections[conv] = k

        self.search_children(brain_regions)

    def find_unique_regions(
        self,
        annotation,
        top_region_name="Basic cell groups and regions",
    ):
        """Find unique regions.

        Finds unique regions ids that are present in an annotation file and
        are contained in the top_region_name Adds also to the list each parent
        of the regions present in the annotation file. Dictionaries parameters

        Parameters
        ----------
        annotation : np.ndarray
            3D numpy ndarray of integers ids of the regions.
        top_region_name : str
            Name of the most broader region included in the uniques.

        Returns
        -------
        uniques : np.ndarray
            Array of unique regions id in the annotation file that are
            included in top_region_name.
        """
        # Take the parent of the top region to stop the loop
        root_allname = self.region_dictionary_to_id_ALLNAME_parent[
            self.name2allname[top_region_name]
        ]
        uniques = []
        for uniq in np.unique(annotation)[1:]:  # Cell regions without outside
            allname = self.id_to_region_dictionary_ALLNAME[uniq]
            if (
                top_region_name in self.id_to_region_dictionary_ALLNAME[uniq]
                and uniq not in uniques
            ):
                uniques.append(uniq)
                parent_allname = self.region_dictionary_to_id_ALLNAME_parent[allname]
                id_parent = self.region_dictionary_to_id_ALLNAME[parent_allname]
                while id_parent not in uniques and parent_allname != root_allname:
                    uniques.append(id_parent)
                    parent_allname = self.region_dictionary_to_id_ALLNAME_parent[
                        parent_allname
                    ]
                    if parent_allname == "":
                        break
                    id_parent = self.region_dictionary_to_id_ALLNAME[parent_allname]

        return np.array(uniques)

    def find_children(self, uniques):
        """Find children.

        Finds the children regions of each region id in uniques and its
        distance from a leaf region in the hierarchy tree. Non leaf regions
        are included in the children list.

        Parameters
        ----------
        uniques : np.ndarray
            List of unique region ids

        Returns
        -------
        children : dict
             Dictionary of region complete name to list of child region ids.
        order_ : np.ndarray
             Array of distances from a leaf region in the hierarchy tree for
             each region in uniques.
        """
        children: dict[str, list[int]] = {}
        order_ = np.zeros(uniques.shape)
        for id_reg, allname in self.id_to_region_dictionary_ALLNAME.items():
            if self.is_leaf[allname]:
                inc = 0
                ids_reg = [id_reg]
                parentname = self.region_dictionary_to_id_ALLNAME_parent[allname]
                while parentname != "":
                    if parentname not in children:
                        children[parentname] = []
                    children[parentname] += ids_reg
                    inc += 1
                    id_parent = self.region_dictionary_to_id_ALLNAME[parentname]
                    if id_parent in uniques:
                        ids_reg.append(id_parent)
                        place_ = np.where(uniques == id_parent)
                        order_[place_] = max(order_[place_], inc)
                    allname = parentname
                    parentname = self.region_dictionary_to_id_ALLNAME_parent[allname]

        for parent, child in children.items():
            children[parent] = np.unique(child)
        return children, order_

    def filter_region(self, annotation, allname, children):
        """Filter a region.

        Computes a 3d boolean mask to filter a region and its subregion from
        the annotations.

        Parameters
        ----------
        annotation : np.ndarray
            3D numpy ndarray of integers ids of the regions.
        allname : str
            Complete name of the region.
        children : dict
            Dictionary of region complete name to list of child region ids.

        Returns
        -------
        filter_ : np.ndarray
            3D numpy ndarray of boolean, boolean mask with all the voxels of a
            region and its children set to True.
        """
        if not self.is_leaf[allname]:
            filter_ = np.in1d(
                annotation,
                np.concatenate(
                    (children[allname], [self.region_dictionary_to_id_ALLNAME[allname]])
                ),
            ).reshape(annotation.shape)
        else:
            filter_ = annotation == self.region_dictionary_to_id_ALLNAME[allname]
        return filter_

    def return_ids_containing_str_list(self, str_list):
        """Return IDs containing all keywords.

        Retrieve the list of region id which complete name contains all the
        keywords in str_list.

        Parameters
        ----------
        str_list : iterable
            List of keyword that the region complete name.

        Returns
        -------
        id_list : list
            List of region id matching condition
        """
        id_list = []
        for kk in self.id_to_region_dictionary_ALLNAME:
            region_is_in = True
            for str1 in str_list:
                # if any of the regions is not there, do not take
                if (self.id_to_region_dictionary_ALLNAME[kk].lower()).find(
                    str1.lower()
                ) < 0:
                    region_is_in = False
                    break
            if region_is_in:
                id_list.append(kk)
        return id_list

    @staticmethod
    def hex_to_rgb(value):
        """Convert a hexadecimal color into its RGB value counterpart.

        Parameters
        ----------
        value : str
            Hexadecimal color to convert.

        Returns
        -------
        tuple
            List of the Red, Green, and Blue components of the color.
        """
        value = value.lstrip("#")
        lv = len(value)
        return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def search_children(self, object_, lastname_ALL="", lastname="", darken=True):
        """Explores the hierarchy dictionary to extract brain regions.

        Explores the hierarchy dictionary to extract its brain regions and
        fills internal dictionaries.

        Parameters
        ----------
        object_ : dict
            Dictionary of regions properties. See
            https://bbpteam.epfl.ch/documentation/projects/voxcell/latest/atlas.html#brain-region-hierarchy
        lastname_ALL : str
            Complete name of the parent of the current brain region.
        lastname : str
            Name of the parent of the current brain region.
        darken : bool
            If True, darkens the region colors too high.
        """
        self.regions_ALLNAME_list.append(lastname_ALL + "|" + object_["name"])
        self.name2allname[object_["name"]] = lastname_ALL + "|" + object_["name"]
        self.allname2name[lastname_ALL + "|" + object_["name"]] = object_["name"]
        self.id_to_region_dictionary[object_["id"]] = object_["name"]
        self.id_to_abv[object_["id"]] = object_["acronym"]
        self.id_to_region_dictionary_ALLNAME[object_["id"]] = (
            lastname_ALL + "|" + object_["name"]
        )
        self.region_dictionary_to_id[object_["name"]] = object_["id"]
        self.region_dictionary_to_abv[object_["name"]] = object_["acronym"]
        self.region_dictionary_to_id_ALLNAME[
            lastname_ALL + "|" + object_["name"]
        ] = object_["id"]
        self.region_dictionary_to_id_ALLNAME_parent[
            lastname_ALL + "|" + object_["name"]
        ] = lastname_ALL
        self.region_dictionary_to_id_parent[object_["name"]] = lastname
        clrTMP = np.float32(
            np.array(list(self.hex_to_rgb(object_["color_hex_triplet"])))
        )
        if np.sum(clrTMP) > 255.0 * 3.0 * 0.75 and darken:
            clrTMP *= 255.0 * 3.0 * 0.75 / np.sum(clrTMP)
        self.region_to_color[lastname_ALL + "|" + object_["name"]] = list(clrTMP)
        self.id_to_color[object_["id"]] = list(clrTMP)
        self.region_keys.append(object_["name"])
        try:
            self.is_leaf[lastname_ALL + "|" + object_["name"]] = 1
            # ~ region_dictionary_to_id_ALLNAME_child[
            #       lastname_ALL + "|" + object_["name"]
            #   ] = children
            # ~ id_children[object_["id"]] = object_["children"]
            for children in object_["children"]:
                self.search_children(
                    children,
                    lastname_ALL + "|" + object_["name"],
                    object_["name"],
                    darken=darken,
                )
                self.is_leaf[lastname_ALL + "|" + object_["name"]] = 0
        except KeyError:
            print("No children of object")
