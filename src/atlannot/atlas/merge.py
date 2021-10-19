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
"""Tools to deal with brain hierarchy from the AIBS.

See https://bbpteam.epfl.ch/project/issues/browse/DEAL-20
"""
import numpy as np

id_to_region_dictionary = {}  # id to region name
id_to_region_dictionary_ALLNAME = {}  # id to complete name
region_dictionary_to_id = {}  # name to id
region_dictionary_to_id_ALLNAME = {}  # complete name to id

region_dictionary_to_id_ALLNAME_parent = {}  # complete name to complete name parent
region_dictionary_to_id_parent = {}  # name to name parent
allname2name = {}  # complete name to name
name2allname = {}  # name to complete name
region_keys = []  # list of regions names
regions_ALLNAME_list = []  # list of complete regions names
is_leaf = {}  # full name to int (! if is leaf, else 0)
id_to_color = {}  # region id to color in RGB
region_to_color = {}  # complete name to color in RGB
id_to_abv = {}
region_dictionary_to_abv = {}


def find_unique_regions(
    annotation,
    id_to_region_dictionary_allname,
    region_dictionary_to_id_allname,
    region_dictionary_to_id_allname_parent,
    name_to_allname,
    top_region_name="Basic cell groups and regions",
):
    """Find unique regions.

    Finds unique regions ids that are present in an annotation file and are
    contained in the top_region_name. Adds also to the list each parent of
    the regions present in the annotation file. Dictionaries parameters
    correspond to the ones produced in JSONread.

    Parameters
    ----------
    annotation
        3D numpy ndarray of integers ids of the regions
    id_to_region_dictionary_allname
        dictionary from region id to region complete name
    region_dictionary_to_id_allname
        dictionary from region complete name to region id
    region_dictionary_to_id_allname_parent
        dictionary from region complete name to its parent complete name
    name_to_allname
        dictionary from region name to region complete name
    top_region_name
        name of the most broader region included in the uniques

    Returns
    -------
    np.ndarray
        List of unique regions id in the annotation file that are included
        in top_region_name
    """
    # Take the parent of the top region to stop the loop
    root_allname = region_dictionary_to_id_allname_parent[
        name_to_allname[top_region_name]
    ]
    uniques = []
    for uniq in np.unique(annotation)[1:]:  # Cell regions without outside
        allname = id_to_region_dictionary_allname[uniq]
        if (
            top_region_name in id_to_region_dictionary_allname[uniq]
            and uniq not in uniques
        ):
            uniques.append(uniq)
            parent_allname = region_dictionary_to_id_allname_parent[allname]
            id_parent = region_dictionary_to_id_allname[parent_allname]
            while id_parent not in uniques and parent_allname != root_allname:
                uniques.append(id_parent)
                parent_allname = region_dictionary_to_id_allname_parent[parent_allname]
                if parent_allname == "":
                    break
                id_parent = region_dictionary_to_id_allname[parent_allname]

    return np.array(uniques)


def find_children(
    uniques,
    id_to_region_dictionary_allname,
    is_leaf,
    region_dictionary_to_id_allname_parent,
    region_dictionary_to_id_allname,
):
    """Find children.

    Finds the children regions of each region id in uniques and its
    distance from a leaf region in the hierarchy tree. Non leaf regions
    are included in the children list Dictionaries parameters correspond
    to the ones produced in JSONread.

    Parameters
    ----------
    uniques
        List of unique region ids
    id_to_region_dictionary_allname
        dictionary from region id to region complete name
    is_leaf
        dictionary from region complete name to boolean, True if the region
        is a leaf region.
    region_dictionary_to_id_allname_parent
        dictionary from region complete name to its parent complete name
    region_dictionary_to_id_allname
        dictionary from region complete name to region id

    Returns
    -------
    children
         Dictionary of region complete name to list of child region ids
    order_
         List of distances from a leaf region in the hierarchy tree for each
         region in uniques.
    """
    children = {}
    order_ = np.zeros(uniques.shape)
    for id_reg, allname in id_to_region_dictionary_allname.items():
        if is_leaf[allname]:
            inc = 0
            ids_reg = [id_reg]
            parentname = region_dictionary_to_id_allname_parent[allname]
            while parentname != "":
                if parentname not in children:
                    children[parentname] = []
                children[parentname] += ids_reg
                inc += 1
                id_parent = region_dictionary_to_id_allname[parentname]
                if id_parent in uniques:
                    ids_reg.append(id_parent)
                    place_ = np.where(uniques == id_parent)
                    order_[place_] = max(order_[place_], inc)
                allname = parentname
                parentname = region_dictionary_to_id_allname_parent[allname]

    for parent, child in children.items():
        children[parent] = np.unique(child)
    return children, order_


def filter_region(
    annotation, allname, children, is_leaf, region_dictionary_to_id_allname
):
    """Filter region.

    Computes a 3d boolean mask to filter a region and its subregion from
    the annotations. Dictionaries parameters correspond to the ones
    produced in JSONread.

    Parameters
    ----------
    annotation
        3D numpy ndarray of integers ids of the regions
    allname
        Complete name of the region
    children
        Dictionary of region complete name to list of child region ids
    is_leaf
        dictionary from region complete name to boolean, True if the region
        is a leaf region.
    region_dictionary_to_id_allname
        dictionary from region complete name to region id

    Returns
    -------
    filter
        3d numpy ndarray of boolean, boolean mask with all the voxels of
        a region and its children set to True.
    """
    if not is_leaf[allname]:
        filter = np.in1d(
            annotation,
            np.concatenate(
                (children[allname], [region_dictionary_to_id_allname[allname]])
            ),
        ).reshape(annotation.shape)
    else:
        filter = annotation == region_dictionary_to_id_allname[allname]
    return filter


def return_ids_containing_str_list(str_list):
    """Retrieve IDs containing keywords.

    Retrieve the list of region id which complete name contains all the
    keywords in str_list.

    Parameters
    ----------
    str_list
        List of keyword that the region complete name.

    Returns
    -------
    id_list
        List of region id matching condition.
    """
    id_list = []
    for kk in id_to_region_dictionary_ALLNAME:
        region_is_in = True
        for str1 in str_list:
            if (id_to_region_dictionary_ALLNAME[kk].lower()).find(
                str1.lower()
            ) < 0:  # if any of the regions is not there, do not take
                region_is_in = False
                break
        if region_is_in:
            id_list.append(kk)
    return id_list


def hex_to_rgb(value):
    """Convert color from hex to RGB representation.

    Converts a Hexadecimal color into its RGB value counterpart.

    Parameters
    ----------
    value
        string hexadecimal color to convert.

    Returns
    -------
    tuple
        List of the Red, Green, and Blue components of the color.
    """
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def search_children(object_, lastname_all="", lastname="", darken=True):
    """Search children.

    Explores the hierarchy dictionary to extract its brain regions and
    fills external dictionaries.

    Parameters
    ----------
    object_
        dictionary of regions properties. See
        https://bbpteam.epfl.ch/documentation/projects/voxcell/latest/atlas.html#brain-region-hierarchy
    lastname_all
        complete name of the parent of the current brain region
    lastname
        name of the parent of the current brain region
    darken
        if True, darkens the region colors too high
    """
    regions_ALLNAME_list.append(lastname_all + "|" + object_["name"])
    name2allname[object_["name"]] = lastname_all + "|" + object_["name"]
    allname2name[lastname_all + "|" + object_["name"]] = object_["name"]
    id_to_region_dictionary[object_["id"]] = object_["name"]
    id_to_abv[object_["id"]] = object_["acronym"]
    id_to_region_dictionary_ALLNAME[object_["id"]] = (
        lastname_all + "|" + object_["name"]
    )
    region_dictionary_to_id[object_["name"]] = object_["id"]
    region_dictionary_to_abv[object_["name"]] = object_["acronym"]
    region_dictionary_to_id_ALLNAME[lastname_all + "|" + object_["name"]] = object_[
        "id"
    ]
    region_dictionary_to_id_ALLNAME_parent[
        lastname_all + "|" + object_["name"]
    ] = lastname_all
    region_dictionary_to_id_parent[object_["name"]] = lastname
    clr_tmp = np.float32(np.array(list(hex_to_rgb(object_["color_hex_triplet"]))))
    if np.sum(clr_tmp) > 255.0 * 3.0 * 0.75 and darken:
        clr_tmp *= 255.0 * 3.0 * 0.75 / np.sum(clr_tmp)
    region_to_color[lastname_all + "|" + object_["name"]] = list(clr_tmp)
    id_to_color[object_["id"]] = list(clr_tmp)
    region_keys.append(object_["name"])
    try:
        is_leaf[lastname_all + "|" + object_["name"]] = 1
        # ~ region_dictionary_to_id_ALLNAME_child[
        #       lastname_ALL+"|"+object_["name"]] = children
        # ~ id_children[object_["id"]] = object_["children"]
        for children in object_["children"]:
            search_children(
                children,
                lastname_all + "|" + object_["name"],
                object_["name"],
                darken=darken,
            )
            is_leaf[lastname_all + "|" + object_["name"]] = 0
    except KeyError:
        print("No children of object")


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
