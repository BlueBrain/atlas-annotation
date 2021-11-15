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
import json
import logging
import textwrap

import pytest

from atlannot.region_meta import RegionMeta


@pytest.fixture()
def structure_graph():
    # The mini structure graph has the following simple structure:
    # 1 (root)
    # ├── 2 (Child 1)
    # │   ├── 4 (Grandchild 1)
    # │   └── 5 (Grandchild 2)
    # └── 3 (Child 2)
    with open("tests/data/structure_graph_mini.json") as fh:
        structure_graph = json.load(fh)

    return structure_graph


def as_aibs_response(structure_graph):
    response = {
        "success": True,
        "id": 0,
        "start_row": 0,
        "num_rows": 1,
        "total_rows": 1,
        "msg": [structure_graph],
    }

    return response


def test_from_dict(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)

    assert rm.background_id == 0
    assert rm.atlas_id == {0: None, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1}
    assert rm.ontology_id == {0: None, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    assert rm.name_ == {
        0: "background",
        1: "root",
        2: "Child 1",
        3: "Child 2",
        4: "Grandchild 1",
        5: "Grandchild 2",
    }
    assert rm.acronym_ == {0: "bg", 1: "root", 2: "C1", 3: "C2", 4: "Gc1", 5: "Gc2"}
    assert rm.color_hex_triplet == {
        0: "000000",
        1: "FFFFFF",
        2: "FFF000",
        3: "000FFF",
        4: "0F0F0F",
        5: "F0F0F0",
    }
    assert rm.graph_order == {0: None, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    assert rm.st_level == {0: None, 1: 0, 2: 0, 4: 0, 5: 0, 3: 0}
    assert rm.hemisphere_id == {0: None, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3}
    assert rm.parent_id == {0: None, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    assert rm.level == {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}


def test_from_dict_raw_response(structure_graph, caplog):
    rm_ref = RegionMeta.from_dict(structure_graph)
    raw_response = as_aibs_response(structure_graph)

    # Test parsing the raw AIBS response
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="atlannot.region_meta"):
        rm = RegionMeta.from_dict(raw_response)
        # Check the parsing still worked
        assert rm.parent_id == rm_ref.parent_id
    assert len(caplog.records) == 1
    assert "raw AIBS response" in caplog.records[0].message

    # Same, but with suppressed warnings
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="atlannot.region_meta"):
        rm = RegionMeta.from_dict(raw_response, warn_raw_response=False)
        # Check the parsing still worked
        assert rm.parent_id == rm_ref.parent_id
    assert len(caplog.records) == 0


def test_load_json(structure_graph, tmp_path):
    json_path = tmp_path / "structure_graph.json"

    # Test loading a bare structure graph
    with json_path.open("w") as fh:
        json.dump(structure_graph, fh)
    rm = RegionMeta.load_json(json_path)
    assert rm.to_dict() == structure_graph

    # Test loading a raw AIBS response
    structure_graph_raw = as_aibs_response(structure_graph)
    with json_path.open("w") as fh:
        json.dump(structure_graph_raw, fh)
    rm = RegionMeta.load_json(json_path)
    assert rm.to_dict() == structure_graph


def test_to_dict(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.to_dict() == structure_graph


def test_to_dict_truncated(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)

    # Find and extract the child region with ID=2
    id_ = 2
    idx = [c["id"] for c in structure_graph["children"]].index(id_)
    truncated_graph = structure_graph["children"][idx]
    truncated_graph["parent_structure_id"] = None

    assert rm.to_dict(id_) == truncated_graph


def test_color_map(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.color_map == {
        0: (0, 0, 0),
        1: (255, 255, 255),
        2: (255, 240, 0),
        3: (0, 15, 255),
        4: (15, 15, 15),
        5: (240, 240, 240),
    }


@pytest.mark.parametrize(
    ("level", "ids_expect"),
    (
        (0, {0}),
        (1, {1}),
        (2, {2, 3}),
        (3, {4, 5}),
        (4, set()),
    ),
)
def test_ids_at_level(structure_graph, level, ids_expect):
    rm = RegionMeta.from_dict(structure_graph)
    ids = list(rm.ids_at_level(level))
    assert len(ids) == len(ids_expect)
    assert set(ids) == ids_expect


@pytest.mark.parametrize(
    ("id_", "is_leaf"),
    (
        (0, False),
        (1, False),
        (2, False),
        (3, True),
        (4, True),
        (5, True),
    ),
)
def test_is_leaf(structure_graph, id_, is_leaf):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.is_leaf(id_) is is_leaf


@pytest.mark.parametrize(
    ("id_", "parent_id"),
    (
        (0, None),
        (1, 0),
        (2, 1),
        (3, 1),
        (4, 2),
        (5, 2),
    ),
)
def test_parent(structure_graph, id_, parent_id):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.parent(id_) == parent_id


@pytest.mark.parametrize(
    ("id_", "children_expect"),
    (
        (0, {1}),
        (1, {2, 3}),
        (2, {4, 5}),
        (3, set()),
        (4, set()),
        (5, set()),
    ),
)
def test_children(structure_graph, id_, children_expect):
    rm = RegionMeta.from_dict(structure_graph)
    children = list(rm.children(id_))
    assert len(children) == len(children_expect)
    assert set(children) == children_expect


def test_in_region_like(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)

    # Simple regex, equivalent to simple substring tests
    assert rm.in_region_like("root", 1)
    assert rm.in_region_like("child", 4)
    assert not rm.in_region_like("child 3", 2)

    # Invalid region ID
    assert not rm.in_region_like("some region", 999)

    # More complicated regex
    assert rm.in_region_like(r"[c|C]hild", 2)
    assert rm.in_region_like(r"[c|C]hild", 4)
    assert rm.in_region_like(r"\w \d", 5)


def test_ancestors(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.ancestors([4, 5]) == {5, 4, 2, 1}
    assert rm.ancestors(4) == {4, 2, 1}
    assert rm.ancestors(4, include_background=True) == {4, 2, 1, 0}


def test_descendants(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.descendants(5) == {5}
    assert rm.descendants(2) == {2, 4, 5}
    assert rm.descendants([2, 3]) == {2, 3, 4, 5}
    assert rm.descendants([1, 2]) == {1, 2, 3, 4, 5}


def test_depth(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.depth == 3


def test_size(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.size == 5


def test_repr(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert str(rm) == repr(rm)
    assert "RegionMeta" in str(rm)
    assert str(rm.size) in str(rm)
    assert str(rm.depth) in str(rm)


def test_is_valid_id(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    for id_ in (0, 1, 2, 3, 4, 5):
        assert rm.is_valid_id(id_)
    for id_ in (6, 7, 8, 9, 10):
        assert not rm.is_valid_id(id_)


def test_name(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.name(0) == "background"
    assert rm.name(1) == "root"
    assert rm.name(2) == "Child 1"
    assert rm.name(3) == "Child 2"
    assert rm.name(4) == "Grandchild 1"
    assert rm.name(5) == "Grandchild 2"
    assert rm.name(999) == ""


def test_acronym(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.acronym(0) == "bg"
    assert rm.acronym(1) == "root"
    assert rm.acronym(2) == "C1"
    assert rm.acronym(3) == "C2"
    assert rm.acronym(4) == "Gc1"
    assert rm.acronym(5) == "Gc2"
    assert rm.acronym(999) == ""


def test_find_by_name(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.find_by_name("background") == 0
    assert rm.find_by_name("root") == 1
    assert rm.find_by_name("Child 1") == 2
    assert rm.find_by_name("Child 2") == 3
    assert rm.find_by_name("Grandchild 1") == 4
    assert rm.find_by_name("Grandchild 2") == 5
    assert rm.find_by_name("Non-existing") is None
    rm.name("asdf")


def test_find_by_acronym(structure_graph):
    rm = RegionMeta.from_dict(structure_graph)
    assert rm.find_by_acronym("bg") == 0
    assert rm.find_by_acronym("root") == 1
    assert rm.find_by_acronym("C1") == 2
    assert rm.find_by_acronym("C2") == 3
    assert rm.find_by_acronym("Gc1") == 4
    assert rm.find_by_acronym("Gc2") == 5
    assert rm.find_by_acronym("Non-existing") is None


def test_print_regions(structure_graph, capsys):
    rm = RegionMeta.from_dict(structure_graph)

    rm.print_regions()
    stdout, stderr = capsys.readouterr()
    expect_stdout = """\
    root (1)
    ├── Child 1 (2)
    |   ├── Grandchild 1 (4)
    |   └── Grandchild 2 (5)
    └── Child 2 (3)
    """
    assert stdout == textwrap.dedent(expect_stdout)
    assert stderr == ""

    rm.print_regions(max_depth=1)
    stdout, stderr = capsys.readouterr()
    expect_stdout = """\
    root (1)
    ├── Child 1 (2)
    └── Child 2 (3)
    """
    assert stdout == textwrap.dedent(expect_stdout)
    assert stderr == ""

    rm.print_regions(2)
    stdout, stderr = capsys.readouterr()
    expect_stdout = """\
    Child 1 (2)
    ├── Grandchild 1 (4)
    └── Grandchild 2 (5)
    """
    assert stdout == textwrap.dedent(expect_stdout)
    assert stderr == ""

    rm.print_regions(3, max_depth=0)
    stdout, stderr = capsys.readouterr()
    expect_stdout = """\
    Child 2 (3)
    """
    assert stdout == textwrap.dedent(expect_stdout)
    assert stderr == ""


# Write rm.prune_to_root(new_root) -> RegionMeta
# Write ancestors_up_to(ancestor_id, leaves)
#   = rm.prune_to_root(ancestor_id).ancestors(leaves)
# Add typing
