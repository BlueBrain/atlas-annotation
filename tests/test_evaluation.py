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
import pathlib

import numpy as np

from atlannot.evaluation import (
    conditional_entropy,
    dist_entropy,
    evaluate,
    evaluate_region,
    iou,
    jaggedness,
)
from atlannot.region_meta import RegionMeta

REGION_META_PATH = pathlib.Path(__file__).parent / "data" / "structure_graph_mini.json"


def test_compute_jaggedness():
    child_id = 3
    parent_id = 1
    region_meta = RegionMeta.load_json(REGION_META_PATH)
    volume = np.zeros((10, 10, 10)).astype(np.int)
    volume[1:5, 1:5, 1:5] = parent_id
    volume[4:8, 4:8, 4:8] = parent_id
    score = jaggedness(volume, region_id=parent_id, region_meta=region_meta)
    assert isinstance(score, float)
    assert not np.isnan(score)

    # Check that if parent does not exist, but parent do, computation is still
    # going through thanks to the mask
    volume = np.zeros((10, 10, 10)).astype(np.int)
    volume[1:5, 1:5, 1:5] = child_id
    volume[4:8, 4:8, 4:8] = child_id
    score = jaggedness(volume, region_id=parent_id, region_meta=region_meta)
    assert isinstance(score, float)
    assert not np.isnan(score)

    # If label not present in the volume, then the score is NaN
    volume = np.ones((10, 10, 10)).astype(np.int)
    score = jaggedness(volume, region_id=3, region_meta=region_meta)
    assert isinstance(score, float)
    assert np.isnan(score)


def test_compute_iou():
    labels = np.arange(10)
    volume = labels * np.ones((10, 10, 10)).astype(np.int)
    region_meta = RegionMeta.load_json(REGION_META_PATH)
    for region_id in [1, 2, 3]:
        score = iou(volume, volume, region_id=region_id, region_meta=region_meta)
        assert isinstance(score, float)
        assert score == 1.0

    # Check that if parent does not exist, but parent do, computation is still
    # going through thanks to the mask
    child_id = 3
    parent_id = 1
    volume = child_id * np.ones((10, 10, 10))
    score = iou(volume, volume, region_id=parent_id, region_meta=region_meta)
    assert isinstance(score, float)
    assert score == 1.0

    # If label not present in the volume, then the score is NaN
    volume = np.ones((10, 10, 10)).astype(np.int)
    score = iou(volume, volume, region_id=3, region_meta=region_meta)
    assert isinstance(score, float)
    assert np.isnan(score)


def test_compute_region_entropy():
    volume = np.ones((10, 10, 10))
    entropy = dist_entropy(volume)
    assert isinstance(entropy, float)
    assert entropy == 0

    nissl = np.random.random((10, 10, 10))
    entropy = dist_entropy(nissl[volume == 1])
    assert isinstance(entropy, float)
    assert entropy > 0


def test_compute_conditional_entropy():
    labels = np.arange(10)
    atlas = labels * np.ones((10, 10, 10))
    nissl = np.random.random((10, 10, 10))
    cond_entr = conditional_entropy(nissl, atlas)
    assert isinstance(cond_entr, float)
    assert cond_entr > 0


def test_evaluate_region():
    labels = np.arange(10)
    volume = labels * np.ones((10, 10, 10)).astype(np.int)
    rm = RegionMeta.load_json("tests/data/structure_graph_mini.json")
    results = evaluate_region([2, 3], volume, volume, rm)
    assert isinstance(results, dict)
    assert "jaggedness" in results.keys()
    assert ["global", "per_region"] == list(results["jaggedness"].keys())
    assert "iou" in results.keys()
    assert ["global", "per_region"] == list(results["iou"].keys())


def test_evaluate():
    labels = np.arange(10)
    volume = labels * np.ones((10, 10, 10))
    nissl = np.random.random((10, 10, 10))

    # Create fake Region Meta
    rm = RegionMeta.load_json("tests/data/structure_graph_mini.json")
    fake_regions_to_evaluate = {
        "Child 1": [2],
        "Child 2": [3],
    }
    results = evaluate(volume, nissl, volume, rm, fake_regions_to_evaluate)
    assert isinstance(results, dict)
    for name in fake_regions_to_evaluate:
        assert name in results.keys()
        assert "jaggedness" in results[name].keys()
        assert "iou" in results[name].keys()
    assert "global" in results.keys()
    assert ["brain_entropy", "conditional_entropy"] == list(results["global"].keys())
