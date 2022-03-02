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


def test_compute_jaggedness():
    labels = np.arange(10)
    volume = labels * np.ones((10, 10, 10))
    results = jaggedness(volume)
    assert isinstance(results, dict)
    for label in labels:
        if label != 0:
            assert label in list(results.keys())

    results = jaggedness(volume, region_ids=[1, 2, 3])
    assert isinstance(results, dict)
    for label in [1, 2, 3]:
        assert label in list(results.keys())

    # If label not present in the volume, then skipped
    absent_label = [
        11,
    ]
    results = jaggedness(volume, region_ids=absent_label)
    assert results == {}


def test_compute_iou():
    labels = np.arange(10)
    volume = labels * np.ones((10, 10, 10))
    results = iou(volume, volume)
    assert isinstance(results, dict)
    for label in labels:
        if label != 0:
            assert label in list(results.keys())

    # If label not present in any of the volumes, then the result is NaN
    scores = iou(volume, volume, region_ids=[11])
    assert scores.keys() == {11}
    assert np.isnan(scores[11])


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
    volume = labels * np.ones((10, 10, 10))
    rm = RegionMeta.load_json("tests/data/structure_graph_mini.json")
    results = evaluate_region([2, 3], volume, volume, rm)
    assert isinstance(results, dict)
    assert "jaggedness" in results.keys()
    assert "iou" in results.keys()


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
