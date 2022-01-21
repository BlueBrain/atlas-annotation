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
from unittest import mock

import numpy as np

from atlannot.region_meta import RegionMeta
from atlannot.evaluation import (
    REGIONS_TO_EVALUATE,
    compute_entropies,
    compute_iou,
    compute_jaggedness,
    compute_region_entropy,
    evaluate,
)


def test_compute_jaggedness():
    labels = np.arange(10)
    volume = labels * np.ones((10, 10, 10))
    results = compute_jaggedness(volume)
    assert isinstance(results, dict)
    for label in labels:
        if label != 0:
            assert label in list(results.keys())

    results = compute_jaggedness(volume, region_ids=[1, 2, 3])
    assert isinstance(results, dict)
    for label in [1, 2, 3]:
        assert label in list(results.keys())


def test_compute_iou():
    labels = np.arange(10)
    volume = labels * np.ones((10, 10, 10))
    results = compute_iou(volume, volume)
    assert isinstance(results, dict)
    for label in labels:
        if label != 0:
            assert label in list(results.keys())


def test_compute_region_entropy():
    volume = np.ones((10, 10, 10))
    entropy = compute_region_entropy(volume, volume, label_value=1)
    assert isinstance(entropy, float)
    assert entropy == 0

    nissl = np.random.random((10, 10, 10))
    entropy = compute_region_entropy(nissl, volume, label_value=1)
    assert isinstance(entropy, float)
    assert entropy > 0


def test_compute_entropies():
    labels = np.arange(10)
    atlas = labels * np.ones((10, 10, 10))
    nissl = np.random.random((10, 10, 10))
    brain_entropy, conditional_entropy = compute_entropies(nissl, atlas)
    assert isinstance(brain_entropy, float)
    assert brain_entropy > 0
    assert isinstance(conditional_entropy, float)
    assert conditional_entropy > 0


def test_evaluate():
    labels = np.arange(10)
    volume = labels * np.ones((10, 10, 10))
    nissl = np.random.random((10, 10, 10))

    # Create fake Region Meta
    with mock.patch.object(RegionMeta, "descendants") as descendants_mocked:
        descendants_mocked.return_value = [2, 3]
        region_meta = RegionMeta()
        for _, id_list in REGIONS_TO_EVALUATE.items():
            for id_ in id_list:
                region_meta.level[id_] = 2

        results = evaluate(volume, nissl, volume, region_meta)
        print(results)
        assert False





    print(results)
    assert False
