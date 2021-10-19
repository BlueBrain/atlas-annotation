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
"""Test of the atlas module."""
import warnings

import numpy as np
import pytest
from tqdm import tqdm

from atlannot.atlas.align import get_misalignment, specific_label_iou, unfurl_regions
from atlannot.atlas.region_meta import RegionMeta


class TestAlign:
    def test_unfurl_regions(self):
        # Definition of region_meta
        region_meta = RegionMeta()
        region_meta.parent_id[2] = 1
        region_meta.parent_id[3] = 1
        region_meta.parent_id[1] = region_meta.background_id
        region_meta.level[1] = 1
        region_meta.level[2] = 2
        region_meta.level[3] = 2
        # Definition of the atlas
        data_1 = np.zeros((1, 20, 20))
        data_1[0, 5:10, 5:15] = 2
        data_1[0, 10:15, 5:15] = 3

        results = unfurl_regions(data_1, region_meta, tqdm)
        assert results.shape == (3, 1, 20, 20)
        assert np.all(results[0][0] == data_1[0])
        data_2 = np.zeros((20, 20))
        data_2[5:15, 5:15] = 1
        assert np.all(results[1][0] == data_2)
        data_3 = np.zeros((20, 20))
        assert np.all(results[2][0] == data_3)

    def test_get_misalignement(self):
        data_1 = np.zeros((20, 20))
        data_2 = np.zeros((10, 20))

        # Check that data have the same shape
        with pytest.raises(ValueError):
            get_misalignment(data_1, data_2)

        # Check that misalignement values are consistent
        data_2 = np.zeros((20, 20))
        assert get_misalignment(data_1, data_2) == 0
        data_1[5:10, 5:15] = 1
        data_1[10:15, 5:15] = 2
        data_2[5:15, 5:15] = 1
        assert get_misalignment(data_1, data_2, fg_only=False) == 2 / 16
        assert get_misalignment(data_1, data_2, fg_only=True) == 2 / 4

    def test_specific_label_iou(self):
        data_1 = np.zeros((20, 20))
        data_2 = np.zeros((10, 20))

        # Check that data have the same shape
        with pytest.raises(ValueError):
            specific_label_iou(data_1, data_2, 0)

        # Check that a warning is raised
        data_2 = np.zeros((20, 20))
        with warnings.catch_warnings(record=True) as w:
            specific_label_iou(data_1, data_2, 10)
            assert len(w) == 1

        # Check that IOU values are consistent
        assert specific_label_iou(data_1, data_2, 0) == 1
        data_1[5:15, 5:15] = 1
        data_2[10:20, 10:20] = 1
        assert specific_label_iou(data_1, data_2, 1) == 1 / 7
