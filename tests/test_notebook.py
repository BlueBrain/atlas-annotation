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
"""Test for notebook package."""
import matplotlib.patches as patches
import numpy as np
import pytest

from atlannot.notebook import create_legend_handles
from atlannot.region_meta import RegionMeta


class TestNotebook:
    @pytest.mark.parametrize("max_value", [1, 5, 10])
    def test_create_handles_legend(self, max_value):
        # Create data
        data = np.arange(0, max_value)
        # Create region meta
        region_meta = RegionMeta()
        for i in range(max_value):
            region_meta.name_[i] = f"name_{i}"
        # Create color map
        color_map = {i: np.array([0, 0, 0]) for i in range(max_value)}
        handles = create_legend_handles([data], region_meta, color_map)
        assert len(handles) == max_value
        for handle in handles:
            assert isinstance(handle, patches.Patch)
