"""Test for notebook package."""
import matplotlib.patches as patches
import numpy as np
import pytest

from atlannot.atlas import RegionMeta
from atlannot.notebook import create_legend_handles


class TestNotebook:
    @pytest.mark.parametrize("max_value", [1, 5, 10])
    def test_create_handles_legend(self, max_value):
        # Create data
        data = np.arange(0, max_value)
        # Create region meta
        region_meta = RegionMeta()
        for i in range(max_value):
            region_meta.name[i] = f"name_{i}"
        # Create color map
        color_map = {i: np.array([0, 0, 0]) for i in range(max_value)}
        handles = create_legend_handles([data], region_meta, color_map)
        assert len(handles) == max_value
        for handle in handles:
            assert isinstance(handle, patches.Patch)
