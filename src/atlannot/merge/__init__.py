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
"""Make CCFv2 and CCFv3 atlases compatible by merging annotated regions.

The CCFv2 and CCFv3 atlases differ in the way the regions are annotated.
Sometimes subregions are missing or different region IDs are used. This module
offers functionality to modify the region annotations in a pair of CCFv2 and
CCFv3 atlases in a way that makes them more compatible and comparable.

At the moment there are two different merging strategies: a "coarse" and a
"fine" one and were developed by Dimitri Rodarie. His original merging code
can be found in the ``atlannot.atlas_merge`` module. This module contains
optimized versions of both strategies that have a much faster runtime, but
are the same in functionality as the original ones. Therefore the output
produced by the original "coarse" and "fine" merging strategies and those
implemented in this module should be the same.

The biggest optimization was to not replace labels directly on the atlases
but on the set of unique labels, which is much smaller than the atlas volume.
The labels in the atlases are remapped at the very end of the whole procedure
using fast vectorized numpy operations, see ``atlas_remap``.

Another important optimization was the use of masked numpy arrays instead of
copying the entire volume.
"""
