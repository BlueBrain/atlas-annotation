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
"""Dimitri's coarse and fine atlas merging."""
import argparse
import json
import sys

import nrrd

from atlannot.merge_original.corse import coarse_merge
from atlannot.merge_original.fine import fine_merge


def parse_args():
    """Parse command line arguments.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("merge_type", choices=("coarse", "fine"))
    parser.add_argument("ccfv2_atlas")
    parser.add_argument("ccfv3_atlas")
    parser.add_argument("brain_regions")
    parser.add_argument("ccfv2_atlas_merged")
    parser.add_argument("ccfv3_atlas_merged")
    args = parser.parse_args()

    return args


def main():
    """Run atlas merging."""
    args = parse_args()

    # Read data
    ccfv2, ccfv2_header = nrrd.read(args.ccfv2_atlas)
    ccfv3, ccfv3_header = nrrd.read(args.ccfv3_atlas)
    with open(args.brain_regions) as fp:
        brain_regions_data = json.load(fp)
        brain_regions = brain_regions_data["msg"][0]

    # Merge
    if args.merge_type == "coarse":
        ccfv2_corrected, ccfv3_corrected = coarse_merge(ccfv2, ccfv3, brain_regions)
    elif args.merge_type == "fine":
        ccfv2_corrected, ccfv3_corrected = fine_merge(ccfv2, ccfv3, brain_regions)
    else:
        # Should be unreachable
        raise ValueError(f"Unknown merge type: {args.merge_type}")

    # Save
    nrrd.write(args.ccfv2_atlas_merged, ccfv2_corrected, ccfv2_header)
    nrrd.write(args.ccfv3_atlas_merged, ccfv3_corrected, ccfv3_header)


if __name__ == "__main__":
    sys.exit(main())
