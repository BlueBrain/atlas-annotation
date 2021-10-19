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
"""Combine atlas annotation and fiber files into one.

For CCFv2 the Allen Institute provide the atlas annotations in two separate
files, one for with the actual annotations (e.g. annotation_25.nrrd) and one
for the fibers (e.g. annotationFiber_25.nrrd). This script combines both
annotation files into one.
"""
import argparse
import sys

import nrrd


def parse_args():
    """Parse command line arguments.

    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_file")
    parser.add_argument("annotation_fiber_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    return args


def main():
    """Combine atlas and fiber annotations."""
    args = parse_args()

    # Read files
    annotation, annotation_header = nrrd.read(args.annotation_file)
    annotation_fiber, _ = nrrd.read(args.annotation_fiber_file)

    # Combine annotation and fiber data
    annotation[annotation_fiber > 0] = annotation_fiber[annotation_fiber > 0]

    # Write result
    nrrd.write(args.output_file, annotation, annotation_header)


if __name__ == "__main__":
    sys.exit(main())
