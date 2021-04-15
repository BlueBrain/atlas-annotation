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
