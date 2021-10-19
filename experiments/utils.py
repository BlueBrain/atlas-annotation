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
"""Utility function for experiments."""
import pathlib
import sys


def get_script_file_name():
    """Get script path.

    Returns
    -------
    file_name: pathlib.Path
        File name path.
    """
    file_name = pathlib.Path(sys.argv[0]).stem

    return file_name


def get_base_dir():
    """Get base directory.

    Returns
    -------
    : pathlib.Path
        Base directory path.
    """
    return pathlib.Path(__file__).parent.parent  # = "../"


def get_results_dir():
    """Get results directory.

    Returns
    -------
    : pathlib.Path
        Results directory path.
    """
    return pathlib.Path(__file__).parent / "results"  # = "./results"


def get_data_dir():
    """Get data directory.

    Returns
    -------
    : pathlib.Path
        data directory path.
    """
    return get_base_dir() / "data"


def get_v2_atlas_fine_path():
    """Get v2 atlas fine path.

    Returns
    -------
    : pathlib.Path
        v2 atlas fine path.
    """
    return get_data_dir() / "ccfv2_atlas_fine.nrrd"


def get_v3_atlas_fine_path():
    """Get v3 atlas fine path.

    Returns
    -------
    : pathlib.Path
        v3 atlas fine path.
    """
    return get_data_dir() / "ccfv3_atlas_fine.nrrd"


def get_v2_atlas_coarse_path():
    """Get v2 atlas coarse path.

    Returns
    -------
    : pathlib.Path
        v2 atlas coarse path.
    """
    return get_data_dir() / "ccfv2_atlas_coarse.nrrd"


def get_v3_atlas_coarse_path():
    """Get v3 atlas coarse path.

    Returns
    -------
    : pathlib.Path
        v3 atlas coarse path.
    """
    return get_data_dir() / "ccfv3_atlas_coarse.nrrd"


def get_nissl_path():
    """Get nissl path.

    Returns
    -------
    : pathlib.Path
        nissl path.
    """
    return get_data_dir() / "ara_nissl_25.nrrd"


def get_avg_brain_path():
    """Get average brain path.

    Returns
    -------
    : pathlib.Path
        Average brain path.
    """
    return get_data_dir() / "average_template_25.nrrd"


def can_write_to_dir(output_dir):
    """Return boolean if possible to write in the specified directory.

    Parameters
    ----------
    output_dir : pathlib.Path
        Output directory to verify.

    Returns
    -------
    : bool
        If True, can write in the given directory.
        Otherwise, can not overwrite in the directory.
    """
    if output_dir.exists():
        print(f"Output directory {output_dir} exists.")
        answer = input("Continue and overwrite? (y/n) ")
        if answer != "y":
            return False
    else:
        output_dir.mkdir(parents=True)
    return True
