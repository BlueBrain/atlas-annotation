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
from __future__ import annotations

import pathlib

import nrrd
import numpy as np
import pytest

import atlannot.utils
from atlannot import load_volume
from atlannot.utils import (
    Remapper,
    add_middle_line,
    atlas_symmetry_score,
    compose,
    image_convolution,
    load_nrrd,
    merge,
    remap_labels,
    saving_results,
    split_halfs,
    stain_symmetry_score,
    write_transform,
)


@pytest.mark.parametrize("value", [0, 0.2, 1])
def test_add_middle_line(recwarn, value):
    """Test add middle line method."""
    image = np.zeros((10, 10))
    new_image = add_middle_line(image, axis=0, thickness=2, value=value)
    image_res = image.copy()
    image_res[4:6, :] = 1 * value
    assert isinstance(new_image, np.ndarray)
    assert new_image.shape == image.shape
    assert np.all(new_image == image_res)

    image = np.zeros((30, 30))
    new_image = add_middle_line(image, axis=1, thickness=4, value=value)
    image_res = image.copy()
    image_res[:, 13:17] = 1 * value
    assert isinstance(new_image, np.ndarray)
    assert new_image.shape == image.shape
    assert np.all(new_image == image_res)

    image = np.zeros((30, 30))
    _ = add_middle_line(image, axis=1, thickness=5)
    assert len(recwarn) == 1


@pytest.mark.parametrize("scale_1", [1, 0.1])
@pytest.mark.parametrize("scale_2", [1, 0.55])
def test_merge(scale_1, scale_2):
    """Test merging of two images."""
    shape = (10, 10)
    image1 = np.zeros(shape)
    image2 = np.ones(shape, dtype=np.float32)
    result = merge(image1, image2, scale_1, scale_2)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape
    assert np.all(result == image2 * scale_2)

    image2 = np.zeros(shape)
    image2[4:6, 4:6] = 1
    image1[0:2, 0:2] = 1
    result = merge(image1, image2, scale_1, scale_2)
    expected = np.zeros(shape, dtype=np.float32)
    expected[4:6, 4:6] = 1 * scale_2
    expected[0:2, 0:2] = 1 * scale_1
    assert isinstance(result, np.ndarray)
    assert result.shape == shape
    assert np.all(result == expected)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shape", [(10, 20), (40, 52)])
def test_split_half(axis, shape):
    """Test splitting half images."""
    image = np.zeros(shape)
    half1, half2 = split_halfs(image, axis=axis)
    new_shape = tuple(
        [int(shape[i] / 2) if i == axis else shape[i] for i in range(len(shape))]
    )
    assert half1.shape == new_shape
    assert half2.shape == new_shape


@pytest.mark.parametrize("n_atls", [1, 5, 10])
@pytest.mark.parametrize("seed", [None, 50])
def test_remap_labels(n_atls, seed):
    """Test remapping of labels."""
    atl = np.arange(0, 100)
    inp_atl = atl * 5
    inp = [inp_atl for _ in range(n_atls)]
    result_atls, mapping = remap_labels(inp, seed=seed)
    assert len(result_atls) == n_atls
    for result in result_atls:
        assert np.all(np.unique(result) == atl)
        assert result.shape == atl.shape
    assert isinstance(mapping, dict)

    atl = np.arange(0, 100)
    atl2 = np.arange(0, 200)
    result_atls, mapping = remap_labels([atl, atl2], seed=seed)
    assert len(result_atls) == 2
    assert result_atls[0].shape == atl.shape
    assert result_atls[1].shape == atl2.shape
    assert isinstance(mapping, dict)


@pytest.mark.parametrize("shape", [(10, 10), (22, 11, 1, 2)])
def test_write_transform(tmpdir, shape):
    """Test writing transformation."""
    path = pathlib.Path(tmpdir) / "transform.nii"
    nii_data = np.zeros(shape)
    write_transform(nii_data, path)
    assert path.exists()


@pytest.mark.parametrize("shape", [(10, 10, 1, 1, 2), (22, 11, 1, 1, 2)])
def test_compose(shape):
    """Test composition of transformations."""
    with pytest.raises(ValueError):
        compose()
    with pytest.raises(ValueError):
        t1 = np.zeros((10, 10))
        t2 = np.zeros((20, 20))
        compose(t1, t2)

    t1, t2 = np.ones(shape), np.ones(shape)
    result = compose(t1, t2)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape
    assert np.all(np.unique(result) == [1, 2])


def test_stains_symmetry_score():
    """Test computation of symmetry score."""
    img = np.ones((100, 100))
    ssmi = stain_symmetry_score(img)
    assert ssmi == 1


def test_atlas_symmetry_score():
    """Test computation of symmetry score."""
    img = np.ones((10, 10))
    mis = atlas_symmetry_score(img)
    assert mis == 1
    img[:, 5:] = 0
    mis = atlas_symmetry_score(img)
    assert mis == 0


def test_load_nrrd(tmpdir):
    """Test loading of nrrd files."""
    path = str(tmpdir) + "test.nrrd"
    img = np.reshape(np.arange(0, 100), (10, 10))
    nrrd.write(path, img)

    data = load_nrrd(pathlib.Path(path), norm=False)
    assert isinstance(data, np.ndarray)
    assert data.shape == img.shape
    assert np.all(data == img)

    data = load_nrrd(pathlib.Path(path), norm=True)
    new_img = img / 99
    assert np.all(data == new_img.astype(np.float32))


@pytest.mark.parametrize("binary", [True, False])
def test_image_convolution(binary):
    """Test convolution of images."""
    img = np.zeros((10, 10))
    img[4:6, :] = 1
    kernel = np.ones((3, 3))
    result = image_convolution(img, kernel, binary=binary)
    assert isinstance(result, np.ndarray)
    if binary:
        assert np.all(np.unique(result) == [0, 1])


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize(
    "kernel",
    [
        "edge_sobel",
        "edge_laplacian_thin",
        "edge_laplacian_thick",
    ],
)
def test_kernels(binary, kernel):
    """Test different convolutions of images."""
    conv = getattr(atlannot.utils, kernel)
    img = np.zeros((10, 10))
    img[4:6, :] = 1
    result = conv(img, binary=binary)
    assert isinstance(result, np.ndarray)
    if binary:
        assert np.all(np.unique(result) == [0, 1])


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("extension", [".nrrd", ".npy"])
def test_load_volume(tmpdir, extension, normalize):
    """Test loading paths."""
    img = np.zeros((10, 10))
    img[4:6, 4:6] = 10
    path = pathlib.Path(str(tmpdir) + "test" + extension)
    if extension == ".nrrd":
        nrrd.write(str(path), img)
    else:
        np.save(path, img)

    result = load_volume(path, normalize=normalize)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)
    if normalize:
        assert np.min(result) == 0
        assert np.max(result) == 1

    # Invalid Path
    path = pathlib.Path("wrong_path")
    result = load_volume(path, normalize=normalize)
    assert result is None

    # Invalid extension
    with pytest.raises(ValueError):
        txt_file_path = pathlib.Path(tmpdir) / "file.txt"
        txt_file_path.touch()
        path = txt_file_path
        load_volume(path, normalize=normalize)


def test_saving_results(tmpdir):
    """Test the saving of the results."""
    output_dir = pathlib.Path(tmpdir) / "tests" / "experiment1"
    saving_results(output_dir=output_dir)
    assert output_dir.exists()
    names = ["fixed.npy", "moving.npy", "df.npy", "registered.npy", "description.txt"]
    for name in names:
        assert not (output_dir / name).exists()

    img = np.zeros((10, 10))
    saving_results(
        output_dir=output_dir,
        img_reg=img,
        img_mov=img,
        img_ref=img,
        df=img,
        description="test",
    )
    for name in names:
        assert (output_dir / name).exists()


class TestRemapper:
    def test_invalid_input(self):

        with pytest.raises(ValueError, match="No volume provided"):
            Remapper()

        with pytest.raises(TypeError, match="numpy arrays"):
            Remapper([3423])

        with pytest.raises(TypeError, match="needs to be uint32"):
            array_1 = np.array([2, 5], dtype=np.int32)
            array_2 = np.array([2, 3], dtype=np.uint32)
            Remapper(array_1, array_2)

    def test_overall(self):
        volume_0 = np.array(
            [
                [10, 30],
                [5, 79],
                [10, 5],
            ],
            dtype=np.uint32,
        )
        volume_1 = np.array(
            [
                [35, 30],
                [52, 10],
                [10, 5],
            ],
            dtype=np.uint32,
        )
        remapper = Remapper(volume_0, volume_1)

        assert len(remapper) == 2

        assert remapper.old2new == {
            5: 0,
            10: 1,
            30: 2,
            35: 3,
            52: 4,
            79: 5,
        }
        assert remapper.new2old == {
            0: 5,
            1: 10,
            2: 30,
            3: 35,
            4: 52,
            5: 79,
        }

        with pytest.raises(IndexError):
            remapper.remap_old_to_new(-1)

        with pytest.raises(IndexError):
            remapper.remap_old_to_new(2)

        # Test remapping
        new_volume_0 = remapper.remap_old_to_new(0)
        new_volume_0_expected = np.array(
            [
                [1, 2],
                [0, 5],
                [1, 0],
            ],
            dtype=np.uint32,
        )
        assert new_volume_0.dtype == np.uint32
        np.testing.assert_array_equal(new_volume_0, new_volume_0_expected)

        new_volume_1 = remapper.remap_old_to_new(1)
        new_volume_1_expected = np.array(
            [
                [3, 2],
                [4, 1],
                [1, 0],
            ],
            dtype=np.uint32,
        )
        assert new_volume_1.dtype == np.uint32
        np.testing.assert_array_equal(new_volume_1, new_volume_1_expected)

        # Undo remapping
        result = remapper.remap_new_to_old(np.array([5, 1, 3], dtype=np.uint32))
        expected_result = np.array([79, 10, 35], dtype=np.uint32)

        np.testing.assert_array_equal(expected_result, result)
