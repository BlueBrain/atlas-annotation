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
import nibabel
import numpy as np
import pytest

import atlannot.ants


def test_register(monkeypatch):
    # 2D registration
    fixed = np.random.randn(10, 20).astype(np.float32)
    moving = np.random.randn(10, 20).astype(np.float32)
    nii = atlannot.ants.register(fixed, moving)
    assert isinstance(nii, np.ndarray)
    assert nii.shape == (10, 20, 1, 1, 2)

    # 3D registration
    fixed = np.random.randn(5, 10, 20).astype(np.float32)
    moving = np.random.randn(5, 10, 20).astype(np.float32)
    nii = atlannot.ants.register(fixed, moving)
    assert isinstance(nii, np.ndarray)
    assert nii.shape == (5, 10, 20, 1, 3)

    # Different shapes
    fixed = np.random.randn(10, 20).astype(np.float32)
    moving = np.random.randn(30, 40).astype(np.float32)
    with pytest.raises(ValueError, match="shape"):
        atlannot.ants.register(fixed, moving)

    # Wrong fixed image dtype
    fixed = np.random.randn(10, 20).astype(np.float64)
    moving = np.random.randn(10, 20).astype(np.float32)
    with pytest.raises(ValueError, match="fixed.* float32/uint32"):
        atlannot.ants.register(fixed, moving)

    # Wrong moving image dtype
    fixed = np.random.randn(10, 20).astype(np.float32)
    moving = np.random.randn(10, 20).astype(np.float64)
    with pytest.raises(ValueError, match="moving.* float32"):
        atlannot.ants.register(fixed, moving)

    # Wrong affine part
    def mock_nibabel_load(_):
        affine = np.diag([1.0, 2.0, 3.0, 4.0])
        nii = nibabel.Nifti1Image(
            dataobj=np.random.randn(10, 20, 1, 1, 2),
            affine=affine,
        )
        return nii

    monkeypatch.setattr("atlannot.ants.nibabel.load", mock_nibabel_load)
    fixed = np.random.randn(10, 20).astype(np.float32)
    moving = np.random.randn(10, 20).astype(np.float32)
    with pytest.raises(RuntimeError, match="affine"):
        atlannot.ants.register(fixed, moving)


def test_transform(monkeypatch):
    # 2D
    image = np.random.randn(10, 20).astype(np.float32)
    nii_data = np.random.randn(10, 20, 1, 1, 2)
    atlannot.ants.transform(image, nii_data)

    # 2D - atlas
    atlas = np.random.randint(5, size=(10, 20)).astype(np.float32)
    nii_data = np.random.randn(10, 20, 1, 1, 2)
    atlannot.ants.transform(atlas, nii_data, interpolator="genericLabel")

    # 3D
    image = np.random.randn(5, 10, 20).astype(np.float32)
    nii_data = np.random.randn(5, 10, 20, 1, 3)
    atlannot.ants.transform(image, nii_data)

    # Wrong image dtype
    image = np.random.randn(5, 10, 20).astype(np.float64)
    nii_data = np.random.randn(5, 10, 20, 3)
    with pytest.raises(ValueError, match="float32"):
        atlannot.ants.transform(image, nii_data)

    # Error during transform
    def mock_apply_transforms(*_1, **_2):
        return 1

    monkeypatch.setattr("atlannot.ants.ants.apply_transforms", mock_apply_transforms)
    image = np.random.randn(5, 10, 20).astype(np.float32)
    nii_data = np.random.randn(5, 1, 1, 1, 3)
    with pytest.raises(RuntimeError):
        atlannot.ants.transform(image, nii_data)


def test_stack_2d_transforms():
    nii_data_array = [np.random.randn(10, 20, 1, 1, 2) for _ in range(5)]
    nii_data_3d = atlannot.ants.stack_2d_transforms(nii_data_array)

    assert nii_data_3d.shape == (5, 10, 20, 1, 3)
