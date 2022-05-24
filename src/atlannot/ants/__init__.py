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
"""Registration using ANTsPy."""
import os
import tempfile

import ants
import nibabel
import numpy as np

from atlannot.utils import remap_labels


def register(fixed, moving, **ants_kwargs):
    """Perform an intensity-based registration.

    Parameters
    ----------
    fixed : np.ndarray
        The fixed reference image. Should have d-type float32 or uint32.
    moving : np.ndarray
        The moving image that will be registered to the fixed image. Should
        have d-type float32 or uint32.
    ants_kwargs
        Any additional registration parameters as specified in the
        documentation for `ants.registration`.

    Returns
    -------
    nii_data : np.ndarray
        The transformation of the moving image encoded in displacement fields.
        The shape will depend on whether the registration is 2D or 3D. For
        2D inputs with shape (a, b) the output will have shape (a, b, 1, 1, 2),
        for 3D inputs with shape (a, b, c) the output will have shape
        (a, b, c, 1, 3). The last axis always contains the displacement
        vectors. This data can be used directly in the `transform` function.

    Raises
    ------
    ValueError
        If the shapes of the input images don't match or the d-type of
        input images is not float32 or uint32.
    RuntimeError
        If the resulting transform produced by ANTsPy doesn't have
        the expected form.
    """
    if fixed.shape != moving.shape:
        raise ValueError("Fixed and moving images have different shapes.")
    if fixed.dtype != np.float32 and fixed.dtype != np.uint32:
        raise ValueError("D-type of fixed image is not float32/uint32")
    if moving.dtype != np.float32 and moving.dtype != np.uint32:
        raise ValueError("D-type of moving image is not float32/uint32")

    fixed = ants.from_numpy(fixed)
    moving = ants.from_numpy(moving)
    meta = ants.registration(fixed=fixed, moving=moving, **ants_kwargs)
    with tempfile.TemporaryDirectory() as out_dir:
        out_prefix = os.path.join(out_dir, "out.nii.gz")
        nii_file = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=meta["fwdtransforms"],
            compose=out_prefix,
        )
        nii = nibabel.load(nii_file)
        if not np.allclose(nii.affine, np.diag([-1.0, -1.0, 1.0, 1.0])):
            raise RuntimeError("Unexpected affine part.")
        nii_data = nii.get_fdata()

    # Remove temporary ANTs files
    for file in meta["fwdtransforms"] + meta["invtransforms"]:
        if os.path.exists(file):
            os.remove(file)

    return nii_data


def transform(image, nii_data, **ants_kwargs):
    """Apply a transform to an image.

    Parameters
    ----------
    image : np.ndarray
        The image to transform.
    nii_data : np.ndarray
        The transformation as returned by the `register` function.
    ants_kwargs
        Additional transformation parameters as specified in the
        documentation for `ants.apply_transforms`. Should not contain
        any of these parameters:

            - fixed
            - moving
            - transforms

        A useful parameter that can be specified in `ants_kwargs` is
        `interpolator`. For transforming usual images it can be set
        to "linear", while for annotation atlases the value "genericLabel"
        is more appropriate. See the ANTsPy documentation for more details.

    Returns
    -------
    warped : np.ndarray
        The warped image.

    Raises
    ------
    ValueError
        If d-type of the input image is not float32.
    RuntimeError
        Whenever the internal call of `ants.apply_transforms` fails.
    """
    if image.dtype != np.float32 or image.dtype != np.uint32:
        raise ValueError("D-type of input image is not float32/uint32")

    # Reconstruct the transform. The `register` function asserts that the
    # affine part is always diag(-1, -1, 1, 1).
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    nii = nibabel.Nifti1Image(
        dataobj=nii_data,
        affine=affine,
    )
    # This specifies that for each voxel the data contains a vector
    nii.header.set_intent("vector")

    image = ants.from_numpy(image)
    with tempfile.TemporaryDirectory() as out_dir:
        # ants.apply_transforms needs a file on disk
        nii_file = os.path.join(out_dir, "out.nii.gz")
        nibabel.save(nii, nii_file)

        warped = ants.apply_transforms(
            fixed=image,  # shouldn't matter...
            moving=image,
            transformlist=[nii_file],
            **ants_kwargs,
        )

    # Delete temporary nii file
    if os.path.exists(nii_file):
        os.remove(nii_file)

    # This is only true if the transformation was successful
    if isinstance(warped, ants.ANTsImage):
        warped = warped.numpy()
    else:
        raise RuntimeError("Could not apply the transformation")

    if remapping:
        warped_temp = np.zeros_like(warped)
        for before, after in labels_mapping.items():
            warped_temp[warped == after] = before
        warped = warped_temp

    return warped


def stack_2d_transforms(nii_data_array):
    """Convert a stack of 2D transforms into one 3D transform.

    Instead of transforming moving images slice by slice the 2D
    transformations can be stacked into a 3D transformation with
    zero displacement along the z-axis and then applied directly
    to the 3D volume of slices.

    Parameters
    ----------
    nii_data_array : sequence of array_like
        A sequence of transforms produced by registering a number of
        2D slices.

    Returns
    -------
    nii_data_3d : np.ndarray
        The combined 3D transform.
    """
    # Stack 2d transforms: N x (h, w, 1, 1, 2) => (N, h, w, 1, 2)
    nii_data_stacked = np.stack(nii_data_array).squeeze(3)

    # Create a zero transform for the z-axis: (N, h, w, 1, 1)
    nii_z = np.zeros_like(nii_data_stacked[..., :1])

    # (N, h, w, 1, 1) + (N, h, w, 1, 2) = (N, h, w, 1, 3)
    nii_data_3d = np.concatenate([nii_z, nii_data_stacked], -1)

    return nii_data_3d
