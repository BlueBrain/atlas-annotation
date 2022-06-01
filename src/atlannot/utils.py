# Copyright 2022, Blue Brain Project, EPFL
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
"""Various utilities. (Refactoring in the near future)."""
from __future__ import annotations

import math
import os
import pathlib
import tempfile
import warnings

import ants
import nibabel
import nrrd
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

from atlannot.atlas import get_misalignment


# LOADING PART
def load_nrrd(path, norm=True):
    """Load nrrd file.

    Parameters
    ----------
    path : pathlib.Path or str
        Path of the file to read (with nrrd format).
    norm : bool
        If True, intensities of the numpy are normalized between 0 and 1.

    Returns
    -------
    data : np.ndarray
        Loaded file.
    """
    data, header = nrrd.read(path)
    if norm:
        data = data.astype(np.float32)
        data = (data - data.min()) / (data.max() - data.min())
    return data


def create_description(name, args=None):
    """Create description from name and args.

    Parameters
    ----------
    name : str
        Name of the experiment.
    args : None or argparse.Namespace
        Information of the experiment.

    Returns
    -------
    description : str
        Entire description of the experiment.
    """
    description = f"experiment_name: {name} \n"
    if args is not None:
        for k, v in vars(args).items():
            description += f"{k:<32}: {v} \n"

    return description


def saving_results(
    output_dir, img_ref=None, img_mov=None, img_reg=None, df=None, description=None
):
    """Save results under output_dir.

    Parameters
    ----------
    output_dir : pathlib.Path
        Output directory where to save all the results.
    img_ref : np.ndarray
        Image used as reference image during the registration.
    img_mov : np.ndarray
        Image used as moving image during the registration.
    img_reg : np.ndarray
        Image resulting after the registration
    df : np.ndarray
        Displacement computed with the registration.
    description : str
        Description of the experiment.
    """
    if not output_dir.exists():
        pathlib.Path.mkdir(output_dir, parents=True)

    if img_ref is not None:
        np.save(str(output_dir / "fixed.npy"), img_ref)

    if img_mov is not None:
        np.save(str(output_dir / "moving.npy"), img_mov)

    if img_reg is not None:
        np.save(str(output_dir / "registered.npy"), img_reg)

    if df is not None:
        np.save(str(output_dir / "df.npy"), df)

    if description is not None:
        with open(output_dir / "description.txt", "w") as fp:
            fp.write(description)


# IMAGE MANIPULATION
def add_middle_line(image, axis=0, thickness=1, value=1):
    """Add middle line on a given image.

    Parameters
    ----------
    image : np.ndarray
        Input image which needs a middle line along one axis.
    axis : int
        Axis along which to create the middle line.
    thickness : int
        Thickness of the middle line.
    value : float
        Pixel intensity of the middle line.

    Returns
    -------
    new_image : np.ndarray
        Image as the input image with the middle line as specified.
    """
    width = image.shape[axis]
    if width % 2 != thickness % 2:
        need = "even" if width % 2 == 0 else "odd"
        warnings.warn(
            f"Warning: middle line not exactly in the middle. "
            f"Try changing line thickness to an {need} number",
            UserWarning,
        )

    # Find position of middle line
    pos = int((width - thickness) / 2)

    # Prepare indices of the form (:, ..., (pos, pos+1, ... pos+thickness), ..., :)
    # representing the line
    ids = tuple(
        tuple(range(pos, pos + thickness)) if i == axis else slice(None)
        for i in range(len(image.shape))
    )

    # Set the line
    new_image = image.copy()
    new_image[ids] = value

    return new_image


def merge(img1, img2, scale1=1.0, scale2=1.0):
    """Merge two images by taking the maximum intensity.

    Parameters
    ----------
    img1 : np.ndarray
        First input image.
    img2 : np.ndarray
        Second input image. Should have the same shape as img1.
    scale1 : float
        Value used to multiply all pixels intensities of img1.
    scale2 : float
        Value used to multiply all pixels intensities of img1.

    Returns
    -------
    img_res : np.ndarray
        Resulting image of the same shape as img1 and img2.
    """
    img_res = np.max([img1 * scale1, img2 * scale2], axis=0).astype(np.float32)
    return img_res


def split_halfs(img, axis):
    """Split image into two half of it. The right part is flipped.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    axis : int
        Axis along which to cut the image into two.

    Returns
    -------
    half_1 : np.ndarray
        First resulting half of the input image.
    half_2 : np.ndarray
        Second resulting half of the input image, flipped.
    """
    ids_1 = tuple(
        slice(0, math.floor(dim / 2)) if i == axis else slice(None)
        for i, dim in enumerate(img.shape)
    )
    ids_2 = tuple(
        slice(math.ceil(dim / 2), dim) if i == axis else slice(None)
        for i, dim in enumerate(img.shape)
    )
    half_1 = img[ids_1]
    half_2 = np.flip(img[ids_2], axis=axis)

    return half_1, half_2


def image_convolution(img, kernel, kernel_trans=None, *, binary=True):
    """Compute convolution of an image.

    Parameters
    ----------
    img : np.ndarray
        Input image for which it is needed to compute a convolution.
    kernel : np.ndarray
        Kernel to apply to the input image.
    kernel_trans : np.ndarray or None
        If specified, a second kernel is applied to the input image.
    binary : bool
        If True, the return image has only binary values,
        otherwise the return image pixel values are results of the convolution.

    Returns
    -------
    grad : np.ndarray
        Resulting image after convolution.
    """
    grad = ndimage.convolve(img, kernel)

    if kernel_trans is not None:
        grad_trans = ndimage.convolve(img, kernel_trans)
        grad = np.hypot(grad, grad_trans)

    if binary:
        grad = (grad > 0).astype(grad.dtype)

    return grad


def edge_sobel(img, binary=True):
    """Apply Sobel to an image.

    Parameters
    ----------
    img : np.ndarray
        Input image for which it is needed to compute a convolution.
    binary : bool
        If True, the return image has only binary values,
        otherwise the return image pixel values are results of the convolution.

    Returns
    -------
    result : np.ndarray
        Resulting Sobel convolution.
    """
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    result = image_convolution(img, kx, ky, binary=binary)
    return result


def edge_laplacian_thick(img, binary=True):
    """Apply Laplacian to an image (results in thick border).

    Parameters
    ----------
    img : np.ndarray
        Input image for which it is needed to compute a convolution.
    binary : bool
        If True, the return image has only binary values,
        otherwise the return image pixel values are results of the convolution.

    Returns
    -------
    result : np.ndarray
        Resulting Laplacian convolution.
    """
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    result = image_convolution(img, kernel, binary=binary)
    return result


def edge_laplacian_thin(img, binary=True):
    """Apply Laplacian to an image (results in thin border).

    Parameters
    ----------
    img : np.ndarray
        Input image for which it is needed to compute a convolution.
    binary : bool
        If True, the return image has only binary values,
        otherwise the return image pixel values are results of the convolution.

    Returns
    -------
    result : np.ndarray
        Resulting Laplacian convolution.
    """
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    result = image_convolution(img, kernel, binary=binary)
    return result


def remap_labels(atlases, seed=None):
    """Remap atlas labels between (0, number of labels).

    Parameters
    ----------
    atlases : Iterable of np.ndarray
        List of atlases.

    seed : None or int
        If None, the mapping is done after sorting the list of unique labels.
        If int, the labels are shuffled before the creation of the mapping
    Returns
    -------
    new_atlases : list of np.ndarray
        List of remapped atlases.

    mapping : dict of int
        Dictionary containing the mapping between previous and new labels.
    """
    new_atlases = [atl.copy() for atl in atlases]
    unique_labels = np.unique(np.concatenate([np.unique(atl) for atl in atlases]))
    if seed is not None:
        np.random.seed(seed=seed)
        new_labels = np.arange(len(unique_labels))
        np.random.shuffle(new_labels)
        mapping = {v: int(i) for i, v in zip(list(new_labels), unique_labels)}
    else:
        mapping = {v: i for i, v in enumerate(sorted(unique_labels))}

    for atl, new_atl in zip(atlases, new_atlases):
        for value_from, value_to in mapping.items():
            new_atl[atl == value_from] = value_to
    return new_atlases, mapping


# TRANSFORMATION COMPOSITION
def write_transform(nii_data, path):
    """Write transform into file.

    Parameters
    ----------
    nii_data : np.ndarray
        Transformation to save.
    path : str
        Path where to save the transformation.
    """
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    nii_img = nibabel.Nifti1Image(
        dataobj=nii_data,
        affine=affine,
    )
    nii_img.header.set_intent("vector")
    nibabel.save(nii_img, path)


def compose(*transforms):
    """Compose several transformations into one.

    Parameters
    ----------
    transforms : Iterable of np.ndarray
        List of transformations to compose. All the transformations should have
        the same shape (w, h, 1, 1, 2) for 2D and (w, h, p, 1, 1, 3) for 3D.

    Returns
    -------
    transform_composed : np.ndarray
        Resulting composed transformations of the same shape as input ones.
    """
    if len(transforms) == 0:
        raise ValueError("Need at least one transformation")
    if not all(t1.shape == t2.shape for t1, t2 in zip(transforms, transforms[1:])):
        raise ValueError("Transformations have different shapes")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-define the output paths for nii files
        paths = [
            os.path.join(tmpdir, f"transform_{i}.nii.gz")
            for i in range(len(transforms))
        ]
        for transform, path in zip(transforms, paths):
            write_transform(transform, path)

        # Compute the composed transformation
        zero_img = np.zeros(shape=transforms[0].shape[:2], dtype=np.float32)
        transform_composed_path = ants.apply_transforms(
            fixed=ants.from_numpy(zero_img),
            moving=ants.from_numpy(zero_img),
            transformlist=list(reversed(paths)),
            compose=os.path.join(tmpdir, "composed_"),
        )
        transform_composed = nibabel.load(transform_composed_path).get_fdata()

        # Remove all temporary files
        for path in paths:
            os.remove(path)
        os.remove(transform_composed_path)

    return transform_composed


# METRICS
def stain_symmetry_score(img, axis=1):
    """Compute the symmetric score (SSIM) of intensity-based image.

    Parameters
    ----------
    img : np.ndarray
        Image for which symmetric score is computed.

    axis : int
        Axis along which the symmetric has to be analyzed.

    Returns
    -------
    ssmi_value : float
        Structural similarity measure between two halfs of the input image.
    """
    half_1, half_2 = split_halfs(img, axis)
    ssim_value = ssim(half_1, half_2)
    return ssim_value


def atlas_symmetry_score(atlas, axis=1):
    """Compute the symmetric score (in misalignment) of label-based image.

    Parameters
    ----------
    atlas : np.ndarray
        Image for which symmetric score is computed.

    axis : int
        Axis along which the symmetric has to be analyzed.

    Returns
    -------
    alignment : float
        Alignment between two halfs of the input atlas.
    """
    half_1, half_2 = split_halfs(atlas, axis)
    alignment = 1 - get_misalignment(half_1, half_2)
    return alignment


class Remapper:
    """Utility class for remapping of labels.

    Parameters
    ----------
    volumes
        Different volumes to consider.
    """

    def __init__(self, *volumes: list[np.ndarray]) -> None:
        # initial checks
        if not volumes:
            raise ValueError("No volume provided")

        for volume in volumes:
            if not isinstance(volume, np.ndarray):
                raise TypeError("The inputs need to be numpy arrays")
            if not volume.dtype == np.uint32:
                raise TypeError("The dtype of the arrays needs to be uint32")

        self.storage: list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]] = [
            (*np.unique(volume, return_inverse=True), volume.shape)
            for volume in volumes
        ]
        unique_overall = set()
        for unique, _, _ in self.storage:
            unique_overall |= set(unique)

        self.old2new = {x: i for i, x in enumerate(sorted(unique_overall))}
        self.new2old = {v: k for k, v in self.old2new.items()}

    def __len__(self) -> int:
        """Compute the number of volumes passed."""
        return len(self.storage)

    @staticmethod
    def remap(
        shape: tuple[int, ...],
        mapping: dict[int, int],
        unique: np.ndarray,
        inv: np.ndarray,
    ) -> np.ndarray:
        """Run the remapping."""
        runique = np.array([mapping[x] for x in unique], dtype=np.uint32)

        return runique[inv].reshape(shape)

    def remap_old_to_new(self, i: int) -> np.ndarray:
        """Remap from old labels to new labels."""
        if not (0 <= i < len(self)):
            raise IndexError

        unique, inv, shape = self.storage[i]
        return self.remap(shape, self.old2new, unique, inv)

    def remap_new_to_old(self, volume: np.ndarray) -> np.ndarray:
        """Remap from new labels to old labels."""
        unique, inv = np.unique(volume, return_inverse=True)

        return self.remap(volume.shape, self.new2old, unique, inv)
