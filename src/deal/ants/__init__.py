import os
import tempfile

import ants
import nibabel
import numpy as np


def register(fixed, moving, **ants_kwargs):
    """Perform an intensity-based registration.

    Parameters
    ----------
    fixed : np.ndarray
        The fixed reference image.
    moving : np.ndarray
        The moving image that will be registered to the fixed image.
    ants_kwargs :
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
        If the shapes of the input images don't match.
    RuntimeError
        If the resulting transform produced by ANTsPy doesn't have
        the expected form.
    """
    if fixed.shape != moving.shape:
        raise ValueError("Fixed and moving images have different shapes.")
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
        if not np.allclose(nii.affine, np.diag([-1., -1., 1, 1])):
            raise RuntimeError("Unexpected affine part.")
        nii_data = nii.get_fdata()

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
            -transforms
    Returns
    -------
    warped : np.ndarray
        The warped image.

    Raises
    ------
    RuntimeError
        Whenever the internal call of `ants.apply_transforms` fails.
    """
    # Reconstruct the transform. The `register` function asserts that the
    # affine part is diag(-1, -1, 1, 1).
    affine = np.diag([-1., -1., 1., 1.])
    nii = nibabel.Nifti1Image(
        dataobj=nii_data,
        affine=affine,
    )

    image = ants.from_numpy(image)
    with tempfile.TemporaryDirectory() as out_dir:
        # ants.apply_transforms needs a file on disk
        nii_file = os.path.join(out_dir, "out.nii.gz")
        nibabel.save(nii, nii_file)

        warped = ants.apply_transforms(
            fixed=image,  # shouldn't matter...
            moving=image,
            transformlist=[str(nii_file)],
            **ants_kwargs,
        )

        # This is only true if the transformation was successful
        if isinstance(warped, ants.ANTsImage):
            return warped.numpy()
        else:
            raise RuntimeError("Could not apply the transformation")
