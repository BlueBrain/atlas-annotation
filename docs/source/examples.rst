Examples
========
Here are some examples of the functionalities that one can find in the
``atlannot`` package.

Registration
------------
One can compute the registration between a fixed and a moving image.
Those images can be of any type (for example Atlas Annotations or simply
intensity images). The inputs can be 2D or 3D, the only constraint is that they
have to be of the same shape.

The main use-case of ``atlannot`` is the registration of brain volumes from one
coordinate framework to another. It is then needed to allow some flexibility in
terms of inputs type to accept any data such as regions annotations, intensity
images.

.. code-block:: python

    import numpy as np

    from atlannot.ants import register, transform

    fixed = np.random.rand(20, 20)   # replace by a real image
    moving = np.random.rand(20, 20)  # replace by a real image
    # Computation of the displacement field from moving image to fixed image.
    nii_data = register(fixed.astype(np.float32), moving.astype(np.float32))
    # Apply the displacement to moving image.
    warped = transform(moving.astype(np.float32), nii_data)


Image Manipulation
------------------
``atlannot`` has also a lot of utility functions to manipulate images in order
to make some pre-processing/post-processing on images.

A concrete example could be to combine a region annotation and an intensity
image together and use the final result as an input to the registration.
To merge information from both images, one could superpose regions borders of
the annotation on top of the intensity image.

.. code-block:: python

    import numpy as np
    from atlannot.utils import edge_laplacian_thin, merge

    intensity_img = np.random.rand(20, 20) # Load intensity image here

    # Create fake annotation image
    annotation_img = np.zeros((20, 20))    # Load annotation image here
    annotation_img[5:15, 5:15] = 1         # Load annotation image here

    # Compute the borders of the annotation image
    borders = edge_laplacian_thin(annotation_img)

    # Merge intensity image and annotation image
    merge_img = merge(intensity_img, borders)

See here other manipulation one can do on any kind of images:

.. code-block:: python

    import numpy as np
    from atlannot.utils import (
      add_middle_line,
      edge_laplacian_thick,
      edge_laplacian_thin,
      edge_sobel,
      image_convolution,
      split_halfs,
    )

    # Instantiate an image
    img = np.random.rand(20, 20)  # Please replace by a real image

    # Apply some convolution to the image
    kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    img1 = image_convolution(img, kernel=kernel)
    img2 = edge_laplacian_thick(img)
    img3 = edge_sobel(img)
    img4 = edge_laplacian_thick(img)

    # Add a middle line, can choose the axis, the tickness, ...
    img5 = add_middle_line(img, axis=0, thickness=2)

    # Split the image into two
    half_imgs = split_halfs(img2, axis=0)[0]


Utilities
---------
The ``atlannot`` contains other utilities:

* Atlas utilities:

  * Merge atlases to harmonize the scripts
  * Unfurl regions if the regions are structured in tree
  * Compute misalignments
  * Remapping the labels

* Notebook utilities:

  * Volume Viewer to see volume in every directions
  * Add colored legend to atlas images

Concrete examples
-----------------
You can find numerous examples of the usage of ``atlannot`` package in the
scripts located in the ``experiments`` directory.

.. code-block:: shell

    git clone https://github.com/BlueBrain/atlas-annotation#egg=atlannot
    cd atlas-annotation/experiments

To execute the scripts in this ``experiments`` folder, please first follow the
data preparation instructions found in the :ref:`Data <data>` section.

Next, one needs also to install additional packages for interactive use.

.. code-block:: shell

    pip install git+https://github.com/BlueBrain/atlas-annotation#egg=atlannot[interactive]

Once the cloning, the installation and the download of data is done, you can
use any script, for example:

.. code-block:: shell

    python ants2d_atlas_fine.py

