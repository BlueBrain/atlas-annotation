Installation
============

Python Version and Environment
------------------------------
Note that due to some of our dependencies we're currently limited to ``python``
version ``3.7``. Please make sure you set up a virtual environment with that
version before trying to install this library. If you're unsure how to do that
please have a look at `conda <https://docs.conda.io>`__ or
`pyenv <https://github.com/pyenv/pyenv>`__.

If you are part of the Blue Brain Project and are working on the BB5 you can
find the correct python version in the archive modules between
``archive/2020-02`` and ``archive/2020-12`` (inclusive). Here's an example of a
set of commands that will set up your environment on the BB5:

.. code-block:: shell

    module purge
    module load archive/2020-12
    module load python
    python -m venv venv
    . ./venv/bin/activate
    python --version

We also recommend that you make sure that ``pip`` is up-to-date and that the
packages ``wheel`` and ``setuptools`` are installed:

.. code-block:: shell

    pip install --upgrade pip wheel setuptools


Install "Atlas Annotation"
--------------------------
In order to access the data and the example scripts a local clone of this
repository is required. Run these commands to get it:

.. code-block:: shell

    git clone https://github.com/BlueBrain/atlas-annotation
    cd atlas-annotation

The "Atlas Interpolation" package can now be installed directly from the clone
we just created:

.. code-block:: shell

    pip install '.[data, interactive]'
