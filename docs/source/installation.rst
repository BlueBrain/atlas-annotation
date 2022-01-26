Installation
============

Python Version and Environment
------------------------------
The currently python versions supported by ``atlannot`` package
are ``3.7``, ``3.8``, and ``3.9``.

If you are part of the Blue Brain Project and are working on the BB5 you can
find the correct python version in the archive modules:
- ``python3.7``: ``archive/2020-02`` - ``archive/2020-12``
- ``python3.8``: ``archive/2021-01`` - ``archive/2021-12``
- ``python3.9``: ``archive/2022-01`` - ``unstable``.
Here's an example of a set of commands that will set up your environment on the BB5:

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
