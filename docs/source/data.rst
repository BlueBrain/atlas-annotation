.. _data:

Data
====

The data for this project is managed by the `DVC tool <https://dvc.org>`__ and
all related files are located in the ``data`` directory. The DVC tool has
already been installed together with the "Atlas Interpolation" package. Every
time you need to run a DVC command (``dvc ...``) make sure to change to the
``data`` directory first (``cd data``).

Remote Storage Access
---------------------
We have already prepared all the data, but it is located on a remote storage
that is only accessible to people within the Blue Brain Project who have
access permissions to project ``proj101``. If you're unsure you can test your
permissions with the following command:

.. code-block:: shell

    ssh bbpv1.bbp.epfl.ch \
    "ls /gpfs/bbp.cscs.ch/data/project/proj101/dvc_remotes"

Possible outcomes:

.. code-block:: shell

    # Access OK
    atlas_annotation
    atlas_interpolation

    # Access denied
    ls: cannot open directory [...]: Permission denied

Depending on whether you have access to the remote storage in the following
sections you will either pull the data from the remote (``dvc pull``) or
download the input data manually and re-run the data processing pipelines to
reproduce the output data (``dvc repro``).

If you work on the BB5 and have access to the remote storage then run the
following command to short-circuit the remote access (because the remote is
located on the BB5 itself):

.. code-block:: shell

    cd data
    dvc remote add --local gpfs_proj101 \
      /gpfs/bbp.cscs.ch/data/project/proj101/dvc_remotes/atlas_annotation
    cd ..


Get the Data
------------
The purpose of the "Atlas Annotation" package is to align brain volumes and
the corresponding atlases. This section explains how to get these data.

If you have access to the remote storage (see above) then all data can be
readily pulled from it:

.. code-block:: shell

    cd data
    dvc pull
    cd ..

In the case where you don't have access to the remote storage, the data need
to be downloaded from the original sources and the pre-processing needs to
be run. Note that the pre-processing may take a long time (around an hour).
Run the following commands to start this process:

.. code-block:: shell

    cd data
    dvc repro
    cd ..

In some cases you might not need all data. Then it is possible to download
unprepared data that you need by running specific DVC stages. Refer to the
``data/README.md`` file for the description of different data files.
