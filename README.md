DEAL - DEep AtLas
=================

Installation
------------
Because of the the limitations of the ANTsPy library, which is used internally and is installed
as part of the `warpme`/`atlas_alignment` dependency, it is necessary to use python version 3.6.

To install run
```shell script
$ pip install .
```

If you are working on the BB5 please also run the following command:
```shell script
$ pip install /gpfs/bbp.cscs.ch/home/krepl/dev/ANTsPy/old_builds/gcc_6.4_jorge/antspyx-0.2.0-cp36-cp36m-linux_x86_64.whl
```

Data
----
The data for this project is managed using the DVC tool. It is automatically
installed together with this library.

If you are working on the BB5 please run the following commands first:
```shell script
$ cd data
$ dvc remote add --local gpfs /gpfs/bbp.cscs.ch/data/project/proj101/dvc/remotes/atlas_annotation
```

All data is stored in the `data` directory. DVC is similar to git. To pull all original
data from the remote run
```shell script
$ cd data
$ dvc pull
```
Note that you need to have permissions for project 101, as the data is stored
in the corresponding GPFS space.

It is also possible to selectively pull data with
```shell script
$ cd data
$ dvc pull <filename>.dvc
```
where `<filename>` should be replaced by one of the filenames found in the `data` directory.
See the `data/README.md` file for the description of different data files.
