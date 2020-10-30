DEAL - DEep AtLas
=================

Installation
------------
Because of the the limitations of the ANTsPy library, which is used internally and is installed
as part of the `warpme`/`atlas_alignment` dependency, it is necessary to use python version 3.6.

To install run
```shell script
pip install .
```

Data
----
The data for this project is managed using the DVC tool. It is automatically
installed together with this library.

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
