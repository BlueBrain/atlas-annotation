# Atlas Annotation

Over the years the Allen Brain institute has constantly improved and updated
their brain region annotation atlases. Unfortunately the old annotation atlases
are not always aligned with the new ones. For example, the CCFv2 annotations
and the Nissl volume are not compatible with the CCFv3 annotation and the
corresponding average brain volume. This package proposes a number of methods
for deforming the Nissl volume and the CCFv2 annotations in order to re-align
them to CCFv3.

* [Installation](#installation)
    * [Installation from source](#installation-from-source)
    * [Installation for development](#installation-for-development)
* [Data](#data)
    * [Downloading data from scratch](#downloading-data-from-scratch)
    * [Pulling from the remote](#pulling-from-the-remote)
* [Examples](#examples)
* [Notebooks, Widgets, and Experiments](#notebooks-widgets-and-experiments)
* [Funding & Acknowledgment](#funding--acknowledgment)

## Installation
Note that due to some of our dependencies we're curently limited to python
version `3.7`. Please make sure you set up a virtual environment with that
version before trying to install this library.

If you are part of the Blue Brain Project and are working on the BB5 you can
find the correct python version in the archive modules between `archive/2020-02`
and `archive/2020-12` (inclusive). Here's an example of a set of commands
that will set up your environment on the BB5:
```shell
module purge
module load archive/2020-12
module load python
python -m venv venv
. ./venv/bin/activate
python --version
```

We also recommend that you make sure that `pip` is up to date and that the
packages `wheel` and `setuptools` are installed:
```shell
pip install --upgrade pip wheel setuptools
```

### Installation from source
If you want to try the latest version, you can install from source.

```shell
pip install git+https://github.com/BlueBrain/atlas-annotation
```

### Installation for development
If you want a dev install, you should install the latest version from source with all the extra requirements for running test.

```shell
git clone https://github.com/BlueBrain/atlas-annotation
cd atlas-annotation
pip install -e '.[data, dev, interactive]'
```

## Data

The data for this project is managed using the DVC tool. There are two options to
get the data:
- Download them from scratch
- Pull the pre-downloaded data from a remote machine (on the BBP intranet)

In either case, one needs to clone the repository and install the extra `data` dependencies.
```shell
git clone https://github.com/BlueBrain/atlas-annotation
cd atlas-annotation/data
pip install git+https://github.com/BlueBrain/atlas-annotation#egg=atlannot[data]
```

### Downloading data from scratch
Downloading data from scratch can be done easily using dvc command.
```shell
dvc repro
```
This step might take some time. 

In some cases you might not need all data. Then it is possible to download unprepared 
data that you need by running specific DVC stages. Refer to the
[`data/README.md`](data/README.md) file for the description of different data files.

### Pulling from the remote
This only works if you have access to `proj101` on BBP intranet. Otherwise, follow 
the previous section [Downloading data from scratch](#downloading-data-from-scratch) 
instructions.

If you are working on the BB5 please run the following commands 
first:
```shell
dvc remote add --local gpfs_proj101 \
/gpfs/bbp.cscs.ch/data/project/proj101/dvc/remotes/atlas_annotation
```

To pull all original data from the remote run
```shell
dvc pull
```

It is also possible to selectively pull data with
```shell
dvc pull <filename>.dvc
```
where `<filename>` should be replaced by one of the filenames found in the `data` directory.
See the [`data/README.md`](data/README.md) file for the description of different data files.

## Examples

You can find numerous examples of the usage of `atlannot` package in the scripts located
in the [`experiments`](experiments) directory. 
```shell
git clone https://github.com/BlueBrain/atlas-annotation#egg=atlannot
cd atlas-annotation/experiments
```

To execute the scripts in this `experiments` folder, please first follow the data
preparation instructions found in the [data](#data) section. 

Next, one needs also to install additional packages for interactive use.
```shell
pip install git+https://github.com/BlueBrain/atlas-annotation#egg=atlannot[interactive]
```

Once the cloning, the installation and the download of data is done, you can use any
script, for example:
```shell
python ants2d_atlas_fine.py
```


## Notebooks, Widgets, and Experiments

The additional functionality related to notebooks, widgets, and experiment
scripts is not activated by default. In order to use it you need to specify
an additional `interactive` option upon installing this package. This can
be done as follows:
```shell
pip install git+https://github.com/BlueBrain/atlas-annotation#egg=atlannot[interactive]
```

Furthermore, you will need JupyterLab or Jupyter Notebook installed in your virtual
environment, as well as the corresponding `ipywidgets` plugin. Follow the following
online instructions in order to do so:
- How to install JupyterLab/Jupyter Notebook: https://jupyter.org/
- How to install the `ipywidgets` plugin: https://ipywidgets.readthedocs.io/en/latest/user_install.html

## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, 
a research center of the École polytechnique fédérale de Lausanne (EPFL), 
from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2021 Blue Brain Project/EPFL
