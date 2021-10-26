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
    * [Experiments](#experiments)
* [Notebooks, Widgets, and Experiments](#notebooks-widgets-and-experiments)
* [Funding & Acknowledgment](#funding--acknowledgment)

## Installation

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

### Experiments

If you want to see how `atlannot` can be used, do not hesitate to check the `experiments/` folder 
containing several scripts that can be easily launched by command line interface. 

To access the experiment scripts, one needs to first clone the repository.
```shell
git clone https://github.com/BlueBrain/atlas-annotation
cd atlas-annotation/experiments
```

To run the experiments, one needs to install additional packages for interactive use.
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
