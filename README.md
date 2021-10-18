# Atlas Annotation


* [Installation](#installation)
    * [Installation from source](#installation-from-source)
    * [Installation for development](#installation-for-development)
* [Examples](#examples)
    * [Experiments folder](#experiments-folder)
* [Data](#data)
* [Notebooks, Widgets, and Experiments](#notebooks-widgets-and-experiments)
* [Funding & Acknowledgment](#funding--acknowledgment)

## Installation

### Installation from source
If you want to try the latest version, you can install from source.

```shell
$ pip install git+https://github.com/BlueBrain/atlas-annotation
```

### Installation for development
If you want a dev install, you should install the latest version from source with all the extra requirements for running test.

```
git clone https://github.com/BlueBrain/atlas-annotation
cd atlas-annotation
pip install -e '.[data, dev, interactive]'
```

## Examples

### Experiments folder
If you want to see how `atlannot` can be used, do not hesitate to check the `experiments/` folder 
containing several scripts that can be easily launched by command line interface. 

```shell
cd experiments/
python ants2d_atlas_fine.py
```

## Data

The data for this project is managed using the DVC tool. It is automatically
installed together with this library.

If you are working on the BB5 please run the following commands first:
```shell
$ cd data
$ dvc remote add --local gpfs /gpfs/bbp.cscs.ch/data/project/proj101/dvc/remotes/atlas_annotation
```

All data is stored in the `data` directory. DVC is similar to git. To pull all original
data from the remote run
```shell
$ cd data
$ dvc pull
```
Note that you need to have permissions for project 101, as the data is stored
in the corresponding GPFS space.

It is also possible to selectively pull data with
```shell
$ cd data
$ dvc pull <filename>.dvc
```
where `<filename>` should be replaced by one of the filenames found in the `data` directory.
See the `data/README.md` file for the description of different data files.

## Notebooks, Widgets, and Experiments

The additional functionality related to notebooks, widgets, and experiment
scripts is not activated by default. In order to use it you need to specify
an additional `interactive` option upon installing this package. This can
be done as follows:
```shell
$ pip install ".[interactive]"
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