Notebooks, Widgets, and Experiments
===================================

The additional functionality related to notebooks, widgets, and experiment
scripts is not activated by default. In order to use it you need to specify
an additional ``interactive`` option upon installing this package. This can
be done as follows:

.. code-block:: shell

    pip install git+https://github.com/BlueBrain/atlas-annotation#egg=atlannot[interactive]

Furthermore, you will need JupyterLab or Jupyter Notebook installed in your
virtual environment, as well as the corresponding ``ipywidgets`` plugin. Follow
the following online instructions in order to do so:

* How to install JupyterLab/Jupyter Notebook: https://jupyter.org
* How to install the ``ipywidgets`` plugin:
  https://ipywidgets.readthedocs.io/en/latest/user_install.html
