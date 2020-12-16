"""The setup script."""
import sys

from setuptools import find_packages, setup

# Note that python 3.6 is enforced in the setup(...) call
antspy_url = "https://github.com/ANTsX/ANTsPy/releases/download"
if sys.platform == "darwin":
    antspy_wheel = "antspy-0.1.6-cp36-cp36m-macosx_10_13_x86_64.whl"
    antspy = f"antspy @ {antspy_url}/v0.1.6/{antspy_wheel}#egg=antspy-0.1.6"
elif sys.platform == "linux":
    antspy_wheel = "antspy-0.1.7-cp36-cp36m-linux_x86_64.whl"
    antspy = f"antspy @ {antspy_url}/v0.1.8/{antspy_wheel}#egg=antspy-0.1.7"
else:
    raise NotImplementedError("Only Linux and MacOS are supported at the moment.")

install_requires = [
    antspy,
    "dvc[ssh]",
    "matplotlib",
    "numpy",
    "warpme[tf] @ git+https://bbpcode.epfl.ch/code/a/ml/atlas_alignment@v0.2",
]

extras_require = {
    "dev": [
        "bandit",
        "black",
        "flake8",
        "isort",
        "pydocstyle",
        "pytest",
        "pytest-cov",
        "tox",
    ],
    "interactive": [
        "ipython",
        "ipywidgets",
        "nibabel",
        "pynrrd",
        "scipy",
        "toml",
        "tqdm",
    ],
}

setup(
    name="deal",
    use_scm_version={
        "write_to": "src/deal/version.py",
        "write_to_template": '"""The package version."""\n__version__ = "{version}"\n',
        "local_scheme": "no-local-version",
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires="~=3.6.0",
    install_requires=install_requires,
    extras_require=extras_require,
)
