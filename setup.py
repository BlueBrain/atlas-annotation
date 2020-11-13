"""The setup script."""
from setuptools import find_packages, setup

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
    install_requires=[
        "dvc[ssh]",
        "matplotlib",
        "numpy",
        "pynrrd",
        "warpme[tf] @ git+https://bbpcode.epfl.ch/code/a/ml/atlas_alignment",
    ],
    extras_require={
        "dev": [
            "bandit",
            "black",
            "flake8",
            "isort",
            "pydocstyle",
            "pytest",
            "pytest-cov",
            "tox",
        ]
    },
)
