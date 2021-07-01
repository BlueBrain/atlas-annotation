"""The setup script."""
from setuptools import find_packages, setup

install_requires = [
    "antspyx==0.2.4",
    "atlalign==0.5.1",
    "dvc[ssh]>=2",
    "matplotlib",
    "numpy",
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
    name="atlannot",
    use_scm_version={
        "write_to": "src/atlannot/version.py",
        "write_to_template": '"""The package version."""\n__version__ = "{version}"\n',
        "local_scheme": "no-local-version",
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires="~=3.7.0",
    install_requires=install_requires,
    extras_require=extras_require,
)
