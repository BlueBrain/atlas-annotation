# Copyright 2021, Blue Brain Project, EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The setup script."""
from setuptools import find_packages, setup

install_requires = [
    "antspyx==0.2.4",
    "atldld==0.2.2",
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
    "data": [
        "dvc[ssh]>=2",
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
