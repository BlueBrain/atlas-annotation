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

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "antspyx==0.2.4",
    "atldld==0.2.2",
    "matplotlib",
    "numpy",
    "pynrrd",
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
    "docs": [
        "sphinx>=1.3",
        "sphinx-bluebrain-theme",
    ],
    "interactive": [
        "ipython",
        "ipywidgets",
        "nibabel",
        "scipy",
        "toml",
        "tqdm",
    ],
}

setup(
    name="atlannot",
    author="Blue Brain Project, EPFL",
    license="Apache-2.0",
    use_scm_version={
        "write_to": "src/atlannot/version.py",
        "write_to_template": '"""The package version."""\n__version__ = "{version}"\n',
        "local_scheme": "no-local-version",
    },
    description="Align and improve brain annotation atlases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BlueBrain/atlas-annotation",
    project_urls={
        "Source": "https://github.com/BlueBrain/atlas-annotation",
        "Tracker": "https://github.com/BlueBrain/atlas-annotation/issues",
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires="~=3.7.0",
    install_requires=install_requires,
    extras_require=extras_require,
)
