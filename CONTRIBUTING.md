# Contributing
This page includes some guidelines to enable you to contribute to the project.

## Found a bug?
If you find a bug in the source code or in using the theme, you can
open an issue on GitHub.
Even better, you can submit a pull request with a fix.

## Submission guidelines
### Submitting an issue
Before you submit an issue, please search the issue tracker, maybe an issue
for your problem already exists and the discussion might inform you of workarounds
readily available.

We want to fix all the issues as soon as possible, but before fixing a bug we
need to reproduce and confirm it. In order to reproduce bugs we will need as
much information as possible, and preferably a sample demonstrating the issue.

### Submitting a pull request (PR)
If you wish to contribute to the code base, please open a pull request by
following GitHub's guidelines.

In order to add new code you need to set up a local environment and a copy of
the repository.

At the moment only python version `3.7` is supported, please make sure the
virtual environment you are using has that python version. It's also advised
to make sure the packages `pip`, `wheel` and `setuptools` are installed and on
their latest versions via
```shell
pip install --upgrade pip wheel setuptools
````

We recommend cloning the repository locally and installing all extra
dependencies including the `dev` ones. A typical set of commands doing this is
the following (replace the URL if you are working on a fork)
```shell
git clone https://github.com/BlueBrain/atlas-annotation
cd atlas-interpolation
pip install -e '.[data, dev, docs, interactive]'
```
This will install `atlannot` in editable mode and all of its dependencies.

## Development Conventions
`atlannot` uses:
- Black for formatting code
- Flake8 for linting code
- PyDocStyle for checking docstrings
