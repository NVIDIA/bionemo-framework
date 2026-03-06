# Getting Started

## Repository structure

### High level overview

This repository is structured as a meta-package that collects together many python packages. We designed in this way
because this is how we expect our users to use bionemo, as a package that they themselves import and use in their
own projects. By structuring code like this ourselves we ensure that bionemo developers follow similar patterns to our
end users.

Each model is stored in its own `sub-packages`. There are useful utility packages, for example:

- `sub-packages/bionemo-scdl`: Single Cell Dataloader (SCDL) provides a dataset implementation that can be used by downstream
  single-cell models in the bionemo package.

Some of the packages represent common functions and abstract base classes that expose APIs:

- `sub-packages/bionemo-core`: mostly just high level APIs

Documentation source is stored in `docs/`

The script for building a local docker container is `./launch.sh` which has some useful commands including:

- `./launch.sh build` to build the container
- `./launch.sh run` to get into a running container with reasonable settings for data/code mounts etc.

### More detailed structure notes

```
$ tree -C -I "*.pyc" -I "test_data" -I "test_experiment" -I "test_finettune_experiment" -I __pycache__ -I "*.egg-info" -I lightning_logs -I results -I data -I MNIST* -I 3rdparty
.
├── CODE-REVIEW.md -> docs/CODE-REVIEW.md
├── CODEOWNERS
├── CONTRIBUTING.md -> docs/CONTRIBUTING.md
├── Dockerfile
├── LICENSE
│   ├── license.txt
│   └── third_party.txt
├── README.md
├── VERSION
├── ci
│   └── scripts
│       ├── nightly_test.sh
│       ├── pr_test.sh
│       └── static_checks.sh
├── docs
│   ├── CODE-REVIEW.md
│   ├── CONTRIBUTING.md
│   ├── Dockerfile
│   ├── README.md
│   ├── docs
│   │   ├── assets
│   │   │   ├── css
│   │   │   │   ├── color-schemes.css
│   │   │   │   ├── custom-material.css
│   │   │   │   └── fonts.css
│   │   │   └── images
│   │   │       ├── favicon.png
│   │   │       ├── logo-icon-black.svg
│   │   │       └── logo-white.svg
│   │   ├── developer-guide
│   │   │   ├── CODE-REVIEW.md
│   │   │   ├── CONTRIBUTING.md
│   │   │   └── jupyter-notebooks.ipynb
│   │   ├── index.md
│   │   └── user-guide
│   │       └── index.md
│   ├── mkdocs.yml
│   ├── requirements.txt
│   └── scripts
│       └── gen_ref_pages.py
├── launch.sh
├── license_header
├── pyproject.toml
├── requirements-cve.txt
├── requirements-dev.txt
├── requirements-test.txt
# 🟢 All work goes into `sub-packages`
#  Sub-packages represent individually installable subsets of the bionemo codebase. We recommend that you
#  create new sub-packages to track your experiments and save any updated models or utilities that you need.
├── sub-packages
│   ├── bionemo-core  # 🟢 bionemo-core is a top level sub-package that does not depend on others
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   ├── setup.py
│   │   ├── src  # 🟢 All sub-packages have a `src` and a `test` sub-directory.
│   │   │   └── bionemo
│   │   │       └── core
│   │   │           ├── __init__.py
│   │   │           ├── api.py
│   │   │           ├── model
│   │   │           │   ├── __init__.py
│   │   │           │   └── config.py
│   │   │           └── utils
│   │   │               ├── __init__.py
│   │   │               ├── batching_utils.py
│   │   │               ├── dtypes.py
│   │   │               └── random_utils.py
│   │   └── tests  # 🟢 Test files should be mirrored with `src` files, and have the same name other than `test_[file_name].py`
│   │       └── bionemo
│   │           ├── core
│   │           └── pytorch
│   │               └── utils
│   │                   └── test_dtypes.py
│   ├── bionemo-scdl  # 🟢
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── examples
│   │   │   └── example_notebook.ipynb
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   ├── setup.py
│   │   ├── src
│   │   │   └── bionemo
│   │   │       └── scdl
│   │   │           ├── __init__.py
│   │   │           ├── api
│   │   │           │   ├── __init__.py
│   │   │           │   └── single_cell_row_dataset.py
│   │   │           ├── index
│   │   │           │   ├── __init__.py
│   │   │           │   └── row_feature_index.py
│   │   │           ├── io
│   │   │           │   ├── __init__.py
│   │   │           │   ├── single_cell_collection.py
│   │   │           │   └── single_cell_memmap_dataset.py
│   │   │           ├── scripts
│   │   │           │   ├── __init__.py
│   │   │           │   └── convert_h5ad_to_scdl.py
│   │   │           └── util
│   │   │               ├── __init__.py
│   │   │               ├── async_worker_queue.py
│   │   │               └── torch_dataloader_utils.py
│   │   └── tests
│   │       └── bionemo
│   │           └── scdl
│   │               ├── conftest.py
│   │               ├── index
│   │               │   └── test_row_feature_index.py
│   │               ├── io
│   │               │   ├── test_single_cell_collection.py
│   │               │   └── test_single_cell_memmap_dataset.py
│   │               └── util
│   │                   ├── test_async_worker_queue.py
│   │                   └── test_torch_dataloader_utils.py
│   └── bionemo-webdatamodule
│       ├── LICENSE
│       ├── README.md
│       ├── pyproject.toml
│       ├── requirements.txt
│       ├── setup.py
│       ├── src
│       │   └── bionemo
│       │       └── webdatamodule
│       │           ├── __init__.py
│       │           ├── datamodule.py
│       │           └── utils.py
│       └── tests
│           └── bionemo
│               └── webdatamodule
│                   ├── __init__.py
│                   ├── conftest.py
│                   └── test_datamodule.py
```

## Installation

### Initializing 3rd-party dependencies as git submodules

For development, the NeMo and Megatron-LM dependencies are vendored in the bionemo-2 repository workspace as git
submodules. The pinned commits for these submodules represent the "last-known-good" versions of these packages that are
confirmed to be working with bionemo2 (and those that are tested in CI).

To initialize these sub-modules when cloning the repo, add the `--recursive` flag to the git clone command:

```bash
git clone --recursive git@github.com:NVIDIA/bionemo-framework.git
```

To download the pinned versions of these submodules within an existing git repository, run

```bash
git submodule update --init --recursive
```
