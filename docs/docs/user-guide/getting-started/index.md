# Getting Started

## Repository structure

### High level overview
This repository is structured as a meta-package that collects together many python packages. We designed in this way
because this is how we expect our users to use bionemo, as a package that they themselves import and use in their
own projects. By structuring code like this ourselves we ensure that bionemo developers follow similar patterns to our
end users.

Each model is stored in its own `sub-packages`. Some examples of models include:

* `sub-packages/bionemo-esm2`: ESM2 model
* `sub-packages/bionemo-geneformer`: Geneformer
* `sub-packages/bionemo-example_model`: A minimal example MNIST model that demonstrates how you can write a lightweight
    megatron model that doesn't actually support any megatron parallelism, but should run fine as long as you only use
    data parallelism to train.

There are also useful utility packages, for example:

* `sub-packages/bionemo-scdl`: Single Cell Dataloader (SCDL) provides a dataset implementation that can be used by downstream
    single-cell models in the bionemo package.
* `sub-packages/bionemo-testing`: a suite of utilities that are useful in testing, think `torch.testing` or `np.testing`.

Finally some of the packages represent common functions and abstract base classes that expose APIs that are useful for
interacting with `NeMo2`. Some examples of these include:

* `sub-packages/bionemo-core`: mostly just high level APIs
* `sub-packages/bionemo-llm`: ABCs for code that multiple large language models (eg BERT variants) share.

Documentation source is stored in `docs/`

The script for building a local docker container is `./launch.sh` which has some useful commands including:

* `./launch.sh build` to build the container
* `./launch.sh run` to get into a running container with reasonable settings for data/code mounts etc.


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
├── scripts   # 🟢 Temporary scripts that demonstrate how to run some of these programs. These will be replaced.
│   ├── artifact_paths.yaml
│   ├── download_artifacts.py
│   ├── gpt-pretrain.py
│   ├── protein
│   │   └── esm2
│           └── esm2_dataset_perplexity.py
# 🟢 All work goes into `sub-packages`
#  Sub-packages represent individually installable subsets of the bionemo codebase. We recommend that you
#  create new sub-packages to track your experiments and save any updated models or utilities that you need.
├── sub-packages
│   ├── bionemo-core  # 🟢 bionemo-core, and bionemo-llm represent top level sub-packages that do not depend on others
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
│   ├── bionemo-esm2  # 🟢 The ESM2 model sub-package. This stores models and dataloaders necessary for pretraining and some example fine-tuning.
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── _requirements-test.txt
│   │   ├── _requirements.txt
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   ├── setup.py
│   │   ├── src
│   │   │   └── bionemo
│   │   │       └── esm2
│   │   │           ├── __init__.py
│   │   │           ├── api.py
│   │   │           └── model
│   │   │               ├── __init__.py
│   │   │               ├── attention.py
│   │   │               ├── embedding.py
│   │   │               ├── lr_scheduler.py
│   │   │               └── model.py
│   │   └── tests
│   │       └── bionemo
│   │           └── esm2
│   │               ├── __init__.py
│   │               ├── conftest.py
│   │               └── model
│   │                   ├── __init__.py
│   │                   ├── test_attention.py
│   │                   ├── test_embedding.py
│   │                   ├── test_lr_scheduler.py
│   │                   └── test_model.py
│   ├── bionemo-example_model  # 🟢 a small example model that demonstrates how to write a megatron model from scratch and train on MNIST
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── _requirements.txt
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   ├── setup.py
│   │   ├── src
│   │   │   └── bionemo
│   │   │       └── example_model
│   │   │           ├── __init__.py
│   │   │           └── lightning_basic.py
│   │   └── tests
│   │       └── bionemo
│   │           └── example_model
│   │               └── test_lightning_basic.py
│   ├── bionemo-fw  # 🟢 a meta-package that pulls together all other packages. A user can install this and get all of bionemo.
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── _requirements-test.txt
│   │   ├── _requirements.txt
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   ├── setup.py
│   │   ├── src
│   │   │   └── bionemo
│   │   │       └── fw
│   │   │           └── __init__.py
│   │   └── tests
│   │       ├── __init__.py
│   │       └── bionemo
│   │           └── fw
│   │               └── test_sub_package_imports.py
│   ├── bionemo-geneformer  # 🟢 geneformer sub-module
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── _requirements-test.txt
│   │   ├── _requirements.txt
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   ├── setup.py
│   │   ├── src
│   │   │   └── bionemo
│   │   │       └── geneformer
│   │   │           ├── __init__.py
│   │   │           ├── api.py
│   │   │           ├── model
│   │   │           │   ├── __init__.py
│   │   │           │   └── finetune_token_regressor.py
│   │   │           └── tokenizer
│   │   │               ├── __init__.py
│   │   │               ├── gene_tokenizer.py
│   │   │               └── label2id_tokenizer.py
│   │   └── tests
│   │       └── bionemo
│   │           └── geneformer
│   │               ├── __init__.py
│   │               ├── test_model.py
│   │               ├── test_stop_and_go.py
│   │               └── test_transformer_specs.py
│   ├── bionemo-llm  # 🟢 shared model code for LLM style models, eg BERT variants, transformer variants, etc.
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── _requirements-test.txt
│   │   ├── _requirements.txt
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   ├── setup.py
│   │   ├── src
│   │   │   └── bionemo
│   │   │       └── llm
│   │   │           ├── __init__.py
│   │   │           ├── lightning.py
│   │   │           ├── model
│   │   │           │   ├── __init__.py
│   │   │           │   ├── biobert
│   │   │           │   │   ├── __init__.py
│   │   │           │   │   ├── lightning.py
│   │   │           │   │   ├── model.py
│   │   │           │   │   ├── testing_utils.py
│   │   │           │   │   └── transformer_specs.py
│   │   │           │   ├── config.py
│   │   │           │   ├── layers.py
│   │   │           │   └── loss.py
│   │   │           └── utils
│   │   │               ├── __init__.py
│   │   │               ├── datamodule_utils.py
│   │   │               ├── iomixin_utils.py
│   │   │               ├── logger_utils.py
│   │   │               ├── remote.py
│   │   │               └── weight_utils.py
│   │   └── tests
│   │       ├── __init__.py
│   │       └── bionemo
│   │           └── llm
│   │               ├── __init__.py
│   │               ├── model
│   │               │   ├── biobert
│   │               │   │   └── test_transformer_specs.py
│   │               │   └── test_loss.py
│   │               ├── test_lightning.py
│   │               └── utils
│   │                   ├── __init__.py
│   │                   ├── test_datamodule_utils.py
│   │                   ├── test_iomixin_utils.py
│   │                   └── test_logger_utils.py
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
│   ├── bionemo-testing
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── _requirements.txt
│   │   ├── pyproject.toml
│   │   ├── requirements.txt
│   │   ├── setup.py
│   │   ├── src
│   │   │   └── bionemo
│   │   │       └── testing
│   │   │           ├── __init__.py
│   │   │           ├── callbacks.py
│   │   │           ├── harnesses
│   │   │           │   ├── __init__.py
│   │   │           │   └── stop_and_go.py
│   │   │           ├── megatron_parallel_state_utils.py
│   │   │           ├── testing_callbacks.py
│   │   │           └── utils.py
│   │   └── tests
│   │       └── bionemo
│   │           └── testing
│   │               └── test_megatron_parallel_state_utils.py
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
