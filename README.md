# README

## Introduction

NVIDIA BioNeMo is a framework for training and deploying large biomolecular language models at supercomputing scale for the discovery and development of theraputics. The large language model (LLM) framework currently has models for small molecules (SMILES) and protein sequences. More information about the models and their training is available in the [model guides](./docs/bionemo/models).

BioNeMo relies on [NeMo](https://github.com/NVIDIA/NeMo). NeMo provides a robust environment for developing, training, and deploying deep learning models, including Megatron models. NeMo provides enhancements to PyTorch Lighting such as hyperparameter configuarbility with YAML files and checkpoint management. It also enables the development and training of large transformer models using NVIDIA's [Megatron](https://github.com/NVIDIA/Megatron-LM) framework, which makes multi-GPU, multi-node training with data parallelism, model parallelism, and mixed precision easily configurable. The [NeMo User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/) contains more information about all of these features. It is highly suggested to review at least the [NeMo Fundamentals](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) tutorial to understand how to configure jobs.

## Repo Organization

The BioNeMo repo is organized by biological entity (molecule, protein) and by deep learning model, e.g. `molecule/megamolbart` or `protein/esm1nv`. Shared code can be found within `utils` files or directories. These are the most important directories and their purpose:

- `bionemo`: The library location, including functionality for data processing, data loading, models, and associated functions.
- `examples`: The files required to run a training or inferences: Python training scripts, YAML configuration files, shell scripts, and Jupyter notebooks.
- `models`: The location for tokenizers and trained models that are downloaded.
- `setup`: Dockerfile and associated setup scripts.

## Code Development

`BIONEMO_PATH` is an environment variable used to select the path to the BioNeMo library in use. By default, this is the library installation path (`/opt/nvidia/bionemo`). For development, code should be mounted inside the container at `/workspace/bionemo`.

The easiest way to develop or test the training of in-development code is to mount a local copy of the code inside the docker container. The `launch.sh` script mounts the current working directory inside the container by default. This behavior can be customized by editing `PROJECT_PATH`, which is the local copy of the code, and `BIONEMO_PATH` in the `.env` file.

It may also be necessary to recompile the Megatron helpers, which can be done by running the script `setup/recompile_megatron_helper.sh`. This recompilation should also be done immediately before training starts on clusters as a best practice.

The auto-formatter and linter are setup via precommit hooks. To use them,
```
pip install pre-commit
pre-commit install
```

### Build and Start Container from Source

The `launch.sh` script for configuration of the required `.env` file) can be used to build and push containers to a registry and start a container for interactive work. See the instructions inside the script for more information.

Once the `.env` script is created, a container can be pulled by running `bash launch.sh pull`. The setup for BioNeMo training and inference is quite involved and requires the use of a Docker container. Due to this complexity, it is highly suggested to pull a prebuilt container. However, if a container must be built, it can be done by running `bash launch.sh build -p`. Executing this command will require an [NGC API key](https://docs.nvidia.com/ngc/ngc-overview/index.html#generating-api-key) to be configured. If pushing to a registry is desired, `bash launch.sh push` will complete this task.

Once a container has been built, it can be started in interactive mode with `bash launch.sh dev`.

### Quick Links
For more details on setting up the repo and launching example jobs, ,checkout the [Quickstart Guide](./docs/bionemo/quickstart-fw.md)
For example data-processing, pretraining and inference setup, checkout the [Tutorials](./docs/bionemo/tutorials-fw.md)
