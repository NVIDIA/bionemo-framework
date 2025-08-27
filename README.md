# BioNeMo Framework

[![Click here to deploy.](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdeploynavy.svg)](https://console.brev.dev/launchable/deploy/now?launchableID=env-2pPDA4sJyTuFf3KsCv5KWRbuVlU)
[![Docs Build](https://img.shields.io/github/actions/workflow/status/NVIDIA/bionemo-framework/pages/pages-build-deployment?label=docs-build)](https://nvidia.github.io/bionemo-framework)
[![Test Status](https://github.com/NVIDIA/bionemo-framework/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/NVIDIA/bionemo-framework/actions/workflows/unit-tests.yml)
[![Latest Tag](https://img.shields.io/github/v/tag/NVIDIA/bionemo-framework?label=latest-version)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework/tags)
[![codecov](https://codecov.io/gh/NVIDIA/bionemo-framework/branch/main/graph/badge.svg?token=XqhegdZRqB)](https://codecov.io/gh/NVIDIA/bionemo-framework)

NVIDIA BioNeMo Framework is a toolkit of reference implementations ("recipes") and pip installable components for developers to build, optimize, and scale Transformer-based AI models. Learn how to use cutting-edge libraries to quickly prototype and scale performant, multi-billion parameter models for biology, chemistry, genomics, and more.

## Our Vision: From Idea to Scalable Discovery
The goal of BioNeMo Framework is to demystify the process of building large, performant AI models in healthcare. We provide a collection of clean, scalable reference implementations that show you how to leverage powerful NVIDIA libraries like:
* Transformer Engine for unlocking peak performance on Hopper and Ampere GPUs.
* Megatron-Core FSDP for unlocking context parallelism and auto-scaling your models.
* NeMo Framework for unlocking 5D parallelism in large scale models. 
Whether you're working with sequences, images, text, or multi-modal data, these recipes serve as a starting point for your own research and development.

## 🚧 Important Notice: Code Structure Update 🚧
We are transitioning from a `sub-packages` structure to a `recipes` and `models` based structure. 
* `sub-packages`: Contains NeMo Framework based implementation of models and a collection of `PYPI` packages
* `recipes`: Contains reference implementations that demonstrate best practices for scaling biological FMs using Transformer-Engine and Megatron-FSDP.
* `models`: Contains Huggingface model definitions integrated with Transformer-Engine. Primary purpose is to run tests, provide ancillary scripts like conversion scripts (convert HF <> TE), export scripts (export checkpoints) in support of TE-compatible checkpoints.

❗**Note:** Each `recipe` and `model` come with their own Dockerfile. They will NOT be available via BioNeMo Framework Docker Container. 

For the time being, models currently in `sub-packages` will also be available as a `recipe`. No ETA yet on deprecating model support from `sub-packages`. Below is the current status, keep monitoring for latest updates. 

|Model      | Status                         |
------------|--------------------------------|
|ESM2       | Available as a Recipe          |
|AMPLIFY    | Available as a Recipe          |
|Geneformer | Available as a Recipe          |
|Evo2       | Not Available as a Recipe (WIP)|

The non-model specific packages (bionemo-scdl, bionemo-noodles, bionemo-moco etc) will remain within `sub-packages`, no structure changes planned as of right now.

Use implementations in `sub-packages` if: 
* you're already familiar and are comfortable with NeMo Framework
* do not mind switching away from vanilla pytorch / HF codebase.
* Comfortable working with BioNeMo Framework container

Use `recipes` if: 
* You'd like to learn how to use Transformer-Engine and Megatron-FSDP to optimize your own implementations
* Need a light-weight install dependencies
* Don't mind building your own container and running with it. Each recipe comes with a Dockerfile, recipes are not available via BioNeMo Framework container.
* Read more [here](./recipes/README.md)

Use `models` if:
* You're curious about HF<>TE model definitions
* You're interested in conversion utilities, export functionalities to convert a HF checkpoint to TE-compatible format.
* Read more [here](./models/README.md)


## 🚀 Quick Start
|Platform | Link |
|---------|------|
|Lepton   |      |
|Colab    |      |
|Brev     |      |

## 🧩 Modular & Pip-Installable Components
Beyond recipes, the BioNeMo Framework also provides modular components that can be integrated into any PyTorch codebase. These are designed to be lightweight, performant, and easy to use.
Install them independently as needed:
Bash
```
# Install performant data loaders for biology
# Install BioNeMo Single-Cell Dataloader
pip install bionemo-scdl 

# Install a thread-safe fasta indexer
pip install bionemo-noodles
```

## Documentation Resources

- **Official Documentation:** For user guides, API references, and troubleshooting, visit our [official documentation](https://docs.nvidia.com/bionemo-framework/latest/).
- **In-Progress Documentation:** To explore the latest features and developments, check the documentation reflecting the current state of the `main` branch [here](https://nvidia.github.io/bionemo-framework/). Note that this may include references to features or APIs that are not yet finalized.
