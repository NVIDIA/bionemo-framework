# `BioNeMo-Vision`: Training a `VisionTransformer` (ViT) with `Megatron-FSDP`

_Adapted ViT model code from huggingface/pytorch-image-models (TImM), which you can check out here: https://github.com/huggingface/pytorch-image-models_

### Pre-Requisites

#### Docker Container

To build a Docker image for this recipe, run the following commands:

```
docker build -t <image_repo>:<image_tag> .
```

To launch a Docker container from the image, run the following command:

```
# Utilize plenty of shared memory (--shm-size) to support loading large batches of image data!
docker run -it --rm --gpus=all --shm-size=16G <image_repo>:<image_tag>
```

#### PIP Install

If you have a virtual environment and CUDA installed, you can install the recipe's dependencies using `pip`:

```
cd recipes/vit
# If this causes problems, you can add PIP_CONSTRAINT= before the `pip install` command to ignore potentially trivial dependency conflicts.
# We strongly recommend installing into a clean virtual environment or CUDA container, such as the image built from the Dockerfile in this recipe.
pip install -r requirements.txt
```

### Training a Vision Transformer

To train a ViT using FSDP, execute the following command in your Docker container, Python virtual environment, or directly after your `docker run` command:

```
torchrun --nproc-per-node ${NGPU} train.py --config-name vit_base_patch16_224 distributed.dp_shard=${NGPU} training.checkpoint.path=./ckpts/vit
```

which will train on a local tiny 5-class version of [ImageNet](https://image-net.org/) ([super-tiny-imagenet-5](./data/super-tiny-imagenet-5/)) and save auto-resumable [Torch DCP](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) checkpoints to the `training.checkpoint.path` directory.

[`train.py`](train.py) is the transparent entrypoint to this script that explains how to modify your own training loop for `Megatron-FSDP` ([PyPI: `megatron-fsdp`](https://pypi.org/project/megatron-fsdp/) / [Source: Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src)) to fully-shard your model across all devices.

The TIMM-derived model code for the ViT can be found in [`vit.py`](vit.py), and data utilities for ImageNet can be found in [`imagenet_*.py`](imagenet_dataset.py).

Various configuration options common in computer vision modeling can be found in [config](./config/).
