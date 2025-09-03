# `BioNeMo-Vision`: Training a `VisionTransformer` (ViT) with `Megatron-FSDP` and `TransformerEngine`

_Adapted ViT model code from huggingface/pytorch-image-models (TImM) written by Ross Wightman (@rwightman / Copyright 2020), which you can check out here: https://github.com/huggingface/pytorch-image-models_

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

[`train.py`](train.py) is the transparent entrypoint to this script that explains how to modify your own training loop for `Megatron-FSDP` ([PyPI: `megatron-fsdp`](https://pypi.org/project/megatron-fsdp/) / [Source: Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src)) to fully-shard your model across all devices. After executing `train.py` for the first time, the de-compressed ImageNet dataset will be available in `data/super-tiny-imagenet-5/...` (sourced from [`super-tiny-imagenet-5.tar.gz`](./data/super-tiny-imagenet-5.tar.gz)) for experimentation and review.

The TIMM-derived model code for the ViT can be found in [`vit.py`](vit.py), and data utilities for ImageNet can be found in [`imagenet_*.py`](imagenet_dataset.py).

Various configuration options common in computer vision modeling can be found in [config](./config/).

#### Checkpoint Conversion

To convert DCP checkpoints to non-distributed Torch checkpoints, and vice-versa, you can run the following command from `torch`:

```
python -m torch.distributed.checkpoint.format_utils --help
usage: format_utils.py [-h] {torch_to_dcp,dcp_to_torch} src dst

positional arguments:
  {torch_to_dcp,dcp_to_torch}
                        Conversion mode
  src                   Path to the source model
  dst                   Path to the destination model

options:
  -h, --help            show this help message and exit
```

For example:

```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch step_75_loss_1.725 torch_ckpt_test.pt
```

or:

```
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp

# Convert DCP model checkpoint to torch.save format.
dcp_to_torch_save(CHECKPOINT_DIR, TORCH_SAVE_CHECKPOINT_PATH)

# Convert torch.save model checkpoint back to DCP format.
torch_save_to_dcp(TORCH_SAVE_CHECKPOINT_PATH, f"{CHECKPOINT_DIR}_new")
```

_Note that `torch.save`-converted Megatron-FSDP distributed checkpoints (DCP) cannot be loaded directly into `MegatronFSDP` module classes, because Megatron-FSDP expects a deterministic unevenly sharded checkpoint when loading using DCP. To load a non-distributed checkpoint for training with Megatron-FSDP, simply load the checkpoint into the unsharded model before calling `fully_shard`!_

```python
# Initialize model.
model = build_vit_model(cfg, device_mesh)

# Load model checkpoint. Remove the "module." prefix from the keys from Megatron-FSDP,
# which is the main discrepancy between Megatron-FSDP and normal checkpoints.
# Must load with weights_only=False if you have an optimizer state in your checkpoint.
# NOTE(@cspades): `from checkpoint import load_torch_checkpoint`
# -> load_torch_checkpoint(megatron_fsdp=True)
model_checkpoint = {
    (k.strip("module.") if megatron_fsdp else k): v
    for k, v in torch.load(checkpoint_path, weights_only=False)["model"].items()
}
# Load with strict=False because the checkpoint may have TE-specific keys that are not
# necessary for inference.
model.load_state_dict(model_checkpoint, strict=False)

# Fully-shard.
model = fully_shard_model(...)
```

TODO(@cspades): For converting DCP directly to HuggingFace SafeTensors checkpoints, you can look into: https://pytorch.org/blog/huggingface-safetensors-support-in-pytorch-distributed-checkpointing/
