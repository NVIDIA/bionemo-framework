# TransformerEngine-accelerated ESM-2 training with native PyTorch training loop

This folder demonstrates how to train TE-accelerated ESM-2 with a native PyTorch training loop, including sequence
packing and FP8 precision, using fully sharded data parallel (FSDP) for distributed training.

## How to use this recipe

This folder contains an independent, minimal training example. It does not depend on any other code in the top-level
bionemo-framework repository. You can download a zipped directory of this folder alone by clicking
[here](https://download-directory.github.io?url=https://github.com/NVIDIA/bionemo-framework/tree/main/bionemo-recipes/recipes/esm2_native_te&filename=esm2-native-te).

### How to deploy this recipe on cloud providers

🚧 Under development

## Supported Models and Training Features

| Model                                     | BF16 | FP8<sup>[1](#fp8)</sup> | THD Input Format | FP8 with THD Input Format | MXFP8<sup>[2](#mxfp8)</sup> | Context Parallelism |
| ----------------------------------------- | ---- | ----------------------- | ---------------- | ------------------------- | --------------------------- | ------------------- |
| [ESM-2](../../models/esm2/README.md)      | ✅   | ✅                      | ✅               | ✅                        | ✅                          | 🚧                  |
| [AMPLIFY](../../models/amplify/README.md) | ✅   | ❌                      | 🚧               | ❌                        | ❌                          | 🚧                  |

✅: Supported
🚧: Under development
❌: Not supported

<a name="fp8">1</a>: Requires compute capacity 9.0 and above (Hopper+)
<a name="mxfp8">2</a>: Requires compute capacity 10.0 and 10.3 (Blackwell), 12.0 support pending

### Distributed Training

This recipe supports distributed training using DDP, FSDP2, and Megatron-FSDP, shown in three separate training
entrypoints:

- [DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), shown in `train_ddp.py`
- [FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), shown in `train_fsdp2.py`
- [Megatron-FSDP](hhttps://pypi.org/project/megatron-fsdp/), shown in `train_mfsdp.py`

## Commands to Launch Training

To run single-process training on one GPU, run:

```bash
python train_ddp.py  # or train_fsdp2.py / train_mfsdp.py
```

To run multi-process training locally on 2+ GPUs, run (e.g. 2 GPUs):

```bash
torchrun --nproc_per_node=2 train_fsdp2.py  # or train_mfsdp.py / train_ddp.py
```

Multi-Node

## See Also

- [ESM-2 Training with Accelerate](../esm2_accelerate_te/README.md)
