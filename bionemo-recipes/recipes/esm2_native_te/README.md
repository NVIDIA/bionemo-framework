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

| Model                                     | BF16 | FP8<sup>[1]</sup> | THD Input Format | FP8 with THD Input Format | MXFP8<sup>[2]</sup> | Context Parallelism |
| ----------------------------------------- | ---- | ----------------- | ---------------- | ------------------------- | ------------------- | ------------------- |
| [ESM-2](../../models/esm2/README.md)      | ✅   | ✅                | ✅               | ✅                        | ✅                  | 🚧                  |
| [AMPLIFY](../../models/amplify/README.md) | ✅   | ❌                | 🚧               | ❌                        | ❌                  | 🚧                  |

✅: Supported <br/>
🚧: Under development <br/>
❌: Not supported <br/>

\[1\]: Requires compute capacity 9.0 and above (Hopper+) <br/>
\[2\]: Requires compute capacity 10.0 and 10.3 (Blackwell), 12.0 support pending <br/>

### Distributed Training

This recipe supports distributed training using DDP, FSDP2, and Megatron-FSDP, shown in three separate training
entrypoints:

- [Distributed Data Parallel (DDP)](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), shown in `train_ddp.py`
- [Fully Sharded Data Parallel 2 (FSDP2)](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), shown in `train_fsdp2.py`
- [Megatron-FSDP (mFSDP)](hhttps://pypi.org/project/megatron-fsdp/), shown in `train_mfsdp.py`

## Commands to Launch Training

To run single-process training on one GPU, run:

```bash
python train_ddp.py  # or train_fsdp2.py / train_mfsdp.py
```

To run multi-process training locally on 2+ GPUs, run (e.g. 2 GPUs):

```bash
torchrun --nproc_per_node=2 train_fsdp2.py  # or train_mfsdp.py / train_ddp.py
```

Multi-Node training is supported with all three strategies, see [`slurm.sh`](slurm.sh) for an example SLURM script.

### FP8 Training

To run training with FP8, enable it by overriding the `fp8_config.enabled=true` configuration parameter. Additional FP8
configuration parameters, including switching to `MXFP8BlockScaling`, can be set via the hydra configuration.

```bash
python train_fsdp2.py --config-name L0_sanity fp8_config.enabled=true
```

### Sequence Packing (THD input format)

Sequence packing is handled via a padding-free collator (in `collator.py`) that provides input arguments (e.g.
`cu_seq_lens_q`) needed for padding-free attention. To enable sequence packing, set `dataset.use_sequence_packing=true`
in the hydra configuration.

```bash
python train_fsdp2.py --config-name L0_sanity dataset.use_sequence_packing=true
```

### FP8 and Sequence Packing

To combine FP8 training with sequence packing, the number of unpadded input tokens must be a multiple of 16. This can be
set via the `dataset.sequence_packing_pad_to_multiple_of=16` configuration parameter.

```bash
python train_fsdp2.py --config-name L0_sanity \
  fp8_config.enabled=true \
  dataset.use_sequence_packing=true \
  dataset.sequence_packing_pad_to_multiple_of=16
```

## Saving and Loading Checkpoints

To enable checkpoint saving, ensure that `checkpoint.ckpt_dir` is set to a writable directory. Checkpointing frequency is
controlled by the `checkpoint.save_every_n_steps` configuration parameter.

```bash
python train_fsdp2.py --config-name L0_sanity \
  checkpoint.ckpt_dir=/path/to/ckpt_dir \
  checkpoint.save_every_n_steps=100
```

To enable checkpoint loading, set `checkpoint.resume_from_checkpoint=true` to resume from the latest checkpoint.

```bash
python train_fsdp2.py --config-name L0_sanity \
  checkpoint.ckpt_dir=/path/to/ckpt_dir \
  checkpoint.resume_from_checkpoint=true
```

We also show how to export a final model at the end of training, which is suitable for uploading to the Hugging Face Hub
or for local inference as a more durable format than torch distributed checkpoints. To enable this, set
`checkpoint.save_final_model=true` in the hydra configuration. The resulting model will be saved to the `final_model`
directory within the checkpoint directory.

Checkpointing is implemented for all three strategies, see [`checkpoint.py`](checkpoint.py) for more details.

## Pre-training from a local model file

To pre-train with a custom model file, set `model_tag` to a local config directory, e.g.

```bash
python train_ddp.py model_tag=example_8m_nvesm_checkpoint
```

## Running Inference with the Trained Model

Models can be loaded from the final checkpoint directory using the `AutoModel.from_pretrained` method. For example:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("path/to/final_model")
tokenizer = AutoTokenizer.from_pretrained("...")

gfp_P42212 = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
    "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
    "HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

inputs = tokenizer(gfp_P42212, return_tensors="pt")
model.eval()
output = model(**inputs)
```

## Performance

🚧 Under development

## See Also

- [ESM-2 Training with Accelerate](../esm2_accelerate_te/README.md)

## Developer Guide

### Running tests

To run tests locally, run `recipes_local_test.py` from the repository root with the recipe directory as an argument.

```bash
./ci/scripts/recipes_local_test.py bionemo-recipes/recipes/esm2_native_te/
```

Tests should be kept relatively fast, using the smallest model and number of training steps required to validate the
feature. Hardware requirements beyond those used in CI (e.g., a single L4) should be annotated with
pytest.mark.requires, e.g. `requires_fp8` and `requires_multi_gpu`.

### Development container

To use the provided devcontainer, use "Dev Containers: Reopen in Container" from the VSCode menu, and choose the
"BioNeMo Recipes Dev Container" option. To run the tests inside the container, run `pytest -v .` in the recipe
directory.

### Hydra Tips

[Hydra](https://hydra.cc/) is a powerful configuration management library for Python. This recipe uses Hydra to manage
training configurations, allowing for easy modification of training hyper-parameters and model settings.

Configuration parameters can be overridden from the command line, e.g.
`python train_fsdp2.py --config-name L0_sanity fp8_config.enabled=true`.
