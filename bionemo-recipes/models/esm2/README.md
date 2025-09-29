# ESM-2 Optimized with NVIDIA TransformerEngine

This folder contains source code and tests for an ESM-2 model that inherits from the transformers `PreTrainedModel`
class and uses TransformerEngine layers. Users don't need to install this package directly, but can load the
model directly from HuggingFace Hub using the standard transformers API (see [Inference Examples](#inference-examples)
below).

## Feature support

The ESM-2 implementation natively supports the following TransformerEngine-provided optimizations:

| Feature                                 | Support                                                                          |
| --------------------------------------- | -------------------------------------------------------------------------------- |
| **FP8**                                 | ✅ Supported on compute capacity 9.0 and above (Hopper+)                         |
| **MXFP8**                               | ✅ Supported on compute capacity 10.0 and 10.3 (Blackwell), 12.0 support pending |
| **Sequence Packing / THD input format** | ✅ Supported                                                                     |
| **FP8 with THD input format**           | ✅ Supported where FP8 is supported                                              |
| **Import from HuggingFace checkpoints** | ✅ Supported                                                                     |
| **Export to HuggingFace checkpoints**   | 🚧 Under development                                                             |

See [BioNemo Recipes](../../recipes/README.md) for more details on how to use these features to accelerate model
training and inference.

## Links to HF checkpoints

Pre-trained ESM-2 models converted from the original Facebook weights are available on HuggingFace as part of the NVIDIA
[BioNeMo collection](https://huggingface.co/collections/nvidia/bionemo-686d3faf75aa1edde8c118d9) on the HuggingFace Hub:

**Available Models:**

- [`nvidia/esm2_t6_8M_UR50D`](https://huggingface.co/nvidia/esm2_t6_8M_UR50D) (8M parameters)
- [`nvidia/esm2_t12_35M_UR50D`](https://huggingface.co/nvidia/esm2_t12_35M_UR50D) (35M parameters)
- [`nvidia/esm2_t30_150M_UR50D`](https://huggingface.co/nvidia/esm2_t30_150M_UR50D) (150M parameters)
- [`nvidia/esm2_t33_650M_UR50D`](https://huggingface.co/nvidia/esm2_t33_650M_UR50D) (650M parameters)
- [`nvidia/esm2_t36_3B_UR50D`](https://huggingface.co/nvidia/esm2_t36_3B_UR50D) (3B parameters)
- [`nvidia/esm2_t48_15B_UR50D`](https://huggingface.co/nvidia/esm2_t48_15B_UR50D) (15B parameters)

## Runtime Requirements

We recommend using the latest [NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
for optimal performance and compatibility. See the provided Dockerfile for details.

## Inference Examples

Quick start example using HuggingFace transformers:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("nvidia/esm2_t6_8M_UR50D")
tokenizer = AutoTokenizer.from_pretrained("nvidia/esm2_t6_8M_UR50D")

gfp_P42212 = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
    "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
    "HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

inputs = tokenizer(gfp_P42212, return_tensors="pt")
output = model(**inputs)
```

## Recipe Links

Training recipes are available in the `bionemo-recipes/recipes/` directory:

- **[esm2_native_te](../../recipes/esm2_native_te/)** - Demonstrates training with a simple native PyTorch training
  loop.
- **[esm2_accelerate_te](../../recipes/esm2_accelerate_te/)** - Trains the model using HuggingFace
  [Accelerate](https://huggingface.co/docs/accelerate/index).

## Commands for converting checkpoints

### HF Transformers to TE conversion

Generate converted ESM-2 checkpoints from existing HuggingFace transformers checkpoints:

```bash
mkdir -p checkpoint_export
docker build -t esm2 .
docker run --rm -it --gpus all \
  -v $PWD/checkpoint_export/:/workspace/bionemo/checkpoint_export \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface \
  esm2 python export.py
```

### TE to HF Transformers conversion

(Coming soon)

## Developer Guide

### Running tests

To run tests locally, run `recipes_local_test.py` from the repository root with the model directory as an argument.

```bash
./ci/scripts/recipes_local_test.py bionemo-recipes/models/esm2/
```

### Development container

To use the provided devcontainer, use "Dev Containers: Reopen in Container" from the VSCode menu, and choose the
"BioNeMo Recipes Dev Container" option. To run the tests inside the container, first install the model package in
editable mode with `pip install -e .`, then run `pytest -v .` in the model directory.

### Deploying converted checkpoints to HuggingFace Hub

After running the checkpoint conversion steps listed in [Commands for converting checkpoints](#commands-for-converting-checkpoints),
you can deploy the converted checkpoints to the HuggingFace Hub by running the following command:

```bash
huggingface-cli upload nvidia/${MODEL_NAME} $PWD/checkpoint_export/${MODEL_NAME}
```

Or, upload all models at once with:

```bash
for dir in *; do huggingface-cli upload nvidia/$(basename "$dir") "$dir/"; done
```
