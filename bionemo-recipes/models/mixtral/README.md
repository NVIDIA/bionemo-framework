# Mixtral Optimized with NVIDIA TransformerEngine

This folder contains source code and tests for Mixtral-style Mixture of Experts (MoE) models that inherit from the
transformers `PreTrainedModel` class and use TransformerEngine layers. The implementation replaces the standard
attention layers with TE `MultiheadAttention` and uses TE `GroupedLinear` for efficient parallel expert computation.

## Feature support

The Mixtral implementation natively supports the following TransformerEngine-provided optimizations:

| Feature                                 | Support                                                                          |
| --------------------------------------- | -------------------------------------------------------------------------------- |
| **FP8**                                 | ✅ Supported on compute capacity 9.0 and above (Hopper+)                         |
| **MXFP8**                               | ✅ Supported on compute capacity 10.0 and 10.3 (Blackwell), 12.0 support pending |
| **Sequence Packing / THD input format** | ✅ Supported                                                                     |
| **FP8 with THD input format**           | ✅ Supported where FP8 is supported                                              |
| **Import from HuggingFace checkpoints** | ✅ Supported                                                                     |
| **Export to HuggingFace checkpoints**   | ✅ Supported                                                                     |
| **KV-cache inference**                  | ✅ Supported                                                                     |

## Inference Examples

### Quick start: convert and run

> **Note:** The snippets below use bare imports (e.g., `from convert import ...`). Run them from the
> `bionemo-recipes/models/mixtral` directory, or install dependencies first with `pip install -r requirements.txt`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert import convert_mixtral_hf_to_te

# Load the original HuggingFace Mixtral model
model_hf = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.bfloat16
)

# Convert to TransformerEngine
model_te = convert_mixtral_hf_to_te(model_hf)
model_te.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("The quick brown fox", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model_te.generate(**inputs, max_new_tokens=16)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Converting Between Model Formats

This section explains how to convert between Hugging Face Transformers and Transformer Engine (TE) Mixtral model
formats. The process demonstrates bidirectional conversion: from Transformers to TE format for optimized training and
inference, and back to Hugging Face Transformers format for sharing and deployment.

### Converting from HF Transformers to TE

> **Note:** Run from the `bionemo-recipes/models/mixtral` directory, or install dependencies first with
> `pip install -r requirements.txt`.

```python
from transformers import AutoModelForCausalLM

from convert import convert_mixtral_hf_to_te

model_hf = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
model_te = convert_mixtral_hf_to_te(model_hf)
model_te.save_pretrained("/path/to/te_checkpoint")
```

### Converting from TE back to HF Transformers

> **Note:** Run from the `bionemo-recipes/models/mixtral` directory, or install dependencies first with
> `pip install -r requirements.txt`.

```python
from convert import convert_mixtral_te_to_hf
from modeling_mixtral_te import NVMixtralForCausalLM

model_te = NVMixtralForCausalLM.from_pretrained("/path/to/te_checkpoint")
model_hf = convert_mixtral_te_to_hf(model_te)
model_hf.save_pretrained("/path/to/hf_checkpoint")
```

### Validating Converted Models

The golden value tests in [test_modeling_mixtral.py](tests/test_modeling_mixtral.py) verify that the converted TE model
produces numerically equivalent outputs to the original HuggingFace model. Specifically:

- `test_golden_values_bshd` — loads both models, runs a forward pass on the same input, and asserts that logits and
  loss match within tolerance.
- `test_round_trip_conversion` — converts HF → TE → HF and verifies the round-tripped model produces identical outputs.

To run these tests locally:

```bash
./ci/scripts/recipes_local_test.py bionemo-recipes/models/mixtral/
```

## Developer Guide

### Running tests

To run tests locally, run `recipes_local_test.py` from the repository root with the model directory as an argument.

```bash
./ci/scripts/recipes_local_test.py bionemo-recipes/models/mixtral/
```

### Exporting to Hugging Face Hub

The model directory includes an `export.py` script that bundles all files needed for Hugging Face Hub distribution. To
create the export bundle, run from the model directory:

```bash
python export.py
```

Before publishing, validate the export by running the local test suite via
[recipes_local_test.py](../../ci/scripts/recipes_local_test.py).

### Development container

To use the provided devcontainer, use "Dev Containers: Reopen in Container" from the VSCode menu, and choose the
"BioNeMo Recipes Dev Container" option. To run the tests inside the container, first install the dependencies with
`pip install -r requirements.txt`, then run `pytest -v .` in the model directory.
