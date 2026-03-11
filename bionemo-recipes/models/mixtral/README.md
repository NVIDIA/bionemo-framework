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

## Running with Low Precision (FP8/FP4)

The TE-optimized Mixtral model supports per-layer quantization via two mechanisms: a **config-level**
`layer_precision` list that declares which layers use which precision, and **constructor-level** recipe
objects (`fp8_recipe`, `fp4_recipe`) that control the quantization behaviour.

### Configuration: `layer_precision`

`NVMixtralConfig.layer_precision` is a list of length `num_hidden_layers` where each element is `"fp8"`,
`"fp4"`, or `None` (BF16 fallback). When set, it controls the `te.autocast` context used for each
transformer layer during both initialization and forward pass.

```python
from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM

# All layers in FP8
config = NVMixtralConfig(
    layer_precision=["fp8"] * 32,
    num_hidden_layers=32,
)
```

If you pass an `fp8_recipe` to the model constructor **without** setting `layer_precision`, it
defaults to `["fp8"] * num_hidden_layers` (all layers FP8). You can also mix precisions, for example
running most layers in FP8 but keeping the first and last layers in BF16:

```python
layer_precision = [None] + ["fp8"] * 30 + [None]
config = NVMixtralConfig(
    layer_precision=layer_precision,
    num_hidden_layers=32,
)
```

### Constructor arguments: `fp8_recipe` and `fp4_recipe`

The model classes (`NVMixtralModel`, `NVMixtralForCausalLM`) accept `fp8_recipe` and `fp4_recipe`
keyword arguments. These are `transformer_engine.common.recipe.Recipe` objects that configure the
quantization algorithm (e.g., delayed scaling, block scaling, MXFP8).

```python
import transformer_engine.common.recipe as te_recipe

from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM

fp8_recipe = te_recipe.DelayedScaling()

config = NVMixtralConfig(
    layer_precision=["fp8"] * 32,
    num_hidden_layers=32,
)
model = NVMixtralForCausalLM(config, fp8_recipe=fp8_recipe)
```

For FP4 (NVFP4) quantization, pass an `fp4_recipe` instead and set the corresponding layers to
`"fp4"` in `layer_precision`:

```python
fp4_recipe = te_recipe.NVFP4BlockScaling()

config = NVMixtralConfig(
    layer_precision=["fp4"] * 32,
    num_hidden_layers=32,
)
model = NVMixtralForCausalLM(config, fp4_recipe=fp4_recipe)
```

You can also mix FP8 and FP4 layers by providing both recipes and a mixed `layer_precision` list.

### Quantized model initialization: `use_quantized_model_init`

When `use_quantized_model_init=True` is set in the config, layers are created inside a
`te.quantized_model_init` context. This tells TransformerEngine to initialize weights directly in
the target quantized format, avoiding a separate quantization step after initialization.

```python
config = NVMixtralConfig(
    layer_precision=["fp4"] * 32,
    num_hidden_layers=32,
    use_quantized_model_init=True,
)
model = NVMixtralForCausalLM(config, fp4_recipe=te_recipe.NVFP4BlockScaling())
```

### Notes

- The `lm_head` always runs in higher precision (`te.autocast(enabled=False)`) regardless of
  `layer_precision`, to avoid numerical instability in the output logits.
- The MoE router gate (`model.layers.*.mlp.gate`) always runs in BF16 regardless of
  `layer_precision`, to maintain stable routing decisions.
- FP8 requires compute capability 9.0+ (Hopper). MXFP8 requires compute capability 10.0+
  (Blackwell).
- If an `fp8_recipe` is provided without `layer_precision`, all layers default to FP8. Providing
  both `fp8_recipe` and `fp4_recipe` without `layer_precision` raises a `RuntimeError`.
- An FP4 layer **requires** an `fp4_recipe`; omitting it raises a `RuntimeError`.

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
