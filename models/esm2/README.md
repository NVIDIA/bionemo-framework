# ESM-2 Implemented with TE layers

Running tests:

```bash
docker build -t esm2 .
docker run --rm -it --gpus all esm2 pytest tests/
```

Generating converted ESM-2 checkpoints:

```bash
docker run --rm -it --gpus all \
  -v /path/to/checkpoint_export/:/workspace/bionemo/checkpoint_export \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface \
  esm2 python export.py
```

## Model Conversion Process

This section explains how to convert between Hugging Face and Transformer Engine (TE) ESM2 model formats. The process demonstrates bidirectional conversion: from Hugging Face to TE format for optimized inference, and back to Hugging Face format for sharing and deployment. The workflow involves several key steps:

### Step 1: Load Original Hugging Face Model

First, load the original ESM2 model from Hugging Face:

```python
from transformers import AutoModelForMaskedLM

model_hf_original = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
```

This loads the pre-trained ESM2 model that will serve as our reference for comparison.

### Step 2: Export to Transformer Engine Format

Convert the Hugging Face model to Transformer Engine format using the high-level export API:

```python
from esm.export import export_hf_checkpoint

te_checkpoint_path = Path("te_checkpoint")
export_hf_checkpoint("esm2_t6_8M_UR50D", te_checkpoint_path)
```

This creates a Transformer Engine checkpoint that can be used for optimized inference.

### Step 3: Export Back to Hugging Face Format

Convert the Transformer Engine checkpoint back to Hugging Face format:

```python
from esm.export import export_te_checkpoint

hf_export_path = Path("hf_export")
exported_model_path = te_checkpoint_path / "esm2_t6_8M_UR50D"
export_te_checkpoint(str(exported_model_path), str(hf_export_path))
```

This step creates a new Hugging Face model that should be functionally equivalent to the original.

### Step 4: Load and Test the Exported Model

Load the exported model and perform validation:

```python
model_hf_exported = AutoModelForMaskedLM.from_pretrained(str(hf_export_path))
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
```

### Step 5: Validate Model Equivalence

Test the exported model against the original using masked language modeling:

```python
# Prepare test sequence
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
inputs = tokenizer(sequence, return_tensors="pt")

# Create masked inputs (15% masking)
input_ids = inputs["input_ids"].clone()
labels = inputs["input_ids"].clone()
mask_token_id = tokenizer.mask_token_id

for i in range(input_ids.shape[1]):
    if torch.rand(1).item() < 0.15:
        input_ids[0, i] = mask_token_id

inputs["input_ids"] = input_ids
inputs["labels"] = labels

# Compare outputs
with torch.no_grad():
    outputs_original = model_hf_original(**inputs)
    outputs_exported = model_hf_exported(**inputs)

# Check differences
logits_diff = torch.abs(outputs_original.logits - outputs_exported.logits).max()
print(f"Max logits difference: {logits_diff:.2e}")

if outputs_original.loss is not None and outputs_exported.loss is not None:
    loss_diff = abs(outputs_original.loss - outputs_exported.loss)
    print(f"Loss difference: {loss_diff:.2e}")
```

## Local development with vscode

To get vscode to run these tests, you can to add the following to your `.vscode/settings.json`:

```json
{
    "python.testing.pytestArgs": [
        "models/esm2/tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```

Additionally, run the following command to install the dependencies:

```bash
cd models/esm2
PIP_CONSTRAINT= pip install -e .[convert,test]
```
