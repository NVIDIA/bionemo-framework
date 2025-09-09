# Geneformer Implemented with TE layers

Running tests:

```bash
docker build -t geneformer .
docker run --rm -it --gpus all geneformer pytest tests/
```

Generating converted Geneformer checkpoints:

```bash
docker run --rm -it --gpus all \
  -v /path/to/checkpoint_export/:/workspace/bionemo/geneformer_export \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface \
  geneformer python export.py hf-to-te --model Geneformer-V2-104M
```

## Model Conversion Process

This section explains how to convert between Hugging Face and Transformer Engine (TE) Geneformer model formats. The process demonstrates bidirectional conversion: from Hugging Face to TE format for optimized inference, and back to Hugging Face format for sharing and deployment. The workflow involves several key steps:

### Step 1: Load Original Hugging Face Model

First, load the original Geneformer model from Hugging Face:

```python
from transformers import AutoModelForMaskedLM

model_hf_original = AutoModelForMaskedLM.from_pretrained(
    "ctheodoris/Geneformer", subfolder="Geneformer-V2-104M"
)
```

This loads the pre-trained Geneformer model that will serve as our reference for comparison.

### Step 2: Export to Transformer Engine Format

Convert the Hugging Face model to Transformer Engine format using the high-level export API:

```python
from geneformer.export import export_hf_checkpoint
from pathlib import Path

te_checkpoint_path = Path("te_checkpoint")
export_hf_checkpoint("Geneformer-V2-104M", te_checkpoint_path / "Geneformer-V2-104M")
```

This creates a Transformer Engine checkpoint that can be used for optimized inference.

### Step 3: Export Back to Hugging Face Format

Convert the Transformer Engine checkpoint back to Hugging Face format:

```python
from geneformer.export import export_te_checkpoint
from pathlib import Path

hf_export_path = Path("hf_export")
exported_model_path = te_checkpoint_path / "Geneformer-V2-104M"
export_te_checkpoint(str(exported_model_path), str(hf_export_path))
```

This step creates a new Hugging Face model that should be functionally equivalent to the original.

## Local development with vscode

To get vscode to run these tests, you can to add the following to your `.vscode/settings.json`:

```json
{
    "python.testing.pytestArgs": [
        "models/geneformer/tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```

Additionally, run the following command to install the dependencies:

```bash
cd models/geneformer
PIP_CONSTRAINT= pip install -e .[test]
```
