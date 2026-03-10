# Getting Started: Extracting Layer Activations from CodonFM for Sparse Autoencoders

This guide walks through running codon sequences through a CodonFM Encodon model and extracting hidden-state activations from a late layer, suitable for training sparse autoencoders (SAEs).

---

## 1. Environment Setup

Use the provided Docker container (recommended) or install dependencies manually.

```bash
# Option A: Docker (recommended)
git clone https://github.com/NVIDIA-Digital-Bio/CodonFM
cd codon-fm
bash run_dev.sh

# Option B: Manual install (ensure CUDA, PyTorch, xformers are already set up)
pip install -e .
```

Key dependencies: `torch`, `xformers`, `lightning`, `safetensors`, `transformers`, `peft`, `fiddle`, `einops`.

---

## 2. Obtain a Checkpoint

Download a pretrained Encodon checkpoint from HuggingFace. Available public checkpoints:

| Model | Link |
|---|---|
| Encodon 80M | [nvidia/NV-CodonFM-Encodon-80M-v1](https://huggingface.co/nvidia/NV-CodonFM-Encodon-80M-v1) |
| Encodon 600M | [nvidia/NV-CodonFM-Encodon-600M-v1](https://huggingface.co/nvidia/NV-CodonFM-Encodon-600M-v1) |
| Encodon 1B | [nvidia/NV-CodonFM-Encodon-1B-v1](https://huggingface.co/nvidia/NV-CodonFM-Encodon-1B-v1) |
| Encodon 1B-Cdwt | [nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1](https://huggingface.co/nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1) |

For safetensors checkpoints, make sure the directory contains both the `.safetensors` weights file and a `config.json`.

For `.ckpt` checkpoints (PyTorch Lightning format), a single file is sufficient.

```bash
# Example: download 1B checkpoint
# (follow HuggingFace instructions for the specific model)
CKPT_PATH="/path/to/your/checkpoint.ckpt"
```

---

## 3. Understanding the Input/Output Pipeline

### Tokenization

CodonFM operates on **codons** (3-nucleotide chunks), not individual nucleotides or amino acids.

- Vocabulary: 69 tokens (5 special + 64 codons)
- Special tokens: `<CLS>` (0), `<SEP>` (1), `<UNK>` (2), `<PAD>` (3), `<MASK>` (4)
- Codon tokens: IDs 5-68, one for each of the 64 DNA codons (AAA, AAC, ..., TTT)

A raw DNA sequence like `ATGAAAGCCTTTGAC` is chunked into codons `ATG AAA GCC TTT GAC` and then tokenized to `[<CLS>, ATG, AAA, GCC, TTT, GAC, <SEP>, <PAD>, ...]`.

### Model Output

When `return_hidden_states=True`, the model returns an `EnCodonOutput` with:

```python
EnCodonOutput(
    logits,              # (batch, seq_len, 69) — MLM prediction logits
    last_hidden_state,   # (batch, seq_len, hidden_size) — final layer output
    all_hidden_states,   # list of (batch, seq_len, hidden_size) — one per layer
)
```

`all_hidden_states[i]` is the output of layer `i` (0-indexed). For a 24-layer model (5B), `all_hidden_states[23]` is the last layer, `all_hidden_states[20]` would be layer 21, etc.

---

## 4. Minimal Example: Extract Activations from a Single Sequence

```python
import torch
import numpy as np
from src.inference.encodon import EncodonInference
from src.tokenizer import Tokenizer
from src.data.preprocess.codon_sequence import process_item

# --- Config ---
CKPT_PATH = "/path/to/checkpoint.ckpt"  # or .safetensors
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LENGTH = 2048

# --- Load Model ---
inference = EncodonInference(model_path=CKPT_PATH, task_type="embedding_prediction")
inference.configure_model()
inference.model.to(DEVICE)
inference.model.eval()

tokenizer = inference.tokenizer

# --- Prepare a single sequence ---
# Must be a coding DNA sequence (length divisible by 3)
dna_seq = "ATGAAAGCCTTTGACGATCGTAAATAA"

# Tokenize: adds <CLS> at start, <SEP> at end, pads to CONTEXT_LENGTH
item = process_item(dna_seq, context_length=CONTEXT_LENGTH, tokenizer=tokenizer)

# Build batch (add batch dimension)
batch = {
    "input_ids": torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0).to(DEVICE),
    "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long).unsqueeze(0).to(DEVICE),
}

# --- Forward pass with hidden states ---
with torch.no_grad():
    output = inference.model(batch, return_hidden_states=True)

# output.all_hidden_states is a list of length num_layers
# Each element: (1, seq_len, hidden_size)
num_layers = len(output.all_hidden_states)
print(f"Model has {num_layers} layers")
print(f"Hidden size: {output.all_hidden_states[0].shape[-1]}")

# Extract activations from a late layer (e.g., second-to-last)
layer_idx = num_layers - 2  # penultimate layer
activations = output.all_hidden_states[layer_idx]  # (1, seq_len, hidden_size)

# Get only non-padding positions
seq_len = item["attention_mask"].sum()  # actual sequence length including CLS/SEP
activations = activations[0, :seq_len, :]  # (seq_len, hidden_size)
print(f"Activations shape: {activations.shape}")
```

---

## 5. Building an Activation Dataset for Sparse Autoencoders

For SAE training you need a large dataset of activation vectors. The approach:

1. Collect many codon sequences (e.g., from NCBI CDS data or your own).
2. Run each through the model with `return_hidden_states=True`.
3. Extract activations from your target layer at every non-special token position.
4. Save them to disk.

### 5a. Prepare Your Input Data

Create a CSV file with columns `id`, `ref_seq`, and `value` (value can be a dummy):

```csv
id,ref_seq,value
seq_001,ATGAAAGCCTTTGACGATCGTAAATAA,0
seq_002,ATGGCAGCTATCGACAAGCTGAACTGA,0
seq_003,ATGCCCAAGTTCACCGATATCTTTGACTGA,0
```

Or, to skip the CSV and work directly with a list of sequences, use the standalone tokenizer approach shown in Section 4.

### 5b. Batch Processing Script

```python
"""
extract_activations.py

Extract hidden-state activations from a CodonFM Encodon model for SAE training.
Saves activations as .npy files suitable for sparse autoencoder training.
"""
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.inference.encodon import EncodonInference
from src.tokenizer import Tokenizer
from src.data.preprocess.codon_sequence import process_item


def load_sequences(filepath):
    """Load sequences from a text file (one DNA sequence per line) or a CSV."""
    seqs = []
    if filepath.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(filepath)
        seq_col = "ref_seq" if "ref_seq" in df.columns else df.columns[1]
        seqs = df[seq_col].tolist()
    else:
        with open(filepath) as f:
            seqs = [line.strip() for line in f if line.strip()]
    return seqs


def extract_activations(
    ckpt_path: str,
    sequences_file: str,
    output_dir: str,
    layer_idx: int = -2,          # which layer to extract (-1 = last, -2 = penultimate, etc.)
    context_length: int = 2048,
    batch_size: int = 8,
    include_special_tokens: bool = False,  # whether to include CLS/SEP positions
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,    # use fp16 to save memory
    max_sequences: int = None,             # limit for testing
):
    """
    Extract activations from a specific layer and save to disk.

    Output: one .npy file per batch, plus a metadata file.
    Each .npy contains an array of shape (num_tokens_in_batch, hidden_size).
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load model ---
    print(f"Loading model from {ckpt_path}...")
    inference = EncodonInference(model_path=ckpt_path, task_type="embedding_prediction")
    inference.configure_model()
    inference.model.to(device)
    inference.model.eval()

    tokenizer = inference.tokenizer

    # Determine the number of layers
    num_layers = len(inference.model.model.layers)
    # Resolve negative indexing
    target_layer = layer_idx if layer_idx >= 0 else num_layers + layer_idx
    print(f"Model has {num_layers} layers. Extracting from layer {target_layer}.")
    hidden_size = inference.model.model.config.hidden_size
    print(f"Hidden size: {hidden_size}")

    # --- Load sequences ---
    sequences = load_sequences(sequences_file)
    if max_sequences:
        sequences = sequences[:max_sequences]
    print(f"Loaded {len(sequences)} sequences.")

    # --- Process in batches ---
    all_token_counts = []
    batch_idx = 0

    for start in tqdm(range(0, len(sequences), batch_size), desc="Extracting"):
        end = min(start + batch_size, len(sequences))
        batch_seqs = sequences[start:end]

        # Tokenize each sequence
        items = [process_item(seq, context_length=context_length, tokenizer=tokenizer) for seq in batch_seqs]

        input_ids = torch.tensor(
            np.stack([it["input_ids"] for it in items]), dtype=torch.long, device=device
        )
        attention_mask = torch.tensor(
            np.stack([it["attention_mask"] for it in items]), dtype=torch.long, device=device
        )

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Forward pass
        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
            output = inference.model(batch, return_hidden_states=True)

        # Extract target layer
        layer_acts = output.all_hidden_states[target_layer]  # (B, L, H)

        # Collect non-padding (and optionally non-special) token activations
        batch_activations = []
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        for i in range(layer_acts.shape[0]):
            seq_len = attention_mask[i].sum().item()
            acts = layer_acts[i, :seq_len, :]  # (seq_len, H)

            if not include_special_tokens:
                # Remove CLS (position 0) and SEP (last non-pad position)
                acts = acts[1:-1, :]  # skip CLS and SEP

            batch_activations.append(acts.float().cpu().numpy())

        # Concatenate all tokens in this batch
        batch_acts = np.concatenate(batch_activations, axis=0)  # (total_tokens, H)
        all_token_counts.append(batch_acts.shape[0])

        # Save this batch
        np.save(os.path.join(output_dir, f"activations_batch_{batch_idx:06d}.npy"), batch_acts)
        batch_idx += 1

        # Free GPU memory
        del output, layer_acts, batch
        torch.cuda.empty_cache()

    # Save metadata
    metadata = {
        "ckpt_path": ckpt_path,
        "layer_idx": target_layer,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "context_length": context_length,
        "num_sequences": len(sequences),
        "num_batches": batch_idx,
        "token_counts_per_batch": all_token_counts,
        "total_tokens": sum(all_token_counts),
        "include_special_tokens": include_special_tokens,
    }
    np.save(os.path.join(output_dir, "metadata.npy"), metadata)
    print(f"Done. Saved {batch_idx} batches, {sum(all_token_counts)} total tokens to {output_dir}/")
    print(f"Each activation vector has dimension {hidden_size}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract CodonFM activations for SAE training")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt or .safetensors checkpoint")
    parser.add_argument("--sequences", required=True, help="Path to sequences file (.csv or .txt, one seq per line)")
    parser.add_argument("--output_dir", required=True, help="Directory to save activation .npy files")
    parser.add_argument("--layer", type=int, default=-2, help="Layer index to extract (negative = from end, default: -2 = penultimate)")
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--include_special_tokens", action="store_true", help="Include CLS/SEP token activations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_sequences", type=int, default=None, help="Limit number of sequences (for testing)")
    args = parser.parse_args()

    extract_activations(
        ckpt_path=args.ckpt,
        sequences_file=args.sequences,
        output_dir=args.output_dir,
        layer_idx=args.layer,
        context_length=args.context_length,
        batch_size=args.batch_size,
        include_special_tokens=args.include_special_tokens,
        device=args.device,
        max_sequences=args.max_sequences,
    )
```

### 5c. Running the Extraction

```bash
# Extract penultimate layer activations
python extract_activations.py \
    --ckpt /path/to/checkpoint.ckpt \
    --sequences /path/to/sequences.csv \
    --output_dir ./activations/layer_penultimate \
    --layer -2 \
    --batch_size 8

# Extract from a specific layer (e.g., layer 20 of a 24-layer model)
python extract_activations.py \
    --ckpt /path/to/checkpoint.ckpt \
    --sequences /path/to/sequences.txt \
    --output_dir ./activations/layer_20 \
    --layer 20 \
    --batch_size 4
```

### 5d. Loading Saved Activations for SAE Training

```python
import numpy as np
from pathlib import Path

activation_dir = Path("./activations/layer_penultimate")

# Load metadata
metadata = np.load(activation_dir / "metadata.npy", allow_pickle=True).item()
print(f"Total tokens: {metadata['total_tokens']}")
print(f"Hidden size:  {metadata['hidden_size']}")
print(f"Layer:        {metadata['layer_idx']} / {metadata['num_layers']}")

# Load all batches into a single array
batch_files = sorted(activation_dir.glob("activations_batch_*.npy"))
all_activations = np.concatenate([np.load(f) for f in batch_files], axis=0)
print(f"Loaded activations: {all_activations.shape}")
# -> (total_tokens, hidden_size), e.g. (500000, 2048) for Encodon 1B
```

---

## 6. Choosing Which Layer to Extract

For SAE work, you typically want a layer near the end of the network but not the very last layer (which is most specialized for the MLM objective). Rules of thumb:

| Model | Num Layers | Suggested SAE Layer | `--layer` arg |
|---|---|---|---|
| Encodon 80M | 6 | 4 (layer 5) | `--layer 4` or `--layer -2` |
| Encodon 600M | 12 | 9-10 | `--layer 10` or `--layer -2` |
| Encodon 1B | 18 | 15-16 | `--layer 16` or `--layer -2` |
| Encodon 5B | 24 | 20-22 | `--layer 22` or `--layer -2` |

The `--layer -2` shorthand (penultimate layer) is a reasonable default for any model size.

---

## 7. What the Activations Represent

Each activation vector in the output is a `hidden_size`-dimensional representation at a single codon position. For the Encodon 1B model, each vector is 2048-dimensional; for the 5B, each is 4096-dimensional.

These vectors encode contextual information about:
- The identity of the codon at that position
- Surrounding codon context (bidirectional, since Encodon is an encoder)
- Codon usage patterns and translational signals

When fed into a sparse autoencoder, interpretable features may correspond to biological concepts like codon usage bias, GC content, secondary structure signals, or functional motifs.

---

## 8. Memory Considerations

| Model | Hidden Size | Approx. GPU Memory | Suggestion |
|---|---|---|---|
| Encodon 80M | 1024 | ~1 GB | batch_size=32+ |
| Encodon 600M | 2048 | ~3 GB | batch_size=16 |
| Encodon 1B | 2048 | ~5 GB | batch_size=8 |
| Encodon 5B | 4096 | ~20 GB | batch_size=2-4 |

Use `--device cpu` if you lack GPU access (much slower). Use `torch.float16` or `torch.bfloat16` via `torch.autocast` to reduce memory (already enabled in the script above).

For very large-scale extraction, consider:
- Processing in multiple runs with `--max_sequences`
- Using the CodonFM data pipeline with PyTorch Lightning's `predict` mode for multi-GPU extraction
- Streaming activations to disk rather than accumulating in memory

---

## 9. Using the Built-in CLI for Embedding Extraction

CodonFM's runner also supports embedding extraction out of the box via the `eval` mode. This uses Lightning's distributed prediction and the built-in `PredWriter` to save results:

```bash
python -m src.runner eval \
    --model_name encodon_1b \
    --checkpoint_path /path/to/checkpoint.ckpt \
    --data_path /path/to/sequences.csv \
    --task_type embedding_prediction \
    --process_item codon_sequence \
    --dataset_name CodonBertDataset \
    --predictions_output_dir ./embeddings_output \
    --num_gpus 1
```

However, this only extracts the CLS token embedding from the final layer. For full per-token, per-layer activations (what you need for SAEs), use the custom script in Section 5b.
