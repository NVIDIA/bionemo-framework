import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from src.inference.encodon import EncodonInference
from src.data.preprocess.codon_sequence import process_item

CKPT = "../../data/jwilber/checkpoints/encodon_1b"
CSV_PATH = "../../data/jwilber/codonfm/data/Primates.csv"
# CSV_PATH = "../../data/jwilber/codonfm/data/merged_seqs_for_validation.csv"
LAYER = -2  # penultimate layer
BATCH_SIZE = 8
CONTEXT_LENGTH = 2048
SHARD_SIZE = 100_000

# Build informative output name: {dataset}_{model}_{layer}
dataset_name = Path(CSV_PATH).stem  # e.g. "merged_seqs_for_validation"
model_name = Path(CKPT).name       # e.g. "encodon_1b"
OUT_DIR = f"../../data/jwilber/codonfm/activations/{dataset_name}_{model_name}_layer{LAYER}"

# Load model
inf = EncodonInference(model_path=CKPT, task_type="embedding_prediction")
inf.configure_model()
inf.model.to("cuda").eval()

num_layers = len(inf.model.model.layers)
target_layer = LAYER if LAYER >= 0 else num_layers + LAYER
hidden_dim = inf.model.model.config.hidden_size

# Load sequences, filter to context length
df = pd.read_csv(CSV_PATH)
# Auto-detect sequence column
seq_col = next(c for c in ["seq", "cds", "sequence"] if c in df.columns)
df = df[df[seq_col].str.len() // 3 <= CONTEXT_LENGTH - 2]
sequences = df[seq_col].tolist()[:100_000]
print(f"Using column: '{seq_col}'")
print(f"Loaded {len(sequences)} sequences (after filtering)")

os.makedirs(OUT_DIR, exist_ok=True)

# -- Extract activations and write parquet shards --
buffer = []
buffer_size = 0
shard_idx = 0
total_tokens = 0

def flush_shard(data, idx):
    """Write a shard as parquet in ActivationStore format."""
    table = pa.table({f"dim_{i}": data[:, i] for i in range(data.shape[1])})
    pq.write_table(table, os.path.join(OUT_DIR, f"shard_{idx:05d}.parquet"), compression="snappy")
    return idx + 1

for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
    batch_seqs = sequences[i:i + BATCH_SIZE]
    items = [process_item(s, context_length=CONTEXT_LENGTH, tokenizer=inf.tokenizer) for s in batch_seqs]

    batch = {
        "input_ids": torch.tensor(np.stack([it["input_ids"] for it in items])).cuda(),
        "attention_mask": torch.tensor(np.stack([it["attention_mask"] for it in items])).cuda(),
    }

    with torch.no_grad():
        out = inf.model(batch, return_hidden_states=True)

    layer_acts = out.all_hidden_states[LAYER]

    for j, it in enumerate(items):
        seq_len = it["attention_mask"].sum()
        acts = layer_acts[j, 1:seq_len - 1, :].float().cpu().numpy()  # (num_codons, hidden_dim)
        buffer.append(acts)
        buffer_size += acts.shape[0]

    # Flush full shards
    while buffer_size >= SHARD_SIZE:
        combined = np.concatenate(buffer, axis=0)
        shard_data = combined[:SHARD_SIZE]
        shard_idx = flush_shard(shard_data, shard_idx)
        total_tokens += SHARD_SIZE

        leftover = combined[SHARD_SIZE:]
        if leftover.shape[0] > 0:
            buffer = [leftover]
            buffer_size = leftover.shape[0]
        else:
            buffer = []
            buffer_size = 0

    del out, layer_acts, batch
    torch.cuda.empty_cache()

# Flush remaining buffer
if buffer_size > 0:
    combined = np.concatenate(buffer, axis=0)
    shard_idx = flush_shard(combined, shard_idx)
    total_tokens += combined.shape[0]

# Write metadata.json (ActivationStore format)
metadata = {
    "n_samples": total_tokens,
    "hidden_dim": hidden_dim,
    "n_shards": shard_idx,
    "shard_size": SHARD_SIZE,
    "model_path": CKPT,
    "layer": LAYER,
    "target_layer": target_layer,
    "num_layers": num_layers,
    "n_sequences": len(sequences),
    "context_length": CONTEXT_LENGTH,
    "csv_path": CSV_PATH,
}
with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# Summary
codons_per_seq = [len(s) // 3 for s in sequences]

print(f"\n{'='*60}")
print(f"Activation extraction complete")
print(f"{'='*60}")
print(f"Model:          {CKPT}")
print(f"Layer:          {target_layer} / {num_layers} (penultimate)")
print(f"Hidden dim:     {hidden_dim}")
print(f"Sequences:      {len(sequences)}")
print(f"Total codons:   {total_tokens}")
print(f"Shards:         {shard_idx}")
print(f"")
print(f"Codons per sequence:")
print(f"  min:    {np.min(codons_per_seq)}")
print(f"  max:    {np.max(codons_per_seq)}")
print(f"  mean:   {np.mean(codons_per_seq):.1f}")
print(f"  median: {np.median(codons_per_seq):.1f}")
print(f"{'='*60}")
print(f"Saved to: {OUT_DIR}")
print(f"  {shard_idx} parquet shards + metadata.json")
print(f"  Compatible with: sae.activation_store.load_activations('{OUT_DIR}')")
