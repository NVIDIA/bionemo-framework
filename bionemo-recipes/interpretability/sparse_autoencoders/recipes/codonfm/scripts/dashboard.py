"""
Generate dashboard data from a trained CodonFM SAE.

Loads the SAE checkpoint + Encodon model, runs sequences through both,
and exports feature statistics + per-sequence activation examples
to parquet files for the interactive dashboard.

    python scripts/dashboard.py \
        --checkpoint ./outputs/merged_1b/checkpoints/checkpoint_final.pt \
        --model-path /path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
        --layer -2 --top-k 32 \
        --csv-path /path/to/Primates.csv \
        --output-dir ./outputs/merged_1b/dashboard
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add codon-fm to path
_RECIPE_DIR = Path(__file__).resolve().parent.parent
_CODONFM_DIR = _RECIPE_DIR / "codon-fm"
if _CODONFM_DIR.exists():
    sys.path.insert(0, str(_CODONFM_DIR))

from src.inference.encodon import EncodonInference
from src.data.preprocess.codon_sequence import process_item

from sae.architectures import TopKSAE
from sae.analysis import compute_feature_stats, compute_feature_umap, save_feature_atlas
from sae.utils import set_seed, get_device

from codonfm_sae.data import read_codon_csv
from codonfm_sae.data.uniprot import resolve_gene_to_alphafold


def parse_args():
    p = argparse.ArgumentParser(description="Generate CodonFM SAE dashboard data")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to SAE checkpoint .pt file")
    p.add_argument("--top-k", type=int, default=None,
                   help="Override top-k (default: read from checkpoint)")
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to Encodon checkpoint (.safetensors)")
    p.add_argument("--layer", type=int, default=-2)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--csv-path", type=str, required=True)
    p.add_argument("--seq-column", type=str, default=None)
    p.add_argument("--num-sequences", type=int, default=2000)
    p.add_argument("--n-examples", type=int, default=6,
                   help="Top examples per feature")
    p.add_argument("--output-dir", type=str, default="./outputs/dashboard")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_sae_from_checkpoint(checkpoint_path: str, top_k_override: int | None = None) -> TopKSAE:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    input_dim = ckpt.get("input_dim")
    hidden_dim = ckpt.get("hidden_dim")
    if input_dim is None or hidden_dim is None:
        w = state_dict["encoder.weight"]
        hidden_dim = hidden_dim or w.shape[0]
        input_dim = input_dim or w.shape[1]

    model_config = ckpt.get("model_config", {})
    normalize_input = model_config.get("normalize_input", False)

    # Use checkpoint's top_k by default, allow CLI override
    top_k = top_k_override or model_config.get("top_k")
    if top_k is None:
        raise ValueError(
            "top_k not found in checkpoint model_config. "
            "Pass --top-k explicitly."
        )
    if top_k_override and model_config.get("top_k") and top_k_override != model_config["top_k"]:
        print(f"  WARNING: overriding checkpoint top_k={model_config['top_k']} with --top-k={top_k_override}")

    sae = TopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        top_k=top_k,
        normalize_input=normalize_input,
    )
    sae.load_state_dict(state_dict)
    print(f"Loaded SAE: {input_dim} -> {hidden_dim:,} latents (top-{top_k})")
    return sae


def extract_activations_3d(
    inference,
    sequences: List[str],
    layer: int,
    context_length: int = 2048,
    batch_size: int = 8,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 3D activations (n_sequences, max_seq_len, hidden_dim) + masks.

    Returns padded activations and masks with CLS/SEP excluded.
    """
    all_embeddings = []
    all_masks = []

    n_batches = (len(sequences) + batch_size - 1) // batch_size
    iterator = tqdm(range(0, len(sequences), batch_size), total=n_batches,
                    desc="Extracting activations")

    with torch.no_grad():
        for i in iterator:
            batch_seqs = sequences[i : i + batch_size]
            items = [
                process_item(s, context_length=context_length, tokenizer=inference.tokenizer)
                for s in batch_seqs
            ]

            batch = {
                "input_ids": torch.tensor(np.stack([it["input_ids"] for it in items])).to(device),
                "attention_mask": torch.tensor(np.stack([it["attention_mask"] for it in items])).to(device),
            }

            out = inference.model(batch, return_hidden_states=True)
            hidden = out.all_hidden_states[layer].float().cpu()  # [B, L, D]
            attn_mask = batch["attention_mask"].cpu()

            # Build mask excluding CLS (pos 0) and SEP (last real pos)
            keep = attn_mask.clone()
            keep[:, 0] = 0
            lengths = attn_mask.sum(dim=1)
            for b in range(keep.shape[0]):
                sep = int(lengths[b].item()) - 1
                if sep > 0:
                    keep[b, sep] = 0

            all_embeddings.append(hidden)
            all_masks.append(keep)

            del out, batch
            torch.cuda.empty_cache()

    # Pad to same seq_len across batches
    max_len = max(e.shape[1] for e in all_embeddings)
    hidden_dim = all_embeddings[0].shape[2]

    padded_emb = []
    padded_masks = []
    for emb, msk in zip(all_embeddings, all_masks):
        B, L, D = emb.shape
        if L < max_len:
            emb = torch.cat([emb, torch.zeros(B, max_len - L, D)], dim=1)
            msk = torch.cat([msk, torch.zeros(B, max_len - L, dtype=msk.dtype)], dim=1)
        padded_emb.append(emb)
        padded_masks.append(msk)

    return torch.cat(padded_emb, dim=0), torch.cat(padded_masks, dim=0)


def export_codon_features_parquet(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    sequences: List[str],
    sequence_ids: List[str],
    masks: torch.Tensor,
    output_dir: Path,
    n_examples: int = 6,
    device: str = "cuda",
):
    """Export per-codon feature activations for dashboard.

    Two-pass algorithm:
        Pass 1: compute max activation per (sequence, feature)
        Pass 2: extract per-codon activations for top examples only
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_sequences = activations.shape[0]
    n_features = sae.hidden_dim

    sae = sae.eval().to(device)

    # Valid lengths per sequence (excluding CLS/SEP/padding)
    valid_lens = masks.sum(dim=1).long()

    # Pass 1: max activation per (sequence, feature)
    print("  Pass 1: Computing max activations per sequence...")
    max_acts = torch.zeros(n_sequences, n_features)

    for i in tqdm(range(n_sequences), desc="  Max activations"):
        vl = int(valid_lens[i].item())
        if vl == 0:
            continue
        emb = activations[i, :vl, :].to(device)  # (vl, hidden_dim)
        with torch.no_grad():
            _, codes = sae(emb)  # codes: (vl, n_features)
        max_acts[i] = codes.max(dim=0).values.cpu()

    # Find top examples per feature
    print("  Finding top examples per feature...")
    top_indices = torch.topk(max_acts, k=min(n_examples, n_sequences), dim=0).indices  # (n_examples, n_features)

    # Build reverse index: which sequences need re-encoding
    needed_sequences = {}
    for feat_idx in range(n_features):
        for rank in range(top_indices.shape[0]):
            seq_idx = int(top_indices[rank, feat_idx].item())
            if seq_idx not in needed_sequences:
                needed_sequences[seq_idx] = set()
            needed_sequences[seq_idx].add(feat_idx)

    # Pass 2: extract per-codon activations for top examples
    print(f"  Pass 2: Extracting per-codon activations ({len(needed_sequences)} sequences)...")
    example_acts = {}

    for seq_idx in tqdm(sorted(needed_sequences.keys()), desc="  Per-codon activations"):
        vl = int(valid_lens[seq_idx].item())
        if vl == 0:
            continue
        emb = activations[seq_idx, :vl, :].to(device)
        with torch.no_grad():
            _, codes = sae(emb)  # (vl, n_features)
        codes_cpu = codes.cpu()

        for feat_idx in needed_sequences[seq_idx]:
            example_acts[(seq_idx, feat_idx)] = codes_cpu[:, feat_idx].numpy().tolist()

    # Build feature_metadata.parquet
    print("  Writing feature_metadata.parquet...")
    meta_rows = []
    for feat_idx in range(n_features):
        freq = (max_acts[:, feat_idx] > 0).float().mean().item()
        max_val = max_acts[:, feat_idx].max().item()
        meta_rows.append({
            "feature_id": feat_idx,
            "description": f"Feature {feat_idx}",
            "activation_freq": freq,
            "max_activation": max_val,
        })

    meta_table = pa.table({
        "feature_id": pa.array([r["feature_id"] for r in meta_rows], type=pa.int32()),
        "description": pa.array([r["description"] for r in meta_rows]),
        "activation_freq": pa.array([r["activation_freq"] for r in meta_rows], type=pa.float32()),
        "max_activation": pa.array([r["max_activation"] for r in meta_rows], type=pa.float32()),
    })
    pq.write_table(meta_table, output_dir / "feature_metadata.parquet", compression='snappy')

    # Resolve gene names to AlphaFold IDs
    print("  Resolving gene names to AlphaFold IDs...")
    unique_ids = list(set(sequence_ids))
    alphafold_map = resolve_gene_to_alphafold(unique_ids)
    n_resolved = sum(1 for v in alphafold_map.values() if v)
    print(f"  Resolved {n_resolved}/{len(unique_ids)} sequence IDs to AlphaFold structures")

    # Build feature_examples.parquet
    print("  Writing feature_examples.parquet...")
    example_rows = []
    for feat_idx in range(n_features):
        for rank in range(top_indices.shape[0]):
            seq_idx = int(top_indices[rank, feat_idx].item())
            key = (seq_idx, feat_idx)
            if key not in example_acts:
                continue

            # Get the codon sequence (triplets)
            raw_seq = sequences[seq_idx]
            n_codons = len(raw_seq) // 3
            codon_seq = " ".join(raw_seq[i*3:(i+1)*3] for i in range(n_codons))

            acts_list = example_acts[key]
            seq_id = sequence_ids[seq_idx]

            example_rows.append({
                "feature_id": feat_idx,
                "example_rank": rank,
                "protein_id": seq_id,
                "alphafold_id": alphafold_map.get(seq_id, ""),
                "sequence": codon_seq,
                "activations": acts_list,
                "max_activation": max(acts_list) if acts_list else 0.0,
            })

    # Sort by feature_id for efficient row-group filtering
    example_rows.sort(key=lambda r: (r["feature_id"], r["example_rank"]))

    examples_table = pa.table({
        "feature_id": pa.array([r["feature_id"] for r in example_rows], type=pa.int32()),
        "example_rank": pa.array([r["example_rank"] for r in example_rows], type=pa.int8()),
        "protein_id": pa.array([r["protein_id"] for r in example_rows]),
        "alphafold_id": pa.array([r["alphafold_id"] for r in example_rows]),
        "sequence": pa.array([r["sequence"] for r in example_rows]),
        "activations": pa.array([r["activations"] for r in example_rows],
                                type=pa.list_(pa.float32())),
        "max_activation": pa.array([r["max_activation"] for r in example_rows], type=pa.float32()),
    })

    row_group_size = n_examples * 100
    pq.write_table(examples_table, output_dir / "feature_examples.parquet",
                    row_group_size=row_group_size, compression='snappy')

    print(f"  Wrote {len(meta_rows)} features, {len(example_rows)} examples")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SAE
    sae = load_sae_from_checkpoint(args.checkpoint, top_k_override=args.top_k)
    n_features = sae.hidden_dim

    # 2. Load Encodon
    print(f"\nLoading Encodon from {args.model_path}...")
    inference = EncodonInference(model_path=args.model_path, task_type="embedding_prediction")
    inference.configure_model()
    inference.model.to(device).eval()

    # 3. Load sequences
    max_codons = args.context_length - 2
    records = read_codon_csv(
        args.csv_path,
        seq_column=args.seq_column,
        max_sequences=args.num_sequences,
        max_codons=max_codons,
    )
    sequences = [r.sequence for r in records]
    sequence_ids = [r.id for r in records]
    print(f"Loaded {len(sequences)} sequences for dashboard")

    # 4. Extract 3D activations
    print("\nExtracting 3D activations...")
    activations, masks = extract_activations_3d(
        inference, sequences, args.layer,
        context_length=args.context_length,
        batch_size=args.batch_size,
        device=device,
    )
    activations_flat = activations[masks.bool()]
    print(f"  {activations_flat.shape[0]:,} codons, dim={activations_flat.shape[1]}")

    # 5. Feature statistics
    print("\n[1/4] Computing feature statistics...")
    t0 = time.time()
    stats, _ = compute_feature_stats(sae, activations_flat, device=device)
    print(f"       Done in {time.time() - t0:.1f}s")

    # 6. UMAP from decoder weights
    print("[2/4] Computing UMAP from decoder weights...")
    t0 = time.time()
    geometry = compute_feature_umap(
        sae, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist,
        random_state=args.seed, compute_clusters=True,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
    )
    print(f"       Done in {time.time() - t0:.1f}s")

    # 7. Feature atlas
    print("[3/4] Saving feature atlas...")
    t0 = time.time()
    atlas_path = output_dir / "features_atlas.parquet"
    save_feature_atlas(stats, geometry, atlas_path)
    print(f"       Saved to {atlas_path} in {time.time() - t0:.1f}s")

    # 8. Protein/codon examples
    print("[4/4] Exporting codon examples...")
    t0 = time.time()
    export_codon_features_parquet(
        sae=sae,
        activations=activations,
        sequences=sequences,
        sequence_ids=sequence_ids,
        masks=masks,
        output_dir=output_dir,
        n_examples=args.n_examples,
        device=device,
    )
    print(f"       Done in {time.time() - t0:.1f}s")

    # Free GPU
    del inference
    torch.cuda.empty_cache()

    print(f"\nDashboard data saved to: {output_dir}")
    print(f"  Atlas:    {atlas_path}")
    print(f"  Metadata: {output_dir / 'feature_metadata.parquet'}")
    print(f"  Examples: {output_dir / 'feature_examples.parquet'}")


if __name__ == "__main__":
    main()
