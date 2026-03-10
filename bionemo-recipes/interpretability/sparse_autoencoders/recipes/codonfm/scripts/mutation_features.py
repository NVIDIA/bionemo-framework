"""
Analyze which SAE features fire at mutation sites in the validation data.

For each sequence in the merged validation CSV that has an annotated mutation
position (var_pos_offset), this script:
  1. Runs the sequence through the Encodon model to get hidden states
  2. Encodes the hidden states through the trained SAE to get per-codon feature codes
  3. Records which features are active at the mutation site codon
  4. Compares feature activity at mutation sites between pathogenic and benign variants

Usage:
    python scripts/mutation_features.py \
        --checkpoint ./outputs/primates_1b_p/checkpoints/checkpoint_final.pt \
        --model-path ../../../checkpoints/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
        --layer -2 --top-k 32 \
        --csv-path ../../../codonfm/data/merged_seqs_for_validation.csv \
        --output-dir ./outputs/primates_1b_p/mutation_analysis
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_CODONFM_DIR = _RECIPE_DIR / "codon-fm"
if _CODONFM_DIR.exists():
    sys.path.insert(0, str(_CODONFM_DIR))

from src.inference.encodon import EncodonInference
from src.data.preprocess.codon_sequence import process_item

from sae.architectures import TopKSAE
from sae.utils import set_seed, get_device


def parse_args():
    p = argparse.ArgumentParser(description="Analyze SAE features at mutation sites")
    p.add_argument("--checkpoint", type=str, required=True, help="SAE checkpoint .pt file")
    p.add_argument("--top-k", type=int, default=None,
                   help="Override top-k (default: read from checkpoint)")
    p.add_argument("--model-path", type=str, required=True, help="Encodon checkpoint")
    p.add_argument("--layer", type=int, default=-2)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--csv-path", type=str, required=True, help="Merged validation CSV with mutation metadata")
    p.add_argument("--output-dir", type=str, default="./outputs/mutation_analysis")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_sae(checkpoint_path, top_k_override=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    w = state["encoder.weight"]
    model_config = ckpt.get("model_config", {})
    top_k = top_k_override or model_config.get("top_k")
    if top_k is None:
        raise ValueError("top_k not found in checkpoint. Pass --top-k explicitly.")
    if top_k_override and model_config.get("top_k") and top_k_override != model_config["top_k"]:
        print(f"  WARNING: overriding checkpoint top_k={model_config['top_k']} with --top-k={top_k_override}")
    sae = TopKSAE(
        input_dim=w.shape[1], hidden_dim=w.shape[0], top_k=top_k,
        normalize_input=model_config.get("normalize_input", False),
    )
    sae.load_state_dict(state)
    return sae


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAE
    sae = load_sae(args.checkpoint, top_k_override=args.top_k).eval().to(device)
    n_features = sae.hidden_dim
    print(f"SAE: {sae.input_dim} -> {n_features} features (top-{sae.top_k})")

    # Load Encodon
    print(f"Loading Encodon from {args.model_path}...")
    inf = EncodonInference(model_path=args.model_path, task_type="embedding_prediction")
    inf.configure_model()
    inf.model.to(device).eval()

    # Load data
    df = pd.read_csv(args.csv_path)
    seq_col = next(c for c in ["seq", "cds", "sequence"] if c in df.columns)
    max_codons = args.context_length - 2
    df = df[df[seq_col].str.len() // 3 <= max_codons].reset_index(drop=True)
    print(f"Loaded {len(df)} sequences")

    # Find mutation position column
    pos_col = next((c for c in ["var_pos_offset", "codon_position"] if c in df.columns), None)
    if pos_col is None:
        print("ERROR: No mutation position column found (var_pos_offset or codon_position)")
        return
    n_with_mut = (df[pos_col] >= 0).sum()
    print(f"Sequences with mutation position ({pos_col} >= 0): {n_with_mut}")

    # Encode each sequence and record features at mutation sites
    results = []
    background_results = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Encoding sequences"):
        mut_pos = row[pos_col]
        if pd.isna(mut_pos) or int(mut_pos) < 0:
            continue
        mut_pos = int(mut_pos)

        seq = row[seq_col]
        n_codons = len(seq) // 3
        if mut_pos >= n_codons:
            continue

        item = process_item(seq, context_length=args.context_length, tokenizer=inf.tokenizer)
        batch = {
            "input_ids": torch.tensor(item["input_ids"]).unsqueeze(0).to(device),
            "attention_mask": torch.tensor(item["attention_mask"]).unsqueeze(0).to(device),
        }

        with torch.no_grad():
            out = inf.model(batch, return_hidden_states=True)
            hidden = out.all_hidden_states[args.layer][0, 1:, :]  # strip CLS
            _, codes = sae(hidden[:n_codons])  # (n_codons, n_features)

        # Features active at mutation site
        mut_codes = codes[mut_pos].cpu()
        active_mask = mut_codes > 0
        active_features = torch.where(active_mask)[0].tolist()
        active_values = mut_codes[active_mask].tolist()

        for feat, val in zip(active_features, active_values):
            results.append({
                "seq_id": row.get("id", i),
                "gene": row.get("gene", ""),
                "mut_pos": mut_pos,
                "ref_codon": row.get("ref_codon", ""),
                "alt_codon": row.get("alt_codon", ""),
                "is_pathogenic": row.get("is_pathogenic", ""),
                "mutation_desc": row.get("MUTATION_DESCRIPTION", ""),
                "cancer_role": row.get("ROLE_IN_CANCER", ""),
                "feature_id": feat,
                "activation_at_mut": val,
            })

        # Background: mean activation across all non-mutation positions
        bg_codes = codes.cpu()
        bg_mask = torch.ones(n_codons, dtype=torch.bool)
        bg_mask[mut_pos] = False
        if bg_mask.sum() > 0:
            bg_mean = bg_codes[bg_mask].mean(dim=0)
            for feat in active_features:
                background_results.append({
                    "seq_id": row.get("id", i),
                    "feature_id": feat,
                    "activation_at_mut": mut_codes[feat].item(),
                    "activation_background": bg_mean[feat].item(),
                })

        del out, hidden, codes, batch
        torch.cuda.empty_cache()

    # Save raw results
    rdf = pd.DataFrame(results)
    rdf.to_csv(output_dir / "mutation_features.csv", index=False)

    bdf = pd.DataFrame(background_results)
    bdf.to_csv(output_dir / "mutation_vs_background.csv", index=False)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Mutation Site Feature Analysis")
    print(f"{'='*60}")
    print(f"Sequences analyzed:                  {n_with_mut}")
    print(f"Feature activations at mutation sites: {len(rdf)}")
    print(f"Unique features at mutation sites:     {rdf.feature_id.nunique()}")

    # Top features by frequency at mutation sites
    top = rdf.groupby("feature_id").agg(
        count=("activation_at_mut", "len"),
        mean_act=("activation_at_mut", "mean"),
        max_act=("activation_at_mut", "max"),
    ).sort_values("count", ascending=False)
    top.to_csv(output_dir / "top_features_at_mutations.csv")

    print(f"\nTop 20 features at mutation sites:")
    print(top.head(20).to_string())

    # Enrichment: activation at mutation vs background
    if len(bdf) > 0:
        enrichment = bdf.groupby("feature_id").agg(
            n=("activation_at_mut", "len"),
            mean_at_mut=("activation_at_mut", "mean"),
            mean_bg=("activation_background", "mean"),
        )
        enrichment["enrichment"] = enrichment["mean_at_mut"] / (enrichment["mean_bg"] + 1e-8)
        enrichment = enrichment[enrichment.n >= 3].sort_values("enrichment", ascending=False)
        enrichment.to_csv(output_dir / "mutation_enrichment.csv")

        print(f"\nTop 15 features enriched at mutation sites vs background:")
        print(enrichment.head(15).to_string())

    # Pathogenic vs benign
    if "is_pathogenic" in rdf.columns:
        rdf_clean = rdf[rdf.is_pathogenic.isin([True, False, "true", "false"])].copy()
        rdf_clean["is_path"] = rdf_clean.is_pathogenic.astype(str).str.lower() == "true"

        if len(rdf_clean) > 0:
            path_counts = rdf_clean[rdf_clean.is_path].feature_id.value_counts()
            benign_counts = rdf_clean[~rdf_clean.is_path].feature_id.value_counts()
            all_feats = set(path_counts.index) | set(benign_counts.index)

            comparison = []
            for f in all_feats:
                p = path_counts.get(f, 0)
                b = benign_counts.get(f, 0)
                if p + b >= 3:
                    comparison.append({
                        "feature_id": f,
                        "pathogenic_count": p,
                        "benign_count": b,
                        "total": p + b,
                        "pathogenic_ratio": p / (p + b),
                    })

            cdf = pd.DataFrame(comparison).sort_values("pathogenic_ratio", ascending=False)
            cdf.to_csv(output_dir / "pathogenic_vs_benign.csv", index=False)

            print(f"\nFeatures most associated with PATHOGENIC mutations:")
            print(cdf.head(10).to_string(index=False))
            print(f"\nFeatures most associated with BENIGN mutations:")
            print(cdf.tail(10).to_string(index=False))

    # By mutation type (synonymous vs missense etc)
    if "mutation_desc" in rdf.columns:
        by_type = rdf.groupby(["mutation_desc", "feature_id"]).size().reset_index(name="count")
        by_type = by_type.sort_values("count", ascending=False)
        by_type.to_csv(output_dir / "features_by_mutation_type.csv", index=False)

        for mut_type in rdf.mutation_desc.dropna().unique():
            subset = by_type[by_type.mutation_desc == mut_type].head(5)
            if len(subset) > 0:
                print(f"\nTop features for '{mut_type}':")
                print(subset.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"  mutation_features.csv          — all (sequence, feature) pairs at mutation sites")
    print(f"  mutation_vs_background.csv     — activation at mutation vs background per feature")
    print(f"  top_features_at_mutations.csv  — features ranked by frequency at mutations")
    print(f"  mutation_enrichment.csv        — features ranked by enrichment (mut/background)")
    print(f"  pathogenic_vs_benign.csv       — features split by pathogenicity")
    print(f"  features_by_mutation_type.csv  — features split by mutation type")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
