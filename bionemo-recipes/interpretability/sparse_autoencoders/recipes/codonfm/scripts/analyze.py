"""
Compute interpretability analysis for CodonFM SAE features.

Generates:
  - Vocabulary logit analysis (which codons each feature promotes/suppresses)
  - Codon-level computed annotations (usage bias, CpG, wobble, amino acid identity)
  - Auto-interp LLM-generated feature labels (optional)

Usage:
    python scripts/analyze.py \
        --checkpoint ./outputs/merged_1b/checkpoints/checkpoint_final.pt \
        --model-path /path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
        --layer -2 --top-k 32 \
        --csv-path /path/to/Primates.csv \
        --dashboard-dir ./outputs/merged_1b/dashboard \
        --output-dir ./outputs/merged_1b/analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

_RECIPE_DIR = Path(__file__).resolve().parent.parent
_CODONFM_DIR = _RECIPE_DIR / "codon-fm"
if _CODONFM_DIR.exists():
    sys.path.insert(0, str(_CODONFM_DIR))

from src.inference.encodon import EncodonInference
from src.data.preprocess.codon_sequence import process_item
from src.tokenizer import Tokenizer

from sae.architectures import TopKSAE
from sae.analysis import compute_feature_logits
from sae.utils import set_seed, get_device

from codonfm_sae.data import read_codon_csv


# ── Standard codon usage table (human, per 1000 codons) ──────────────
# Source: Kazusa Codon Usage Database, Homo sapiens
HUMAN_CODON_USAGE = {
    'TTT': 17.6, 'TTC': 20.3, 'TTA': 7.7, 'TTG': 12.9,
    'CTT': 13.2, 'CTC': 19.6, 'CTA': 7.2, 'CTG': 39.6,
    'ATT': 16.0, 'ATC': 20.8, 'ATA': 7.5, 'ATG': 22.0,
    'GTT': 11.0, 'GTC': 14.5, 'GTA': 7.1, 'GTG': 28.1,
    'TCT': 15.2, 'TCC': 17.7, 'TCA': 12.2, 'TCG': 4.4,
    'CCT': 17.5, 'CCC': 19.8, 'CCA': 16.9, 'CCG': 6.9,
    'ACT': 13.1, 'ACC': 18.9, 'ACA': 15.1, 'ACG': 6.1,
    'GCT': 18.4, 'GCC': 27.7, 'GCA': 15.8, 'GCG': 7.4,
    'TAT': 12.2, 'TAC': 15.3, 'TAA': 1.0, 'TAG': 0.8,
    'CAT': 10.9, 'CAC': 15.1, 'CAA': 12.3, 'CAG': 34.2,
    'AAT': 17.0, 'AAC': 19.1, 'AAA': 24.4, 'AAG': 31.9,
    'GAT': 21.8, 'GAC': 25.1, 'GAA': 29.0, 'GAG': 39.6,
    'TGT': 10.6, 'TGC': 12.6, 'TGA': 1.6, 'TGG': 13.2,
    'CGT': 4.5, 'CGC': 10.4, 'CGA': 6.2, 'CGG': 11.4,
    'AGT': 12.1, 'AGC': 19.5, 'AGA': 12.2, 'AGG': 12.0,
    'GGT': 10.8, 'GGC': 22.2, 'GGA': 16.5, 'GGG': 16.5,
}

CODON_TO_AA = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def parse_args():
    p = argparse.ArgumentParser(description="Analyze CodonFM SAE features")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--top-k", type=int, default=None,
                   help="Override top-k (default: read from checkpoint)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--layer", type=int, default=-2)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--csv-path", type=str, required=True,
                   help="CSV with codon sequences (e.g. Primates.csv)")
    p.add_argument("--num-sequences", type=int, default=None,
                   help="Max sequences to analyze (default: all)")
    p.add_argument("--dashboard-dir", type=str, default=None,
                   help="If provided, updates features_atlas.parquet with labels")
    p.add_argument("--output-dir", type=str, default="./outputs/analysis")
    p.add_argument("--auto-interp", action="store_true",
                   help="Run LLM auto-interpretation")
    p.add_argument("--llm-provider", type=str, default="anthropic",
                   choices=["anthropic", "openai", "nim", "nvidia-internal"],
                   help="LLM provider for auto-interp (default: anthropic)")
    p.add_argument("--llm-model", type=str, default=None,
                   help="LLM model name (defaults: anthropic=claude-sonnet-4-20250514, openai=gpt-4o, nim=nvidia/llama-3.1-nemotron-70b-instruct, nvidia-internal=aws/anthropic/bedrock-claude-3-7-sonnet-v1)")
    p.add_argument("--max-features", type=int, default=None,
                   help="Limit number of features to analyze (for testing)")
    p.add_argument("--max-auto-interp-features", type=int, default=None,
                   help="Limit auto-interp to top N features by activation frequency (default: all with codon annotations)")
    p.add_argument("--auto-interp-workers", type=int, default=1,
                   help="Number of parallel workers for LLM calls (default: 1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_sae(checkpoint_path: str, top_k_override: int | None = None) -> TopKSAE:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    w = state_dict["encoder.weight"]
    hidden_dim, input_dim = w.shape
    model_config = ckpt.get("model_config", {})
    top_k = top_k_override or model_config.get("top_k")
    if top_k is None:
        raise ValueError("top_k not found in checkpoint. Pass --top-k explicitly.")
    if top_k_override and model_config.get("top_k") and top_k_override != model_config["top_k"]:
        print(f"  WARNING: overriding checkpoint top_k={model_config['top_k']} with --top-k={top_k_override}")
    sae = TopKSAE(
        input_dim=input_dim, hidden_dim=hidden_dim, top_k=top_k,
        normalize_input=model_config.get("normalize_input", False),
    )
    sae.load_state_dict(state_dict)
    return sae


# ── 1. Vocabulary logit analysis ─────────────────────────────────────

def compute_vocab_logits(sae, inference, device="cuda"):
    """Project SAE decoder through the Encodon LM head to get per-feature codon logits."""
    encodon = inference.model.model
    tokenizer = inference.tokenizer

    # Build vocab list indexed by token ID
    vocab = [tokenizer.decoder.get(i, f"<{i}>") for i in range(tokenizer.vocab_size)]

    # Get the LM head (cls module)
    lm_head = encodon.cls

    # Decoder weights: (input_dim, n_features)
    W_dec = sae.decoder.weight.to(device)

    # Project each feature's decoder column through the LM head
    with torch.no_grad():
        # LM head expects (batch, hidden_dim) and outputs (batch, vocab_size)
        logits = lm_head(W_dec.T)  # (n_features, vocab_size)

    # Build per-feature top promoted/suppressed codons
    n_features = logits.shape[0]
    results = {}
    for f in range(n_features):
        feat_logits = logits[f].cpu()
        top_pos_idx = feat_logits.topk(10).indices.tolist()
        top_neg_idx = feat_logits.topk(10, largest=False).indices.tolist()

        top_positive = [(vocab[i], feat_logits[i].item()) for i in top_pos_idx]
        top_negative = [(vocab[i], feat_logits[i].item()) for i in top_neg_idx]

        # Group top positive by amino acid
        aa_counts = {}
        for codon, val in top_positive:
            aa = CODON_TO_AA.get(codon, "?")
            aa_counts[aa] = aa_counts.get(aa, 0) + 1

        results[f] = {
            "top_positive": top_positive,
            "top_negative": top_negative,
            "top_aa_counts": aa_counts,
        }

    return results


# ── 2. Codon-level annotations ───────────────────────────────────────

def compute_codon_annotations(
    sae, activations_3d, masks, sequences, device="cuda"
):
    """Compute per-codon annotation correlations (amino acid, usage, CpG, wobble, position)."""
    n_sequences = activations_3d.shape[0]
    n_features = sae.hidden_dim
    valid_lens = masks.sum(dim=1).long()

    # Accumulate per-feature correlation with annotations
    # We'll use simple mean activation for positive vs negative sets
    aa_activations = {aa: {f: [] for f in range(n_features)} for aa in set(CODON_TO_AA.values())}
    rare_codon_acts = {f: [] for f in range(n_features)}
    common_codon_acts = {f: [] for f in range(n_features)}
    cpg_acts = {f: [] for f in range(n_features)}
    non_cpg_acts = {f: [] for f in range(n_features)}
    wobble_gc_acts = {f: [] for f in range(n_features)}
    wobble_at_acts = {f: [] for f in range(n_features)}
    first30_acts = {f: [] for f in range(n_features)}
    rest_acts = {f: [] for f in range(n_features)}

    print("  Computing codon-level annotations...")
    for i in tqdm(range(n_sequences), desc="  Codon annotations"):
        vl = int(valid_lens[i].item())
        if vl == 0:
            continue

        seq = sequences[i]
        codons = [seq[j*3:(j+1)*3] for j in range(vl)]

        emb = activations_3d[i, :vl, :].to(device)
        with torch.no_grad():
            _, codes = sae(emb)
        codes_cpu = codes.cpu().numpy()

        for j, codon in enumerate(codons):
            codon_upper = codon.upper()
            aa = CODON_TO_AA.get(codon_upper, "?")
            usage = HUMAN_CODON_USAGE.get(codon_upper, 10.0)
            wobble = codon_upper[2] if len(codon_upper) == 3 else "?"

            # CpG: check if this codon ends with C and next starts with G (or vice versa)
            is_cpg = False
            if j < vl - 1:
                next_codon = codons[j + 1].upper() if j + 1 < len(codons) else ""
                if len(codon_upper) == 3 and len(next_codon) >= 1:
                    is_cpg = (codon_upper[2] == "C" and next_codon[0] == "G")

            for f in range(n_features):
                act = float(codes_cpu[j, f])
                if act <= 0:
                    continue

                # Amino acid
                aa_activations[aa][f].append(act)

                # Codon usage
                if usage < 10.0:
                    rare_codon_acts[f].append(act)
                else:
                    common_codon_acts[f].append(act)

                # CpG
                if is_cpg:
                    cpg_acts[f].append(act)
                else:
                    non_cpg_acts[f].append(act)

                # Wobble position
                if wobble in ("G", "C"):
                    wobble_gc_acts[f].append(act)
                else:
                    wobble_at_acts[f].append(act)

                # Position in gene
                if j < 30:
                    first30_acts[f].append(act)
                else:
                    rest_acts[f].append(act)

    # Summarize: for each feature, find strongest amino acid
    print("  Summarizing annotations...")
    results = {}
    for f in range(n_features):
        annotations = {}

        # Best amino acid
        best_aa = None
        best_aa_count = 0
        total_fires = sum(len(aa_activations[aa][f]) for aa in aa_activations)
        if total_fires > 0:
            for aa in aa_activations:
                count = len(aa_activations[aa][f])
                frac = count / total_fires if total_fires > 0 else 0
                if count > best_aa_count:
                    best_aa_count = count
                    best_aa = aa
            if best_aa and best_aa_count / total_fires > 0.3:
                annotations["amino_acid"] = {
                    "aa": best_aa,
                    "fraction": best_aa_count / total_fires,
                }

        # Rare vs common codons
        n_rare = len(rare_codon_acts[f])
        n_common = len(common_codon_acts[f])
        if n_rare + n_common > 10:
            rare_frac = n_rare / (n_rare + n_common)
            if rare_frac > 0.6:
                annotations["codon_usage"] = {"bias": "rare", "fraction": rare_frac}
            elif rare_frac < 0.2:
                annotations["codon_usage"] = {"bias": "common", "fraction": 1 - rare_frac}

        # CpG
        n_cpg = len(cpg_acts[f])
        n_non = len(non_cpg_acts[f])
        if n_cpg + n_non > 10:
            cpg_frac = n_cpg / (n_cpg + n_non)
            if cpg_frac > 0.3:  # CpG is normally ~1-2% of dinucleotides
                annotations["cpg"] = {"enrichment": cpg_frac}

        # Wobble preference
        n_gc = len(wobble_gc_acts[f])
        n_at = len(wobble_at_acts[f])
        if n_gc + n_at > 10:
            gc_frac = n_gc / (n_gc + n_at)
            if gc_frac > 0.7:
                annotations["wobble"] = {"preference": "GC", "fraction": gc_frac}
            elif gc_frac < 0.3:
                annotations["wobble"] = {"preference": "AT", "fraction": 1 - gc_frac}

        # Position in gene
        n_first = len(first30_acts[f])
        n_rest = len(rest_acts[f])
        if n_first + n_rest > 10:
            first_frac = n_first / (n_first + n_rest)
            expected_frac = 30 / 600  # roughly 5% for avg gene
            if first_frac > expected_frac * 3:
                annotations["position"] = {"region": "N-terminal", "enrichment": first_frac / expected_frac}

        if annotations:
            results[f] = annotations

    return results


# ── 3. Auto-interpretation ───────────────────────────────────────────

def get_llm_client(provider: str, model: str = None):
    """Create LLM client based on provider."""
    from sae.autointerp import (
        AnthropicClient,
        OpenAIClient,
        NIMClient,
        NVIDIAInternalClient,
    )

    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-sonnet-4-20250514")
    elif provider == "openai":
        return OpenAIClient(model=model or "gpt-4o")
    elif provider == "nim":
        return NIMClient(model=model or "nvidia/llama-3.1-nemotron-70b-instruct")
    elif provider == "nvidia-internal":
        return NVIDIAInternalClient(model=model or "aws/anthropic/bedrock-claude-3-7-sonnet-v1")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def run_auto_interp(
    sae, vocab_logits, activations_3d, masks, sequences,
    feature_indices, device="cuda", llm_provider="anthropic", llm_model=None,
    num_workers=1,
):
    """Run LLM auto-interpretation on selected features."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = get_llm_client(llm_provider, llm_model)
    n_features = sae.hidden_dim
    valid_lens = masks.sum(dim=1).long()

    # Build per-feature example strings
    print("  Preparing examples for auto-interp...")
    feature_examples = {}
    for f in tqdm(feature_indices, desc="  Collecting examples"):
        # Find top sequences for this feature
        max_acts = []
        for i in range(activations_3d.shape[0]):
            vl = int(valid_lens[i].item())
            if vl == 0:
                continue
            emb = activations_3d[i, :vl, :].to(device)
            with torch.no_grad():
                _, codes = sae(emb)
            max_act = codes[:, f].max().item()
            max_acts.append((max_act, i))

        max_acts.sort(reverse=True)
        top_seqs = max_acts[:5]

        examples = []
        for max_act, seq_idx in top_seqs:
            vl = int(valid_lens[seq_idx].item())
            seq = sequences[seq_idx]
            codons = [seq[j*3:(j+1)*3] for j in range(vl)]

            emb = activations_3d[seq_idx, :vl, :].to(device)
            with torch.no_grad():
                _, codes = sae(emb)
            acts = codes[:, f].cpu().numpy()

            # Mark top activating codons
            threshold = np.percentile(acts[acts > 0], 80) if (acts > 0).sum() > 0 else 0
            marked = []
            for j, (codon, act) in enumerate(zip(codons, acts)):
                aa = CODON_TO_AA.get(codon.upper(), "?")
                if act > threshold:
                    marked.append(f"***{codon}({aa})***")
                else:
                    marked.append(f"{codon}({aa})")
            examples.append(" ".join(marked))

        feature_examples[f] = examples

    # Build prompts and call LLM in parallel
    print(f"  Running LLM interpretation with {num_workers} workers...")

    def interpret_feature(f):
        """Interpret a single feature and score confidence."""
        logits_info = vocab_logits.get(f, {})
        top_pos = logits_info.get("top_positive", [])[:5]
        top_neg = logits_info.get("top_negative", [])[:5]

        pos_str = ", ".join(f"{tok}({CODON_TO_AA.get(tok, '?')}): {v:.2f}" for tok, v in top_pos)
        neg_str = ", ".join(f"{tok}({CODON_TO_AA.get(tok, '?')}): {v:.2f}" for tok, v in top_neg)

        examples_str = "\n".join(f"  Seq {i+1}: {ex}" for i, ex in enumerate(feature_examples.get(f, [])))

        prompt = f"""This is a feature from a sparse autoencoder trained on a DNA codon language model (CodonFM).
Each token is a codon (3 nucleotides) that encodes an amino acid.

Top promoted codons (decoder logits): {pos_str}
Top suppressed codons: {neg_str}

Top activating sequences (***highlighted*** = high activation):
{examples_str}

In 1 short sentence starting with "Fires on", describe what biological pattern this feature detects.
Consider: amino acid identity, specific codon choice, codon usage bias, positional context, CpG sites, wobble position patterns.

Format your response as:
Label: <one short phrase>
Confidence: <0.00 to 1.00>"""

        try:
            response = client.generate(prompt)
            text = response.text.strip()

            # Parse label and confidence from response
            label = None
            confidence = 0.0

            for line in text.split('\n'):
                if line.startswith('Label:'):
                    label = line.replace('Label:', '').strip()
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.replace('Confidence:', '').strip())
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                    except ValueError:
                        confidence = 0.0

            # Fallback if parsing failed
            if not label:
                label = f"Feature {f}"

            return f, label, confidence
        except Exception as e:
            print(f"  Warning: auto-interp failed for feature {f}: {e}")
            return f, f"Feature {f}", 0.0

    interpretations = {}
    confidences = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(interpret_feature, f): f for f in feature_indices}
        for future in tqdm(as_completed(futures), total=len(feature_indices), desc="  Auto-interp"):
            f, label, confidence = future.result()
            interpretations[f] = label
            confidences[f] = confidence

    return interpretations, confidences


# ── Build summary labels ─────────────────────────────────────────────

def build_feature_labels(
    n_features, vocab_logits, codon_annotations, auto_interp_labels=None,
):
    """Combine all analyses into a single label per feature."""
    labels = {}
    details = {}
    llm_confidences = {}  # Track LLM label confidence scores

    for f in range(n_features):
        parts = []

        # Auto-interp label takes priority
        if auto_interp_labels and f in auto_interp_labels:
            label_entry = auto_interp_labels[f]
            # Handle both new dict format and old string format
            if isinstance(label_entry, dict):
                labels[f] = label_entry.get("label", f"Feature {f}")
                llm_confidences[f] = label_entry.get("confidence", 0.0)
            else:
                labels[f] = label_entry
                llm_confidences[f] = 0.0
            details[f] = {
                "label": labels[f],
                "llm_confidence": llm_confidences[f],
                "vocab_logits": vocab_logits.get(f, {}),
                "codon_annotations": codon_annotations.get(f, {}),
            }
            continue

        # No LLM label for features with only codon annotations
        llm_confidences[f] = 0.0

        # Amino acid identity
        ann = codon_annotations.get(f, {})
        if "amino_acid" in ann:
            aa = ann["amino_acid"]["aa"]
            frac = ann["amino_acid"]["fraction"]
            parts.append(f"{aa} ({frac:.0%})")

        # Codon usage
        if "codon_usage" in ann:
            parts.append(f"{ann['codon_usage']['bias']} codons")

        # Wobble
        if "wobble" in ann:
            parts.append(f"wobble {ann['wobble']['preference']}")

        # CpG
        if "cpg" in ann:
            parts.append("CpG enriched")

        # Position
        if "position" in ann:
            parts.append("N-terminal")

        if parts:
            labels[f] = " | ".join(parts)
        else:
            labels[f] = f"Feature {f}"

        details[f] = {
            "label": labels[f],
            "llm_confidence": llm_confidences[f],
            "vocab_logits": vocab_logits.get(f, {}),
            "codon_annotations": codon_annotations.get(f, {}),
        }

    return labels, details, llm_confidences


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    # Load SAE
    sae = load_sae(args.checkpoint, top_k_override=args.top_k).eval().to(device)
    n_features = sae.hidden_dim
    print(f"SAE: {sae.input_dim} -> {n_features} features")

    # Load model
    print(f"\nLoading Encodon from {args.model_path}...")
    inference = EncodonInference(model_path=args.model_path, task_type="embedding_prediction")
    inference.configure_model()
    inference.model.to(device).eval()

    # Check for activations checkpoint first
    activations_ckpt = output_dir / "activations_checkpoint.pt"
    if activations_ckpt.exists():
        print("\nLoading activations from checkpoint...")
        ckpt = torch.load(activations_ckpt)
        activations_3d = ckpt["activations_3d"]
        masks = ckpt["masks"]
        sequences = ckpt["sequences"]
        print(f"Loaded: {activations_3d.shape}")
    else:
        # Load sequences
        max_codons = args.context_length - 2
        records = read_codon_csv(
            args.csv_path,
            max_sequences=args.num_sequences,
            max_codons=max_codons,
        )
        sequences = [r.sequence for r in records]
        print(f"Loaded {len(sequences)} sequences")

        # Extract 3D activations
        print("\nExtracting activations...")
        all_emb, all_masks = [], []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), args.batch_size), desc="Extracting"):
                batch_seqs = sequences[i:i + args.batch_size]
                items = [process_item(s, context_length=args.context_length, tokenizer=inference.tokenizer) for s in batch_seqs]
                batch = {
                    "input_ids": torch.tensor(np.stack([it["input_ids"] for it in items])).to(device),
                    "attention_mask": torch.tensor(np.stack([it["attention_mask"] for it in items])).to(device),
                }
                out = inference.model(batch, return_hidden_states=True)
                hidden = out.all_hidden_states[args.layer].float().cpu()
                attn = batch["attention_mask"].cpu()
                keep = attn.clone()
                keep[:, 0] = 0
                lengths = attn.sum(dim=1)
                for b in range(keep.shape[0]):
                    sep = int(lengths[b].item()) - 1
                    if sep > 0:
                        keep[b, sep] = 0
                all_emb.append(hidden)
                all_masks.append(keep)
                del out, batch
                torch.cuda.empty_cache()

        max_len = max(e.shape[1] for e in all_emb)
        hdim = all_emb[0].shape[2]
        padded_emb, padded_masks = [], []
        for emb, msk in zip(all_emb, all_masks):
            B, L, D = emb.shape
            if L < max_len:
                emb = torch.cat([emb, torch.zeros(B, max_len - L, D)], dim=1)
                msk = torch.cat([msk, torch.zeros(B, max_len - L, dtype=msk.dtype)], dim=1)
            padded_emb.append(emb)
            padded_masks.append(msk)
        activations_3d = torch.cat(padded_emb, dim=0)
        masks = torch.cat(padded_masks, dim=0)
        print(f"Activations: {activations_3d.shape}")

        # Save checkpoint
        torch.save({
            "activations_3d": activations_3d,
            "masks": masks,
            "sequences": sequences,
        }, activations_ckpt)
        print(f"Saved activations checkpoint to {activations_ckpt}")

    # ── Analysis ─────────────────────────────────────────────────────

    # 1. Vocabulary logits
    print("\n[1/3] Vocabulary logit analysis...")
    vocab_logits_file = output_dir / "vocab_logits_checkpoint.json"
    if vocab_logits_file.exists():
        print("  Loading vocab logits from checkpoint...")
        with open(vocab_logits_file) as f:
            vocab_logits = json.load(f)
            vocab_logits = {int(k): v for k, v in vocab_logits.items()}
    else:
        vocab_logits = compute_vocab_logits(sae, inference, device)
        with open(vocab_logits_file, "w") as f:
            json.dump(vocab_logits, f)
    print(f"  Computed logits for {len(vocab_logits)} features")

    # 2. Codon-level annotations
    print("\n[2/3] Codon-level annotations...")
    codon_annotations_file = output_dir / "codon_annotations_checkpoint.json"
    if codon_annotations_file.exists():
        print("  Loading codon annotations from checkpoint...")
        with open(codon_annotations_file) as f:
            codon_annotations = json.load(f)
            codon_annotations = {int(k): v for k, v in codon_annotations.items()}
    else:
        codon_annotations = compute_codon_annotations(sae, activations_3d, masks, sequences, device)
        with open(codon_annotations_file, "w") as f:
            json.dump(codon_annotations, f, default=str)
    print(f"  {len(codon_annotations)} features with codon annotations")

    # 3. Auto-interp (optional)
    auto_interp_labels = {}
    auto_interp_ckpt = output_dir / "auto_interp_checkpoint.json"
    if auto_interp_ckpt.exists():
        print("  Loading auto-interp checkpoint...")
        with open(auto_interp_ckpt) as f:
            ckpt_data = json.load(f)
            # Handle both old format (string) and new format (dict with label/confidence)
            for k, v in ckpt_data.items():
                k_int = int(k)
                if isinstance(v, dict):
                    auto_interp_labels[k_int] = v
                else:
                    # Old format: string label, default confidence to 0.0
                    auto_interp_labels[k_int] = {"label": v, "confidence": 0.0}
        print(f"  Loaded {len(auto_interp_labels)} existing interpretations")

    if args.auto_interp:
        print("\n[3/3] Auto-interpretation (LLM)...")
        # Get all alive features
        alive_features = [f for f in range(n_features) if f in codon_annotations]

        # Sort by vocab logits magnitude (proxy for feature importance)
        alive_features_sorted = sorted(
            alive_features,
            key=lambda f: max([abs(v) for _, v in vocab_logits[f].get("top_positive", [])], default=0),
            reverse=True,
        )

        # Limit to max features if specified
        if args.max_auto_interp_features:
            alive_features_sorted = alive_features_sorted[:args.max_auto_interp_features]

        # Filter out already-done features
        todo_features = [f for f in alive_features_sorted if f not in auto_interp_labels]

        if todo_features:
            print(f"  Running auto-interp on {len(todo_features)} features "
                  f"({len(auto_interp_labels)} already done)")
            new_labels, new_confidences = run_auto_interp(
                sae, vocab_logits, activations_3d, masks, sequences,
                todo_features, device, llm_provider=args.llm_provider, llm_model=args.llm_model,
                num_workers=args.auto_interp_workers,
            )
            # Store as dicts with label and confidence
            for f in new_labels:
                auto_interp_labels[f] = {
                    "label": new_labels[f],
                    "confidence": new_confidences[f],
                }
            # Save checkpoint after each batch
            with open(auto_interp_ckpt, "w") as f:
                json.dump(auto_interp_labels, f, indent=2)
        else:
            print(f"  All {len(auto_interp_labels)} features already interpreted")
    else:
        print("\n[3/3] Skipping auto-interp (use --auto-interp to enable)")

    # Build labels
    print("\nBuilding feature labels...")
    labels, details, llm_confidences = build_feature_labels(
        n_features, vocab_logits, codon_annotations, auto_interp_labels,
    )
    n_labeled = sum(1 for v in labels.values() if not v.startswith("Feature "))
    print(f"  {n_labeled}/{n_features} features labeled")

    # Save results
    print("\nSaving results...")

    # Full analysis JSON
    with open(output_dir / "feature_analysis.json", "w") as f:
        json.dump(details, f, indent=2, default=str)

    # Labels JSON (for dashboard)
    with open(output_dir / "feature_labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    # LLM confidences JSON (for dashboard sorting)
    with open(output_dir / "llm_confidences.json", "w") as f:
        json.dump(llm_confidences, f, indent=2)

    # Vocab logits JSON (for dashboard detail page)
    logits_export = {}
    for feat_id, data in vocab_logits.items():
        logits_export[str(feat_id)] = {
            "top_positive": [[tok, round(val, 3)] for tok, val in data["top_positive"]],
            "top_negative": [[tok, round(val, 3)] for tok, val in data["top_negative"]],
        }
    with open(output_dir / "vocab_logits.json", "w") as f:
        json.dump(logits_export, f)

    # Update dashboard atlas if requested
    if args.dashboard_dir:
        dashboard_dir = Path(args.dashboard_dir)
        atlas_path = dashboard_dir / "features_atlas.parquet"
        if atlas_path.exists():
            import pyarrow.parquet as pq
            import pyarrow as pa

            print(f"\nUpdating {atlas_path} with labels and confidence scores...")
            table = pq.read_table(atlas_path)
            n = table.num_rows
            label_col = [labels.get(i, f"Feature {i}") for i in range(n)]
            confidence_col = [llm_confidences.get(i, 0.0) for i in range(n)]
            table = table.drop("label") if "label" in table.column_names else table
            table = table.drop("llm_confidence") if "llm_confidence" in table.column_names else table
            table = table.append_column("label", pa.array(label_col))
            table = table.append_column("llm_confidence", pa.array(confidence_col, type=pa.float32()))
            pq.write_table(table, atlas_path, compression='snappy')
            print(f"  Updated {n} feature labels and confidence scores in atlas")

    # Copy vocab_logits.json to dashboard dir for the detail page
    if args.dashboard_dir:
        import shutil
        dashboard_dir = Path(args.dashboard_dir)
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["vocab_logits.json", "feature_labels.json", "feature_analysis.json"]:
            src = output_dir / fname
            dst = dashboard_dir / fname
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied {fname} to dashboard dir")

    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
