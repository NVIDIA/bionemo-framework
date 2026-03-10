#!/usr/bin/env python3
import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on path (this file is expected in scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress sklearn warnings that clutter output
logging.getLogger("sklearn").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Valid amino acid characters (standard 20 amino acids)
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

from src.inference.encodon import EncodonInference
from src.inference.decodon import DecodonInference
from src.inference.task_types import TaskTypes
from src.tokenizer.tokenizer import Tokenizer  # type: ignore

from src.inference.generation_utils import (
    split_codons,
    random_dna_for_protein,
    generate_stratified_initial_sequences,
    tournament_select,
    crossover_codons,
    mutate_codons,
    amino_acids_from_dna,
)
from src.tokenizer.mappings import CODON_TABLE

ModelInference = Union[EncodonInference, DecodonInference]


# =====================
# Codon Tables and Metrics
# =====================

GENETIC_CODE: Dict[str, str] = CODON_TABLE.DNA

# Human codon usage table (per 1000 codons)
HUMAN_CODON_USAGE = {
    'TTT': 17.6, 'TTC': 20.3, 'TTA': 7.7, 'TTG': 12.9,
    'CTT': 13.2, 'CTC': 19.6, 'CTA': 7.2, 'CTG': 39.6,
    'ATT': 16.0, 'ATC': 20.8, 'ATA': 7.5, 'ATG': 22.0,
    'GTT': 11.0, 'GTC': 14.5, 'GTA': 7.1, 'GTG': 28.1,
    'TAT': 12.2, 'TAC': 15.3, 'TAA': 1.0, 'TAG': 0.8,
    'CAT': 10.9, 'CAC': 15.1, 'CAA': 12.3, 'CAG': 34.2,
    'AAT': 17.0, 'AAC': 19.1, 'AAA': 24.4, 'AAG': 31.9,
    'GAT': 21.8, 'GAC': 25.1, 'GAA': 29.0, 'GAG': 39.6,
    'TCT': 15.2, 'TCC': 17.7, 'TCA': 12.2, 'TCG': 4.4,
    'CCT': 17.5, 'CCC': 19.8, 'CCA': 16.9, 'CCG': 6.9,
    'ACT': 13.1, 'ACC': 18.9, 'ACA': 15.1, 'ACG': 6.1,
    'GCT': 18.4, 'GCC': 27.7, 'GCA': 15.8, 'GCG': 7.4,
    'TGT': 10.6, 'TGC': 12.6, 'TGA': 1.6, 'TGG': 13.2,
    'CGT': 4.5, 'CGC': 10.4, 'CGA': 6.2, 'CGG': 11.4,
    'AGT': 12.1, 'AGC': 19.5, 'AGA': 12.2, 'AGG': 12.0,
    'GGT': 10.8, 'GGC': 22.2, 'GGA': 16.5, 'GGG': 16.5,
}

# Mouse codon usage table (per 1000 codons)
MOUSE_CODON_USAGE = {
    'TTT': 17.2, 'TTC': 21.8, 'TTA': 6.7, 'TTG': 12.6,
    'CTT': 12.8, 'CTC': 20.2, 'CTA': 7.8, 'CTG': 40.3,
    'ATT': 15.1, 'ATC': 22.7, 'ATA': 7.1, 'ATG': 22.4,
    'GTT': 10.8, 'GTC': 15.8, 'GTA': 7.0, 'GTG': 29.1,
    'TAT': 12.0, 'TAC': 16.0, 'TAA': 0.7, 'TAG': 0.5,
    'CAT': 10.4, 'CAC': 15.3, 'CAA': 11.8, 'CAG': 34.6,
    'AAT': 16.4, 'AAC': 20.8, 'AAA': 22.4, 'AAG': 34.3,
    'GAT': 21.0, 'GAC': 26.4, 'GAA': 27.4, 'GAG': 40.8,
    'TCT': 15.4, 'TCC': 18.5, 'TCA': 11.4, 'TCG': 4.5,
    'CCT': 17.3, 'CCC': 20.0, 'CCA': 16.8, 'CCG': 7.0,
    'ACT': 12.8, 'ACC': 20.0, 'ACA': 14.8, 'ACG': 6.2,
    'GCT': 18.6, 'GCC': 28.5, 'GCA': 15.0, 'GCG': 7.5,
    'TGT': 10.5, 'TGC': 13.4, 'TGA': 1.3, 'TGG': 13.1,
    'CGT': 4.6, 'CGC': 10.8, 'CGA': 6.3, 'CGG': 11.5,
    'AGT': 11.9, 'AGC': 19.6, 'AGA': 11.4, 'AGG': 11.3,
    'GGT': 11.0, 'GGC': 23.3, 'GGA': 16.3, 'GGG': 16.6,
}

# Build amino acid to codons mapping
_aa_to_codons: Dict[str, List[str]] = {}
for _codon, _aa in GENETIC_CODE.items():
    if _aa == '*':
        continue
    if _aa not in _aa_to_codons:
        _aa_to_codons[_aa] = []
    _aa_to_codons[_aa].append(_codon)

# Build codon to (aa, degeneracy) mapping
CODON_TO_AA_WITH_DEGENERACY: Dict[str, Tuple[str, int]] = {}
for _aa, _codons in _aa_to_codons.items():
    for _codon in _codons:
        CODON_TO_AA_WITH_DEGENERACY[_codon] = (_aa, len(_codons))

def _build_aa_by_degeneracy() -> Dict[int, List[str]]:
    """Group amino acids by their codon degeneracy for ENC calculation."""
    result: Dict[int, List[str]] = {}
    for aa, codons in _aa_to_codons.items():
        deg = len(codons)
        if deg not in result:
            result[deg] = []
        result[deg].append(aa)
    return result

AA_BY_DEGENERACY = _build_aa_by_degeneracy()


def get_optimal_codons(codon_usage: dict) -> set:
    """Get the set of optimal codons from a codon usage table."""
    aa_to_codons_local: Dict[str, List[Tuple[str, float]]] = {}
    for codon, freq in codon_usage.items():
        if codon in CODON_TO_AA_WITH_DEGENERACY:
            aa, n_syn = CODON_TO_AA_WITH_DEGENERACY[codon]
            if n_syn > 1:
                if aa not in aa_to_codons_local:
                    aa_to_codons_local[aa] = []
                aa_to_codons_local[aa].append((codon, freq))
    
    optimal = set()
    for aa, codon_freqs in aa_to_codons_local.items():
        best_codon = max(codon_freqs, key=lambda x: x[1])[0]
        optimal.add(best_codon)
    return optimal


OPTIMAL_HUMAN = get_optimal_codons(HUMAN_CODON_USAGE)
OPTIMAL_MOUSE = get_optimal_codons(MOUSE_CODON_USAGE)


def build_relative_adaptiveness(codon_usage: dict) -> dict:
    """Compute w_i = frequency / max_frequency for synonymous codons."""
    aa_to_codons_local: Dict[str, List[str]] = {}
    for codon, aa in GENETIC_CODE.items():
        if aa == '*':
            continue
        if aa not in aa_to_codons_local:
            aa_to_codons_local[aa] = []
        aa_to_codons_local[aa].append(codon)
    
    w: Dict[str, float] = {}
    for aa, codons in aa_to_codons_local.items():
        freqs = [codon_usage.get(c, 0) for c in codons]
        max_freq = max(freqs) if freqs else 1
        for c, f in zip(codons, freqs):
            w[c] = f / max_freq if max_freq > 0 else 0
    return w


W_HUMAN = build_relative_adaptiveness(HUMAN_CODON_USAGE)
W_MOUSE = build_relative_adaptiveness(MOUSE_CODON_USAGE)


# =====================
# Metric Computation Functions
# =====================

def compute_gc_content(dna_seq: str) -> float:
    """Compute GC content percentage."""
    s = dna_seq.upper()
    gc_count = sum(1 for nt in s if nt in 'GC')
    return 100.0 * gc_count / max(1, len(s))


def compute_u_content(dna_seq: str) -> float:
    """Compute uridine (T in DNA) content percentage."""
    s = dna_seq.upper()
    t_count = sum(1 for nt in s if nt == 'T')
    return 100.0 * t_count / max(1, len(s))


def compute_cai(dna_seq: str, w_table: dict) -> float:
    """Compute Codon Adaptation Index."""
    dna_seq = dna_seq.upper()
    codons = [dna_seq[i:i+3] for i in range(0, len(dna_seq) - 2, 3)]
    
    valid_w = []
    for c in codons:
        if c in w_table and w_table[c] > 0:
            valid_w.append(w_table[c])
    
    if not valid_w:
        return np.nan
    
    log_sum = sum(np.log(w) for w in valid_w)
    return np.exp(log_sum / len(valid_w))


def compute_cbi(dna_seq: str, optimal_codons: set) -> float:
    """
    Compute Codon Bias Index (CBI).
    CBI = (N_opt - N_exp) / (N_tot - N_exp)
    Ranges from -1 to 1 (1 = max bias toward optimal codons).
    """
    dna_seq = dna_seq.upper()
    codons = [dna_seq[i:i+3] for i in range(0, len(dna_seq) - 2, 3)]
    
    n_opt = 0
    n_tot = 0
    n_exp_sum = 0.0
    
    for codon in codons:
        if codon not in CODON_TO_AA_WITH_DEGENERACY:
            continue
        aa, n_syn = CODON_TO_AA_WITH_DEGENERACY[codon]
        if n_syn == 1:
            continue
        n_tot += 1
        if codon in optimal_codons:
            n_opt += 1
        n_exp_sum += 1.0 / n_syn
    
    if n_tot == 0:
        return np.nan
    
    n_exp = n_exp_sum
    denominator = n_tot - n_exp
    if denominator == 0:
        return 0.0
    
    return (n_opt - n_exp) / denominator


def compute_enc(dna_seq: str) -> float:
    """
    Compute Effective Number of Codons (ENC).
    ENC = 2 + 9/F̄₂ + 1/F̄₃ + 5/F̄₄ + 3/F̄₆
    Ranges from 20 (extreme bias) to 61 (no bias).
    """
    dna_seq = dna_seq.upper()
    codons_list = [dna_seq[i:i+3] for i in range(0, len(dna_seq) - 2, 3)]
    codon_counts = Counter(codons_list)
    
    aa_F_values: Dict[str, float] = {}
    
    for aa, aa_codons in _aa_to_codons.items():
        if len(aa_codons) == 1:
            continue
        
        counts = [codon_counts.get(c, 0) for c in aa_codons]
        n = sum(counts)
        
        if n <= 1:
            continue
        
        sum_p_squared = sum((c / n) ** 2 for c in counts)
        F = (n * sum_p_squared - 1) / (n - 1)
        aa_F_values[aa] = F
    
    F_averages: Dict[int, float] = {}
    for deg, aas in AA_BY_DEGENERACY.items():
        F_vals = [aa_F_values[aa] for aa in aas if aa in aa_F_values]
        if F_vals:
            F_averages[deg] = np.mean(F_vals)
    
    if not F_averages:
        return np.nan
    
    coefficients = {2: 9, 3: 1, 4: 5, 6: 3}
    enc = 2.0
    for deg, coef in coefficients.items():
        if deg in F_averages and F_averages[deg] > 0:
            enc += coef / F_averages[deg]
        else:
            enc += coef
    return min(enc, 61.0)


def compute_mfe_vienna(dna_seq: str) -> float:
    """Compute actual MFE using ViennaRNA."""
    try:
        import RNA
        rna_seq = dna_seq.replace("T", "U").replace("t", "u")
        fc = RNA.fold_compound(rna_seq)
        _, mfe = fc.mfe()
        return float(mfe)
    except ImportError:
        return np.nan
    except Exception as e:
        logger.warning(f"MFE computation failed: {e}")
        return np.nan


def compute_mfe_batch(
    dna_seqs: List[str],
    use_parallel: bool = True,
    cache: Optional[dict] = None,
) -> Tuple[List[float], dict]:
    """Compute MFE for a batch of sequences using ViennaRNA with caching."""
    try:
        import RNA  # noqa: F401
    except ImportError:
        logger.warning("ViennaRNA not installed. Install with: pip install ViennaRNA")
        return [np.nan] * len(dna_seqs), cache or {}
    
    if cache is None:
        cache = {}
    
    mfe_values: List[Optional[float]] = []
    seqs_to_compute = []
    seq_indices = []
    
    for i, seq in enumerate(dna_seqs):
        seq_hash = hashlib.md5(seq.upper().encode()).hexdigest()
        if seq_hash in cache:
            mfe_values.append(cache[seq_hash])
        else:
            mfe_values.append(None)
            seqs_to_compute.append(seq)
            seq_indices.append(i)
    
    if seqs_to_compute:
        if use_parallel and len(seqs_to_compute) > 10:
            try:
                from joblib import Parallel, delayed
                computed_mfes = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(compute_mfe_vienna)(seq) for seq in seqs_to_compute
                )
            except ImportError:
                computed_mfes = [compute_mfe_vienna(seq) for seq in seqs_to_compute]
        else:
            computed_mfes = [compute_mfe_vienna(seq) for seq in seqs_to_compute]
        
        for idx, mfe in zip(seq_indices, computed_mfes):
            mfe_values[idx] = mfe
            seq_hash = hashlib.md5(dna_seqs[idx].upper().encode()).hexdigest()
            cache[seq_hash] = mfe
    
    return [v if v is not None else np.nan for v in mfe_values], cache


# =====================
# GA Configuration
# =====================

@dataclass
class GAConfig:
    """Configuration for GA hyperparameters."""
    population_size: int = 100
    generations: int = 50
    elite_fraction: float = 0.1
    tournament_size: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.02
    
    # Fitness weights
    weight_naturalness: float = 0.0  # Model naturalness weight (EnCodon or DeCodon)
    weight_mfe: float = 1.0          # MFE weight via ViennaRNA (higher = more stable)
    weight_te: float = 1.0           # TE weight (higher = better translation)
    weight_gc: float = 0.0           # GC content proximity weight
    weight_u: float = 0.0            # U content (minimize)
    weight_cai: float = 0.0          # CAI weight
    weight_cbi: float = 0.0          # CBI weight
    weight_enc: float = 0.0          # ENC weight (minimize for bias)
    
    target_gc_pct: float = 50.0
    
    def to_dict(self) -> dict:
        return asdict(self)


def detect_device(user_device: Optional[str]) -> torch.device:
    if user_device:
        return torch.device(user_device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def try_load_te_predictor(predictor_dir: str, device):
    """Attempt to load TE predictor with proper error reporting."""
    if not os.path.isdir(predictor_dir):
        logger.warning(f"TE predictor directory not found: {predictor_dir}")
        return None
    try:
        from notebooks.te_predictor import load_te_predictor  # type: ignore
        return load_te_predictor(predictor_dir, device)
    except FileNotFoundError as e:
        logger.warning(f"TE predictor files missing in {predictor_dir}: {e}")
        return None
    except ImportError as e:
        logger.warning(f"Failed to import TE predictor module: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load TE predictor from {predictor_dir}: {type(e).__name__}: {e}")
        return None


@torch.no_grad()
def compute_all_metrics(
    dna_seqs: List[str],
    model: Optional[ModelInference],
    context_length: int,
    te_predictor,
    organism: str = "human",
    organism_token: Optional[str] = None,
    batch_size: int = 32,
    mfe_cache: Optional[dict] = None,
) -> Tuple[List[Dict[str, Optional[float]]], dict]:
    """Compute naturalness, MFE, TE, and codon metrics for DNA sequences."""
    if not dna_seqs:
        return [], mfe_cache or {}
    if not dna_seqs[0]:
        raise ValueError("compute_all_metrics received empty DNA sequence")
    
    n = len(split_codons(dna_seqs[0]))
    w_table = W_MOUSE if organism == "mouse" else W_HUMAN
    optimal_codons = OPTIMAL_MOUSE if organism == "mouse" else OPTIMAL_HUMAN
    
    naturalness_scores = [np.nan] * len(dna_seqs)
    if model is not None:
        all_naturalness = []
        for i in range(0, len(dna_seqs), batch_size):
            batch_seqs = dna_seqs[i:i + batch_size]
            if isinstance(model, DecodonInference):
                batch_input = model.build_generation_batch(batch_seqs, n, context_length, organism_token=organism_token)
            else:
                batch_input = model.build_generation_batch(batch_seqs, n, context_length)
            batch_scores = model.predict_fitness(batch_input).fitness
            all_naturalness.extend(batch_scores.tolist())
        naturalness_scores = all_naturalness
    
    mfe_scores, mfe_cache = compute_mfe_batch(dna_seqs, cache=mfe_cache)
    
    te_scores: List[float] = [np.nan] * len(dna_seqs)
    if te_predictor is not None:
        try:
            te_scores = [float(t) for t in te_predictor.predict(dna_seqs, verbose=False)]
        except Exception as e:
            logger.warning(f"TE prediction failed: {e}")
    
    results = []
    for i, seq in enumerate(dna_seqs):
        results.append({
            "naturalness": float(naturalness_scores[i]) if not np.isnan(naturalness_scores[i]) else None,
            "mfe": float(mfe_scores[i]) if mfe_scores[i] is not None and not np.isnan(mfe_scores[i]) else None,
            "te": float(te_scores[i]) if not np.isnan(te_scores[i]) else None,
            "gc": compute_gc_content(seq),
            "u_pct": compute_u_content(seq),
            "cai": compute_cai(seq, w_table),
            "cbi": compute_cbi(seq, optimal_codons),
            "enc": compute_enc(seq),
        })
    
    return results, mfe_cache or {}


def _to_float_or_nan(val) -> float:
    """Convert value to float, treating None as NaN (but preserving 0.0)."""
    if val is None:
        return np.nan
    return float(val)


@torch.no_grad()
def compute_ga_fitness(
    metrics: List[Dict[str, Optional[float]]],
    config: GAConfig,
) -> np.ndarray:
    """Compute GA fitness as weighted combination of normalized metrics."""
    n = len(metrics)
    if n == 0:
        return np.array([])
    
    # Extract metrics, treating None as NaN (not as 0.0)
    naturalness_scores = np.array([_to_float_or_nan(m.get("naturalness")) for m in metrics])
    mfe_scores = np.array([_to_float_or_nan(m.get("mfe")) for m in metrics])
    te_scores = np.array([_to_float_or_nan(m.get("te")) for m in metrics])
    gc_scores = np.array([m.get("gc", 50.0) for m in metrics])
    u_scores = np.array([m.get("u_pct", 25.0) for m in metrics])
    cai_scores = np.array([_to_float_or_nan(m.get("cai")) for m in metrics])
    cbi_scores = np.array([_to_float_or_nan(m.get("cbi")) for m in metrics])
    enc_scores = np.array([_to_float_or_nan(m.get("enc")) for m in metrics])
    
    def normalize(arr: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
        valid_mask = ~np.isnan(arr)
        if valid_mask.sum() == 0:
            return np.zeros(len(arr))
        
        arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
        if arr_max == arr_min:
            result = np.ones(len(arr)) * 0.5
            result[~valid_mask] = 0.0
            return result
        
        normalized = (arr - arr_min) / (arr_max - arr_min)
        if not higher_is_better:
            normalized = 1.0 - normalized
        normalized[~valid_mask] = 0.0
        return normalized
    
    fitness = np.zeros(n)
    
    if config.weight_naturalness > 0:
        fitness += config.weight_naturalness * normalize(naturalness_scores, higher_is_better=True)
    if config.weight_mfe > 0:
        # More negative MFE = more thermodynamically stable
        fitness += config.weight_mfe * normalize(mfe_scores, higher_is_better=False)
    if config.weight_te > 0:
        fitness += config.weight_te * normalize(te_scores, higher_is_better=True)
    if config.weight_gc > 0:
        gc_proximity = 1.0 - np.abs(gc_scores - config.target_gc_pct) / 100.0
        fitness += config.weight_gc * gc_proximity
    if config.weight_u > 0:
        fitness += config.weight_u * normalize(u_scores, higher_is_better=False)
    if config.weight_cai > 0:
        fitness += config.weight_cai * normalize(cai_scores, higher_is_better=True)
    if config.weight_cbi > 0:
        fitness += config.weight_cbi * normalize(cbi_scores, higher_is_better=True)
    if config.weight_enc > 0:
        # Higher ENC = less codon bias = more like natural sequences
        fitness += config.weight_enc * normalize(enc_scores, higher_is_better=True)
    
    return fitness


def run_pipeline(
    aa_sequence: str,
    model_type: str,
    encodon_ckpt: str,
    decodon_ckpt: Optional[str],
    organism_tokens_file: Optional[str],
    te_predictor_dir: str,
    device: torch.device,
    # Decode mode
    decode_mode: str,
    mask_ratio: float,
    parallel_iterations: int,
    temperature: float,
    sample: bool,
    # Bidirectional diversity parameters (temperature annealing)
    temperature_start: float,
    temperature_end: float,
    # Beam search
    beam_width: int,
    context_length: int,
    bf16: bool,
    beam_top_k: Optional[int],
    batch_size: int,
    # GA
    ga_population_size: int,
    ga_generations: int,
    ga_elite_fraction: float,
    ga_tournament_size: int,
    ga_crossover_rate: float,
    ga_mutation_rate: float,
    # Weights
    weight_naturalness: float,
    weight_mfe: float,
    weight_te: float,
    weight_gc: float,
    weight_u: float,
    weight_cai: float,
    weight_cbi: float,
    weight_enc: float,
    target_gc_pct: float,
    # Additional options
    organism: str,
    mfe_cache_path: Optional[str],
) -> Tuple[str, float, str, float, List[str], List[float], list, List[Dict[str, Optional[float]]]]:
    """
    Run codon optimization: beam/bidirectional generation followed by GA refinement.
    
    Returns (best_beam_seq, best_beam_score, ga_best_seq, ga_best_fitness,
             final_population, final_population_fitness, beam_top_k_list, final_metrics).
    beam_top_k_list preserves original generation order for reproducibility.
    """
    if beam_width <= 0:
        raise ValueError(f"beam_width must be positive, got {beam_width}")
    if ga_population_size <= 0:
        raise ValueError(f"ga_population_size must be positive, got {ga_population_size}")
    if ga_generations < 0:
        raise ValueError(f"ga_generations must be non-negative, got {ga_generations}")
    if not (0.0 <= ga_elite_fraction <= 1.0):
        raise ValueError(f"ga_elite_fraction must be in [0, 1], got {ga_elite_fraction}")
    if ga_tournament_size <= 0:
        raise ValueError(f"ga_tournament_size must be positive, got {ga_tournament_size}")
    if sample and temperature <= 0:
        raise ValueError(f"temperature must be > 0 when sampling is enabled, got {temperature}")
    if not (0.0 < mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in (0, 1], got {mask_ratio}")
    
    if model_type == "decodon" and decode_mode == "bidirectional":
        raise ValueError(
            "Decodon (autoregressive model) cannot be used with bidirectional decode mode. "
            "Use --decode-mode=autoregressive with --model-type=decodon."
        )
    
    organism_token_map = {"human": "9606", "mouse": "10090"}
    organism_token = organism_token_map.get(organism)
    
    if model_type == "decodon":
        if decodon_ckpt is None:
            raise ValueError("--decodon-ckpt must be provided when using --model-type=decodon")
        if not os.path.exists(decodon_ckpt):
            raise FileNotFoundError(f"Decodon checkpoint not found: {decodon_ckpt}")
        
        logger.info(f"Loading Decodon model from: {decodon_ckpt}")
        if organism_tokens_file:
            logger.info(f"Using organism tokens file: {organism_tokens_file}")
        model: ModelInference = DecodonInference(
            model_path=decodon_ckpt,
            task_type=TaskTypes.NEXT_CODON_PREDICTION,
            organism_tokens_file=organism_tokens_file,
        )
    else:  # encodon
        if not os.path.exists(encodon_ckpt):
            raise FileNotFoundError(f"Encodon checkpoint not found: {encodon_ckpt}")
        
        logger.info(f"Loading Encodon model from: {encodon_ckpt}")
        model = EncodonInference(
            model_path=encodon_ckpt,
            task_type=TaskTypes.FITNESS_PREDICTION,
        )
    
    model.configure_model()
    model.to(device)
    model.eval()
    tokenizer: Tokenizer = model.tokenizer

    te_predictor = None
    if ga_generations > 0 and weight_te > 0:
        te_predictor = try_load_te_predictor(te_predictor_dir, device)
    
    if ga_generations > 0 and weight_mfe > 0:
        try:
            import RNA  # noqa: F401
            logger.info("Using ViennaRNA for MFE computation")
        except ImportError:
            logger.warning("ViennaRNA not installed. MFE will not be computed. Install with: pip install ViennaRNA")

    mfe_cache: Dict[str, float] = {}
    if mfe_cache_path and os.path.exists(mfe_cache_path):
        try:
            with open(mfe_cache_path, "rb") as f:
                mfe_cache = pickle.load(f)
            logger.info(f"Loaded MFE cache with {len(mfe_cache)} entries")
        except Exception as e:
            logger.warning(f"Could not load MFE cache: {e}")

    ga_config = GAConfig(
        population_size=ga_population_size,
        generations=ga_generations,
        elite_fraction=ga_elite_fraction,
        tournament_size=ga_tournament_size,
        crossover_rate=ga_crossover_rate,
        mutation_rate=ga_mutation_rate,
        weight_naturalness=weight_naturalness,
        weight_mfe=weight_mfe,
        weight_te=weight_te,
        weight_gc=weight_gc,
        weight_u=weight_u,
        weight_cai=weight_cai,
        weight_cbi=weight_cbi,
        weight_enc=weight_enc,
        target_gc_pct=target_gc_pct,
    )

    if ga_generations > 0:
        active_components = []
        if ga_config.weight_naturalness > 0:
            active_components.append(f"naturalness (w={ga_config.weight_naturalness})")
        if ga_config.weight_mfe > 0:
            active_components.append(f"MFE/ViennaRNA (w={ga_config.weight_mfe})")
        if ga_config.weight_te > 0:
            active_components.append(f"TE (w={ga_config.weight_te})")
        if ga_config.weight_gc > 0:
            active_components.append(f"GC (w={ga_config.weight_gc}, target={target_gc_pct}%)")
        if ga_config.weight_u > 0:
            active_components.append(f"U% (w={ga_config.weight_u})")
        if ga_config.weight_cai > 0:
            active_components.append(f"CAI (w={ga_config.weight_cai})")
        if ga_config.weight_cbi > 0:
            active_components.append(f"CBI (w={ga_config.weight_cbi})")
        if ga_config.weight_enc > 0:
            active_components.append(f"ENC (w={ga_config.weight_enc})")
        logger.info(f"Active fitness components: {', '.join(active_components) or 'none'}")

    if decode_mode == "autoregressive":
        first_aa = aa_sequence[0] if len(aa_sequence) > 0 else "M"

        if isinstance(model, DecodonInference):
            if organism_token:
                logger.info(f"Using Decodon generation with organism token: {organism_token}")
            else:
                logger.info("Using Decodon generation with synonymous codon constraints")
            seed_codon = tokenizer.aa_to_codon.get(first_aa, ["ATG"])[0]
            seed_codons = [organism_token + seed_codon] + [None] * max(0, len(aa_sequence) - 1)
            beam_candidates = model.generate_with_aa_constraints(
                amino_acid_sequence=aa_sequence,
                num_sequences=beam_width,
                organism_token=organism_token,
                temperature=temperature,
                sample=sample,
                seed_codons=seed_codons,
            )
        else:
            seed_codon = tokenizer.aa_to_codon.get(first_aa, ["ATG"])[0]
            seed_codons = [seed_codon] + [None] * max(0, len(aa_sequence) - 1)
            num_codons = len(aa_sequence)
            beam_candidates = model.generate_autoregressive(
                amino_acid_sequence=aa_sequence,
                context_length=context_length,
                num_sequences=beam_width,
                seed_codons=seed_codons,
                temperature=temperature,
                sample=sample,
                bf16=bf16,
                batch_size=batch_size,
            )
    elif decode_mode == "bidirectional":
        # Bidirectional uses mask-and-predict with temperature annealing
        assert isinstance(model, EncodonInference), "Bidirectional mode requires Encodon model"
        
        logger.info(f"Using bidirectional generation with temperature annealing: {temperature_start} -> {temperature_end}")
        initial_seqs, full_mask_argmax_indices, full_mask_sample_indices = generate_stratified_initial_sequences(
            amino_acid_sequence=aa_sequence,
            num_sequences=beam_width,
            num_full_mask_argmax=1,
            num_full_mask_sample=1,
        )        
        beam_candidates = model.generate_bidirectional(
            dna_seqs=initial_seqs,
            amino_acid_sequence=aa_sequence,
            context_length=context_length,
            mask_ratio=mask_ratio,
            num_iterations=parallel_iterations,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            bf16=bf16,
            full_mask_argmax_indices=full_mask_argmax_indices,
            full_mask_sample_indices=full_mask_sample_indices,
            batch_size=batch_size,
        )
        
    else:
        raise ValueError(f"Invalid decode mode: {decode_mode}")

    if not beam_candidates:
        raise RuntimeError(
            f"Generation produced no candidates. Check model compatibility and input sequence."
        )
    
    for i, dna in enumerate(beam_candidates):
        decoded_aa = "".join(amino_acids_from_dna(dna))
        if decoded_aa != aa_sequence:
            logger.warning(
                f"Candidate {i} decodes to different AA sequence: "
                f"expected {len(aa_sequence)}aa, got {len(decoded_aa)}aa"
            )

    # Score candidates in batches to avoid OOM
    num_codons = len(aa_sequence)
    all_scores = []
    for i in range(0, len(beam_candidates), batch_size):
        batch_seqs = beam_candidates[i:i + batch_size]
        if isinstance(model, DecodonInference):
            batch_input = model.build_generation_batch(batch_seqs, num_codons, context_length, organism_token=organism_token)
        else:
            batch_input = model.build_generation_batch(batch_seqs, num_codons, context_length)
        batch_scores = model.predict_fitness(batch_input).fitness.tolist()
        all_scores.extend(batch_scores)
    model_scores = all_scores
    
    # Preserve generation order for reproducibility with fixed seed
    beam_candidates_with_scores = [(float(s), dna) for s, dna in zip(model_scores, beam_candidates)]
    
    best_idx = int(np.argmax(model_scores))
    best_beam_score = float(model_scores[best_idx])
    best_beam_seq = beam_candidates[best_idx]

    k = beam_top_k if (beam_top_k is not None and beam_top_k > 0) else len(beam_candidates_with_scores)
    k = min(k, len(beam_candidates_with_scores))
    beam_top_k_list = beam_candidates_with_scores[:k]

    population: List[str] = [seq for _, seq in beam_top_k_list]
    while len(population) < ga_population_size:
        population.append(random_dna_for_protein(aa_sequence))

    elite_count = max(1, int(ga_elite_fraction * ga_population_size))

    metrics, mfe_cache = compute_all_metrics(
        dna_seqs=population,
        model=model,
        context_length=context_length,
        te_predictor=te_predictor,
        organism=organism,
        organism_token=organism_token,
        batch_size=batch_size,
        mfe_cache=mfe_cache,
    )

    for gen in tqdm(range(ga_generations), desc="GA optimization", disable=ga_generations == 0):
        fitnesses = compute_ga_fitness(metrics, ga_config)
        
        if gen > 0 and gen % 10 == 0:
            unique_seqs = len(set(population))
            diversity = unique_seqs / len(population)
            logger.info(f"Gen {gen}: best={np.max(fitnesses):.4f}, mean={np.mean(fitnesses):.4f}, diversity={diversity:.2%}")
        
        ranked = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
        elites = ranked[:elite_count]
        effective_tournament_size = min(ga_tournament_size, len(ranked))
        mating_pool = tournament_select(
            ranked, k=ga_population_size - elite_count, tournament_size=effective_tournament_size
        )
        next_pop: List[str] = [seq for _, seq in elites]
        i = 0
        while len(next_pop) < ga_population_size:
            if len(mating_pool) < 2:
                next_pop.append(random_dna_for_protein(aa_sequence))
                continue
            
            p1_idx = i % len(mating_pool)
            p2_idx = (i + 1) % len(mating_pool)
            _, p1 = mating_pool[p1_idx]
            _, p2 = mating_pool[p2_idx]
            i += 2
            
            if random.random() < ga_crossover_rate:
                c1, c2 = crossover_codons(p1, p2)
            else:
                c1, c2 = p1, p2
            c1 = mutate_codons(c1, aa_sequence, ga_mutation_rate)
            c2 = mutate_codons(c2, aa_sequence, ga_mutation_rate)
            if len(next_pop) < ga_population_size:
                next_pop.append(c1)
            if len(next_pop) < ga_population_size:
                next_pop.append(c2)
        population = next_pop
        
        metrics, mfe_cache = compute_all_metrics(
            dna_seqs=population,
            model=model,
            context_length=context_length,
            te_predictor=te_predictor,
            organism=organism,
            organism_token=organism_token,
            batch_size=batch_size,
            mfe_cache=mfe_cache,
        )

    final_fit = compute_ga_fitness(metrics, ga_config)
    best_idx = int(np.argmax(final_fit))
    ga_best_fit = float(final_fit[best_idx])
    ga_best_seq = population[best_idx]
    final_fit_list = [float(f) for f in final_fit]

    if mfe_cache_path:
        try:
            with open(mfe_cache_path, "wb") as f:
                pickle.dump(mfe_cache, f)
            logger.info(f"Saved MFE cache with {len(mfe_cache)} entries")
        except Exception as e:
            logger.warning(f"Could not save MFE cache: {e}")

    return best_beam_seq, float(best_beam_score), ga_best_seq, ga_best_fit, population, final_fit_list, beam_top_k_list, metrics


def validate_aa_sequence(aa_seq: str) -> str:
    """Validate and clean amino acid sequence, removing stop codons and whitespace."""
    aa_seq = aa_seq.upper().replace("*", "").replace(" ", "").replace("\n", "").replace("\t", "")
    
    if not aa_seq:
        raise ValueError("Empty amino acid sequence after preprocessing.")
    
    invalid_chars = set(aa_seq) - VALID_AMINO_ACIDS
    if invalid_chars:
        raise ValueError(
            f"Invalid amino acid characters found: {sorted(invalid_chars)}. "
            f"Valid amino acids are: {''.join(sorted(VALID_AMINO_ACIDS))}"
        )
    
    return aa_seq


def read_aa_sequence(args: argparse.Namespace) -> str:
    """Read amino acid sequence from --aa or --aa-file (supports FASTA format)."""
    if args.aa and args.aa_file:
        raise ValueError("Provide either --aa or --aa-file, not both.")
    
    if args.aa_file:
        aa_file_path = Path(args.aa_file)
        if not aa_file_path.exists():
            raise FileNotFoundError(f"AA sequence file not found: {args.aa_file}")
        
        text = aa_file_path.read_text().strip()
        if text.startswith(">"):
            lines = [ln.strip() for ln in text.splitlines() if ln and not ln.startswith(">")]
            seq = "".join(lines)
        else:
            seq = text
        return validate_aa_sequence(seq)
    
    if args.aa:
        return validate_aa_sequence(args.aa)
    
    raise ValueError("Amino acid sequence not provided. Use --aa or --aa-file.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Codon optimization via beam search + GA fitness optimization."
    )
    p.add_argument("--aa", type=str, help="Amino acid sequence (string).")
    p.add_argument("--aa-file", type=str, help="Path to file containing AA sequence (raw or FASTA).")

    p.add_argument(
        "--model-type",
        type=str,
        choices=["encodon", "decodon"],
        default="encodon",
        help="Model type to use: 'encodon' (bidirectional) or 'decodon' (autoregressive). Default: encodon",
    )
    p.add_argument(
        "--encodon-ckpt",
        type=str,
        default="/data/checkpoints/release_ckpts/release_80m/checkpoints/last.ckpt",
        help="Path to Encodon checkpoint (used when --model-type=encodon).",
    )
    p.add_argument(
        "--decodon-ckpt",
        type=str,
        default=None,
        help="Path to Decodon checkpoint (used when --model-type=decodon).",
    )
    p.add_argument(
        "--organism-tokens-file",
        type=str,
        default=None,
        help="Path to organism tokens file for Decodon (e.g., /data/nopathogen_organism_tokens.txt).",
    )
    p.add_argument(
        "--te-predictor-dir",
        type=str,
        default="/data/codon_optimization/TE/predictor",
        help="Path to TE predictor directory.",
    )

    p.add_argument("--device", type=str, default=None, help="Torch device (e.g., cuda:0 or cpu). Default: auto-detect")
    p.add_argument("--no-bf16", action="store_true", help="Disable bfloat16 precision.")

    p.add_argument(
        "--decode-mode",
        type=str,
        choices=["autoregressive", "bidirectional"],
        default="autoregressive",
        help="Decoding mode: 'autoregressive' predicts one codon at a time, 'bidirectional' uses mask-and-predict refinement.",
    )
    p.add_argument(
        "--mask-ratio",
        type=float,
        default=0.2,
        help="Fraction of positions to mask per iteration in parallel mode (default: 0.2).",
    )
    p.add_argument(
        "--parallel-iterations",
        type=int,
        default=20,
        help="Number of mask-and-predict iterations in parallel mode (default: 10).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation (default: 1.0).",
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help="Sample from distribution instead of taking argmax.",
    )
    
    # Bidirectional diversity parameters (temperature annealing)
    p.add_argument(
        "--temperature-start",
        type=float,
        default=1.2,
        help="Starting temperature for annealing in diverse mode (default: 1.2).",
    )
    p.add_argument(
        "--temperature-end",
        type=float,
        default=0.5,
        help="Ending temperature for annealing in diverse mode (default: 0.5).",
    )

    p.add_argument("--beam-width", type=int, default=10, help="Beam width for sequence generation.")
    p.add_argument("--context-length", type=int, default=2048, help="Model context length.")
    p.add_argument("--beam-top-k", type=int, default=10, help="Top-K beam sequences to seed GA.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")

    p.add_argument("--ga-population-size", type=int, default=10, help="GA population size.")
    p.add_argument("--ga-generations", type=int, default=10, help="Number of GA generations.")
    p.add_argument("--ga-elite-fraction", type=float, default=0.1, help="Fraction of elites to preserve.")
    p.add_argument("--ga-tournament-size", type=int, default=3, help="Tournament selection size.")
    p.add_argument("--ga-crossover-rate", type=float, default=0.7, help="Crossover probability.")
    p.add_argument("--ga-mutation-rate", type=float, default=0.02, help="Per-codon mutation probability.")

    p.add_argument("--weight-naturalness", type=float, default=0.0,
                   help="Weight for model naturalness in fitness (default: 0.0).")
    p.add_argument("--weight-mfe", type=float, default=0.5,
                   help="Weight for MFE (stability) in fitness (default: 0.5).")
    p.add_argument("--weight-te", type=float, default=0.5,
                   help="Weight for TE (translation efficiency) in fitness (default: 0.5).")
    p.add_argument("--weight-gc", type=float, default=0.0,
                   help="Weight for GC content proximity in fitness (default: 0.0).")
    p.add_argument("--weight-u", type=float, default=0.0,
                   help="Weight for U content (minimize) in fitness (default: 0.0).")
    p.add_argument("--weight-cai", type=float, default=0.0,
                   help="Weight for CAI in fitness (default: 0.0).")
    p.add_argument("--weight-cbi", type=float, default=0.0,
                   help="Weight for CBI in fitness (default: 0.0).")
    p.add_argument("--weight-enc", type=float, default=0.0,
                   help="Weight for ENC in fitness (default: 0.0).")
    p.add_argument("--target-gc-pct", type=float, default=50.0,
                   help="Target GC percentage for GC fitness component.")

    p.add_argument("--organism", type=str, default="human", choices=["human", "mouse"],
                   help="Organism for codon usage tables.")
    p.add_argument("--mfe-cache-path", type=str, default=None,
                   help="Path to pickle cache for ViennaRNA MFE results.")

    p.add_argument("--out-json", type=str, default=None, help="Path to save JSON results.")
    p.add_argument("--print-top", type=int, default=3, help="Number of top beam candidates to print.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    device = detect_device(args.device)
    bf16 = not args.no_bf16
    aa_seq = read_aa_sequence(args)

    (
        best_beam_seq,
        best_beam_score,
        ga_best_seq,
        ga_best_fit,
        population,
        population_fitness,
        beam_top_k_list,
        final_metrics,
    ) = run_pipeline(
        aa_sequence=aa_seq,
        model_type=args.model_type,
        encodon_ckpt=args.encodon_ckpt,
        decodon_ckpt=args.decodon_ckpt,
        organism_tokens_file=args.organism_tokens_file,
        te_predictor_dir=args.te_predictor_dir,
        device=device,
        # Decode mode
        decode_mode=args.decode_mode,
        mask_ratio=args.mask_ratio,
        parallel_iterations=args.parallel_iterations,
        temperature=args.temperature,
        sample=args.sample,
        # Bidirectional diversity (temperature annealing)
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        # Beam
        beam_width=args.beam_width,
        context_length=args.context_length,
        bf16=bf16,
        beam_top_k=args.beam_top_k,
        batch_size=args.batch_size,
        # GA
        ga_population_size=args.ga_population_size,
        ga_generations=args.ga_generations,
        ga_elite_fraction=args.ga_elite_fraction,
        ga_tournament_size=args.ga_tournament_size,
        ga_crossover_rate=args.ga_crossover_rate,
        ga_mutation_rate=args.ga_mutation_rate,
        # Weights
        weight_naturalness=args.weight_naturalness,
        weight_mfe=args.weight_mfe,
        weight_te=args.weight_te,
        weight_gc=args.weight_gc,
        weight_u=args.weight_u,
        weight_cai=args.weight_cai,
        weight_cbi=args.weight_cbi,
        weight_enc=args.weight_enc,
        target_gc_pct=args.target_gc_pct,
        # Additional options
        organism=args.organism,
        mfe_cache_path=args.mfe_cache_path,
    )

    num_codons = len(aa_seq)
    print("== Codon Optimization ==")
    print(f"Model: {args.model_type} | AA length: {num_codons} aa | Device: {device} | bf16: {bf16}")
    if args.decode_mode == "bidirectional":
        print(f"Decode mode: {args.decode_mode} (mask_ratio={args.mask_ratio}, temp {args.temperature_start}->{args.temperature_end})")
    else:
        print(f"Decode mode: {args.decode_mode}")
    print(f"Best beam score: {best_beam_score:.4f} (avg per codon: {best_beam_score/max(1,num_codons):.4f})")
    print(f"Best GA fitness: {ga_best_fit:.4f}")

    if args.print_top and args.print_top > 0:
        print(f"\nFirst {min(args.print_top, len(beam_top_k_list))} beam candidates (generation order):")
        for i, (score, seq) in enumerate(beam_top_k_list[:args.print_top], start=1):
            print(f"  {i}. Score: {score:.4f} | len={len(seq)}bp")

    print("\nGA best sequence (first 180bp):")
    print(ga_best_seq[:180] + ("..." if len(ga_best_seq) > 180 else ""))

    config_obj = {
        "model_type": args.model_type,
        "decode_mode": args.decode_mode,
        "mask_ratio": args.mask_ratio,
        "parallel_iterations": args.parallel_iterations,
        "temperature": args.temperature,
        "sample": args.sample,
        "temperature_start": args.temperature_start,
        "temperature_end": args.temperature_end,
        "beam_width": args.beam_width,
        "context_length": args.context_length,
        "bf16": bf16,
        "beam_top_k": args.beam_top_k,
        "batch_size": args.batch_size,
        "ga_population_size": args.ga_population_size,
        "ga_generations": args.ga_generations,
        "ga_elite_fraction": args.ga_elite_fraction,
        "ga_tournament_size": args.ga_tournament_size,
        "ga_crossover_rate": args.ga_crossover_rate,
        "ga_mutation_rate": args.ga_mutation_rate,
        # Fitness weights
        "weight_naturalness": args.weight_naturalness,
        "weight_mfe": args.weight_mfe,
        "weight_te": args.weight_te,
        "weight_gc": args.weight_gc,
        "weight_u": args.weight_u,
        "weight_cai": args.weight_cai,
        "weight_cbi": args.weight_cbi,
        "weight_enc": args.weight_enc,
        "target_gc_pct": args.target_gc_pct,
        "organism": args.organism,
        "mfe_cache_path": args.mfe_cache_path,
        "encodon_ckpt": args.encodon_ckpt,
        "decodon_ckpt": args.decodon_ckpt,
        "organism_tokens_file": args.organism_tokens_file,
        "te_predictor_dir": args.te_predictor_dir,
        "seed": args.seed,
    }
    if args.out_json:
        top_k_for_json = [
            {"index": i, "score": float(score), "sequence": seq}
            for i, (score, seq) in enumerate(beam_top_k_list)
        ]
        ga_population_for_json = [
            {"index": i, "fitness": float(fit), "sequence": seq, "metrics": seq_metrics}
            for i, (fit, seq, seq_metrics) in enumerate(zip(population_fitness, population, final_metrics))
        ]
        payload_path = Path(args.out_json)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_payload = {
            "protein_sequence": aa_seq,
            "config": config_obj,
            "beam_best": {"sequence": best_beam_seq, "score": float(best_beam_score)},
            "beam_candidates": top_k_for_json,
            "ga_best": {"sequence": ga_best_seq, "fitness": float(ga_best_fit)},
            "ga_population": ga_population_for_json,
        }
        
        with open(payload_path, "w") as f:
            json.dump(output_payload, f, indent=2)
        
        logger.info(f"Saved results to: {payload_path}")

if __name__ == "__main__":
    main()