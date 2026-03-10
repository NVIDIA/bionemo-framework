from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import random

from src.tokenizer import Tokenizer
from src.tokenizer.mappings import CODON_TABLE, AA_TABLE


def split_codons(dna_seq: str) -> List[str]:
    return [dna_seq[i:i + 3] for i in range(0, len(dna_seq), 3) if i + 3 <= len(dna_seq)]


def join_codons(codons: List[str]) -> str:
    return "".join(codons)


def amino_acids_from_dna(dna_seq: str) -> List[str]:
    codons = split_codons(dna_seq)
    result = []
    for c in codons:
        aa = CODON_TABLE.DNA.get(c)
        if aa is None:
            raise ValueError(f"Invalid codon: '{c}'")
        result.append(aa)
    return result


def random_dna_for_protein(amino_acid_sequence: str) -> str:
    """Generate a random DNA sequence for a protein using random synonymous codons."""
    codons: List[str] = []
    for aa in amino_acid_sequence:
        choices = AA_TABLE.DNA.get(aa)
        if not choices:
            raise ValueError(f"Unknown amino acid: '{aa}'")
        codons.append(random.choice(choices))
    return join_codons(codons)


# GC content of each codon (precomputed)
CODON_GC_COUNT = {
    codon: sum(1 for nt in codon if nt in 'GC')
    for codon in CODON_TABLE.DNA.keys()
}

# Human codon usage table (per 1000 codons)
# Reference: https://www.kazusa.or.jp/codon/ - taxid 9606
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

# Build best codon per amino acid (CAI-optimized)
_BEST_CODON_PER_AA: Dict[str, str] = {}
for _codon, _usage in HUMAN_CODON_USAGE.items():
    _aa = CODON_TABLE.DNA.get(_codon)
    if _aa and _aa != '*':
        if _aa not in _BEST_CODON_PER_AA or _usage > HUMAN_CODON_USAGE.get(_BEST_CODON_PER_AA[_aa], 0):
            _BEST_CODON_PER_AA[_aa] = _codon


def gc_biased_dna_for_protein(amino_acid_sequence: str, prefer_high_gc: bool = True) -> str:
    """
    Generate DNA sequence biased towards high or low GC content.
    
    Args:
        amino_acid_sequence: Target amino acid sequence.
        prefer_high_gc: If True, prefer high-GC codons; if False, prefer low-GC codons.
        
    Returns:
        DNA sequence with GC bias.
    """
    codons: List[str] = []
    for aa in amino_acid_sequence:
        choices = AA_TABLE.DNA.get(aa)
        if not choices:
            raise ValueError(f"Unknown amino acid: '{aa}'")
        
        if len(choices) == 1:
            codons.append(choices[0])
        else:
            # Sort by GC content
            sorted_choices = sorted(choices, key=lambda c: CODON_GC_COUNT.get(c, 0), reverse=prefer_high_gc)
            # Pick the most biased codon (with some randomization among top choices)
            # Take top half to add some variety
            top_choices = sorted_choices[:max(1, len(sorted_choices) // 2)]
            codons.append(random.choice(top_choices))
    
    return join_codons(codons)


def alternating_gc_dna_for_protein(amino_acid_sequence: str) -> str:
    """
    Generate DNA sequence with alternating GC preference (high-low-high-low...).
    Creates sequences with varied local GC content.
    """
    codons: List[str] = []
    for aa in amino_acid_sequence:
        choices = AA_TABLE.DNA.get(aa)
        if not choices:
            raise ValueError(f"Unknown amino acid: '{aa}'")
        
        if len(choices) == 1:
            codons.append(choices[0])
        else:
            # Use output position to maintain consistent alternation
            prefer_high = (len(codons) % 2 == 0)
            sorted_choices = sorted(choices, key=lambda c: CODON_GC_COUNT.get(c, 0), reverse=prefer_high)
            codons.append(sorted_choices[0])
    
    return join_codons(codons)


def rare_codon_dna_for_protein(amino_acid_sequence: str, codon_usage: Optional[Dict[str, float]] = None) -> str:
    """
    Generate DNA sequence preferring rare/uncommon codons.
    Useful for creating diverse initial sequences.
    
    Args:
        amino_acid_sequence: Target amino acid sequence.
        codon_usage: Optional codon usage frequency table. If None, falls back to random selection.
        
    Returns:
        DNA sequence with rare codon preference (when usage table provided).
    """
    codons: List[str] = []
    for aa in amino_acid_sequence:
        choices = AA_TABLE.DNA.get(aa)
        if not choices:
            raise ValueError(f"Unknown amino acid: '{aa}'")
        
        if len(choices) == 1 or codon_usage is None:
            # No usage data available - fall back to random selection
            codons.append(random.choice(choices))
        else:
            # Sort by usage frequency (ascending - rare first)
            sorted_choices = sorted(choices, key=lambda c: codon_usage.get(c, 0.5))
            # Pick from rarer half
            rare_choices = sorted_choices[:max(1, len(sorted_choices) // 2)]
            codons.append(random.choice(rare_choices))
    
    return join_codons(codons)


def cai_optimized_dna_for_protein(amino_acid_sequence: str) -> str:
    """
    Generate DNA sequence using the highest-frequency codon for each amino acid.
    This produces a CAI-optimized sequence (maximum CAI for human).
    
    Args:
        amino_acid_sequence: Target amino acid sequence.
        
    Returns:
        DNA sequence using the most frequent human codons (CAI-optimized).
    """
    codons: List[str] = []
    for aa in amino_acid_sequence:
        best_codon = _BEST_CODON_PER_AA.get(aa)
        if best_codon:
            codons.append(best_codon)
        else:
            # Fallback for unknown amino acids
            choices = AA_TABLE.DNA.get(aa)
            if not choices:
                raise ValueError(f"Unknown amino acid: '{aa}'")
            codons.append(choices[0])
    
    return join_codons(codons)


def cai_biased_dna_for_protein(amino_acid_sequence: str, codon_usage: Optional[Dict[str, float]] = None) -> str:
    """
    Generate DNA sequence biased towards high-CAI codons with some randomization.
    Unlike cai_optimized_dna_for_protein, this adds variety by sampling from top codons.
    
    Args:
        amino_acid_sequence: Target amino acid sequence.
        codon_usage: Optional codon usage frequency table. If None, uses human codon usage.
        
    Returns:
        DNA sequence with CAI bias (sampling from top codons per amino acid).
    """
    usage = codon_usage if codon_usage is not None else HUMAN_CODON_USAGE
    codons: List[str] = []
    
    for aa in amino_acid_sequence:
        choices = AA_TABLE.DNA.get(aa)
        if not choices:
            raise ValueError(f"Unknown amino acid: '{aa}'")
        
        if len(choices) == 1:
            codons.append(choices[0])
        else:
            # Sort by usage frequency (descending - most frequent first)
            sorted_choices = sorted(choices, key=lambda c: usage.get(c, 0.0), reverse=True)
            # Pick from top half to add variety while maintaining CAI bias
            top_choices = sorted_choices[:max(1, len(sorted_choices) // 2)]
            codons.append(random.choice(top_choices))
    
    return join_codons(codons)


def generate_stratified_initial_sequences(
    amino_acid_sequence: str,
    num_sequences: int,
    num_full_mask_argmax: int = 1,  # How many fully-masked sequences with argmax decoding
    num_full_mask_sample: int = 1,  # How many fully-masked sequences with sampling
) -> Tuple[List[str], List[int], List[int]]:
    """
    Generate stratified initial sequences with diverse biases.
    
    Args:
        amino_acid_sequence: The amino acid sequence to generate DNA for
        num_sequences: Total number of sequences to generate
        num_full_mask_argmax: Number of fully-masked sequences decoded with argmax (greedy)
        num_full_mask_sample: Number of fully-masked sequences decoded with sampling (diverse)
    
    Returns:
        Tuple of:
        - initial_seqs: List of initial DNA sequences
        - full_mask_argmax_indices: Indices of sequences that should be fully masked with argmax
        - full_mask_sample_indices: Indices of sequences that should be fully masked with sampling
    """
    initial_seqs: List[str] = []
    full_mask_argmax_indices: List[int] = []
    full_mask_sample_indices: List[int] = []
    
    strategies = []
    
    # Reserve slots for fully-masked generation with argmax decoding
    for _ in range(num_full_mask_argmax):
        strategies.append(('full_mask_argmax', None))
    
    # Reserve slots for fully-masked generation with sampling
    for _ in range(num_full_mask_sample):
        strategies.append(('full_mask_sample', None))
    
    # Fill remaining slots with diverse initializations
    remaining = num_sequences - len(strategies)
    if remaining > 0:
        diverse_strategies = [
            ('cai_optimized', cai_optimized_dna_for_protein),
            ('cai_biased', cai_biased_dna_for_protein),
            ('high_gc', lambda aa: gc_biased_dna_for_protein(aa, prefer_high_gc=True)),
            ('low_gc', lambda aa: gc_biased_dna_for_protein(aa, prefer_high_gc=False)),
            ('alternating', alternating_gc_dna_for_protein),
            ('random', random_dna_for_protein),
        ]
        
        # Cycle through strategies to fill remaining slots
        for i in range(remaining):
            strategy_name, strategy_fn = diverse_strategies[i % len(diverse_strategies)]
            strategies.append((strategy_name, strategy_fn))
    
    # Generate sequences
    for idx, (strategy_name, strategy_fn) in enumerate(strategies):
        if strategy_name == 'full_mask_argmax':
            # Placeholder - will be fully masked and generated with argmax
            initial_seqs.append(random_dna_for_protein(amino_acid_sequence))
            full_mask_argmax_indices.append(idx)
        elif strategy_name == 'full_mask_sample':
            # Placeholder - will be fully masked and generated with sampling
            initial_seqs.append(random_dna_for_protein(amino_acid_sequence))
            full_mask_sample_indices.append(idx)
        else:
            initial_seqs.append(strategy_fn(amino_acid_sequence))
    
    return initial_seqs, full_mask_argmax_indices, full_mask_sample_indices


def build_codon_to_token_id(tokenizer: Tokenizer) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for codon in CODON_TABLE.DNA.keys():
        tokens = tokenizer.encode(codon, add_special_tokens=False)
        if len(tokens) == 1:
            mapping[codon] = tokens[0]
    return mapping


def tournament_select(population: List[Tuple[float, str]], k: int, tournament_size: int) -> List[Tuple[float, str]]:
    if not population:
        raise ValueError("Cannot select from empty population")
    if tournament_size < 1:
        raise ValueError("tournament_size must be >= 1")
    
    selected: List[Tuple[float, str]] = []
    n = len(population)
    for _ in range(k):
        contestants = [population[random.randrange(n)] for _ in range(tournament_size)]
        winner = max(contestants, key=lambda x: x[0])
        selected.append(winner)
    return selected


def crossover_codons(a: str, b: str) -> Tuple[str, str]:
    codons_a = split_codons(a)
    codons_b = split_codons(b)
    if len(codons_a) != len(codons_b) or len(codons_a) < 2:
        return a, b
    point = random.randint(1, len(codons_a) - 1)
    child1 = join_codons(codons_a[:point] + codons_b[point:])
    child2 = join_codons(codons_b[:point] + codons_a[point:])
    return child1, child2


def mutate_codons(dna_seq: str, amino_acid_sequence: str, mutation_rate: float) -> str:
    codons = split_codons(dna_seq)
    if len(codons) != len(amino_acid_sequence):
        raise ValueError(
            f"Length mismatch: {len(codons)} codons vs {len(amino_acid_sequence)} amino acids"
        )
    
    for i, aa in enumerate(amino_acid_sequence):
        if random.random() < mutation_rate:
            synonyms = AA_TABLE.DNA.get(aa, [])
            if len(synonyms) > 1:
                alternatives = [c for c in synonyms if c != codons[i]]
                codons[i] = random.choice(alternatives)
    return join_codons(codons)
