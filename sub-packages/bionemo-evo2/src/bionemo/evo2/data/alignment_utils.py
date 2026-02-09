import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta

def compute_polio_alignment(sequence: str, polio_ref_path: str) -> float:
    """Compute alignment score between sequence and poliovirus reference.

    Returns:
        float: Alignment percentage (0-100)
    """
    # Load poliovirus reference
    polio_fasta = fasta.FastaFile.read(polio_ref_path)
    polio_seq = list(polio_fasta.values())[0]

    # Convert to biotite sequences
    from biotite.sequence import NucleotideSequence
    seq1 = NucleotideSequence(sequence)
    seq2 = NucleotideSequence(polio_seq)

    # Perform alignment (you may want to use a more sophisticated method)
    # This is a simplified example - you might want to use pairwise2 or similar
    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    alignment = align.align_optimal(seq1, seq2, matrix)[0]

    # Calculate identity percentage
    matches = sum(a == b for a, b in zip(alignment[0], alignment[1]) if a != '-' and b != '-')
    total = sum(1 for a, b in zip(alignment[0], alignment[1]) if a != '-' or b != '-')
    
    return (matches / total) * 100 if total > 0 else 0.0

def get_alignment_token(score: float) -> str:
    """Convert alignment score to token string.
    
    Returns:
        str: Token string like "+~", "+^", "+#", "+$", or "+!"
    """
    if score >= 95:
        return "+~"
    elif score >= 80:
        return "+^"
    elif score >= 70:
        return "+#"
    elif score >= 50:
        return "+$"
    else:
        return "+!"