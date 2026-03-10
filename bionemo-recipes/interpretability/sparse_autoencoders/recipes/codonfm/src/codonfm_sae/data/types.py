from dataclasses import dataclass


@dataclass
class CodonRecord:
    """Container for a codon DNA sequence record."""

    id: str
    sequence: str  # raw DNA string (e.g., "ATGCGT...")

    @property
    def num_codons(self) -> int:
        return len(self.sequence) // 3
