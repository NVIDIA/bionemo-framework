from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class CodonRecord:
    """Container for a codon DNA sequence record."""

    id: str
    sequence: str  # raw DNA string (e.g., "ATGCGT...")
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_codons(self) -> int:
        return len(self.sequence) // 3
