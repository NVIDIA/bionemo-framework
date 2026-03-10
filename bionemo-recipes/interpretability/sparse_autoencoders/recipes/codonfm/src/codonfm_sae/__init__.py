"""CodonFM SAE Recipe: Sparse Autoencoders for CodonFM Codon Language Models."""

from .data import read_codon_csv, CodonRecord

__version__ = "0.1.0"

__all__ = [
    "read_codon_csv",
    "CodonRecord",
]
