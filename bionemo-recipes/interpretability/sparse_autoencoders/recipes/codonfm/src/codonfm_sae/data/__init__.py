"""CodonFM data loading and processing."""

from .csv_loader import read_codon_csv
from .types import CodonRecord

__all__ = ["read_codon_csv", "CodonRecord"]
