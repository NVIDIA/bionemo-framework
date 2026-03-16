"""CodonFM data loading and processing."""

from .csv_loader import read_codon_csv
from .types import CodonRecord
from .uniprot import resolve_gene_to_alphafold

__all__ = ["read_codon_csv", "CodonRecord", "resolve_gene_to_alphafold"]
