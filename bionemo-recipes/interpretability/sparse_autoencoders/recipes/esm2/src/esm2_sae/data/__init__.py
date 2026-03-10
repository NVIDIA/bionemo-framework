"""Protein data loading and processing."""

from .fasta import read_fasta
from .uniprot import download_swissprot, download_uniref50
from .annotations import (
    download_annotated_proteins,
    load_annotations_tsv,
    proteins_to_concept_labels,
)
from .dataset import ProteinDataset
from .types import ProteinRecord

__all__ = [
    'read_fasta',
    'download_swissprot',
    'download_uniref50',
    'download_annotated_proteins',
    'load_annotations_tsv',
    'proteins_to_concept_labels',
    'ProteinDataset',
    'ProteinRecord',
]
