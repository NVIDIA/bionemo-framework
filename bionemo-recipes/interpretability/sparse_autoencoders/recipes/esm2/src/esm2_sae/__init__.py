"""
ESM2 SAE Recipe: Sparse Autoencoders for ESM2 Protein Language Models

This package provides ESM2-specific implementations for training and evaluating
Sparse Autoencoders on protein embeddings.

Main Components:
    - data: Protein datasets (FASTA, SwissProt, annotations)
    - eval: Biology-specific evaluation (F1 scores, comprehensive evaluation)
    - analysis: Protein ranking and interpretability
    - data_export: Data export and visualization pipeline
"""

from .data import (
    read_fasta,
    download_swissprot,
    download_uniref50,
    download_annotated_proteins,
    load_annotations_tsv,
    proteins_to_concept_labels,
)
from .eval import compute_f1_scores
from .data_export import (
    save_activations_parquet,
    save_activations_duckdb,
    save_feature_data,
    build_dashboard_data,
    export_protein_features_json,
    export_protein_features_parquet,
    launch_protein_dashboard,
)

__version__ = "0.1.0"

__all__ = [
    # Data
    'read_fasta',
    'download_swissprot',
    'download_uniref50',
    'download_annotated_proteins',
    'load_annotations_tsv',
    'proteins_to_concept_labels',
    # Eval
    'compute_f1_scores',
    # Data Export
    'save_activations_parquet',
    'save_activations_duckdb',
    'save_feature_data',
    'build_dashboard_data',
    'export_protein_features_json',
    'export_protein_features_parquet',
    'launch_protein_dashboard',
]
