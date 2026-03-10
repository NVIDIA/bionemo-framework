from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

@dataclass
class MaskedLMOutput:
    preds: np.ndarray
    labels: np.ndarray
    ids: np.ndarray = None

@dataclass
class MutationPredictionOutput:
    ref_likelihoods: np.ndarray
    alt_likelihoods: np.ndarray
    likelihood_ratios: np.ndarray
    ids: np.ndarray = None

@dataclass
class FitnessPredictionOutput:
    fitness: np.ndarray
    ids: np.ndarray = None

@dataclass
class EmbeddingOutput:
    embeddings: np.ndarray
    ids: np.ndarray = None

@dataclass
class DownstreamPredictionOutput:
    """Output for downstream task predictions (classification or regression)."""
    predictions: np.ndarray  # Raw predictions from downstream head
    probabilities: np.ndarray = None  # For classification tasks (softmax applied)
    predicted_classes: np.ndarray = None  # For classification tasks (argmax of logits)
    ids: np.ndarray = None
    
@dataclass
class NextCodonPredictionOutput:
    preds: np.ndarray
    ids: np.ndarray = None
    labels: np.ndarray = None

@dataclass
class SequenceGenerationOutput:
    """Output for sequence generation tasks."""
    generated_ids: np.ndarray  # Generated token IDs (padded to max_length)
    ids: np.ndarray = None  # Input sequence identifiers