"""
SAE: Generic Sparse Autoencoder Package

A domain-agnostic implementation of Sparse Autoencoders (SAEs) for
interpretability research. Provides multiple SAE architectures, training
utilities, and evaluation metrics.

Main Components:
    - architectures: SAE implementations (ReLU-L1, Top-K)
    - training: Training loop and configuration
    - eval: Evaluation metrics (reconstruction, loss recovered, dead latents)
    - utils: Utility functions (device, seed, memory)
"""

from .architectures import SparseAutoencoder, ReLUSAE, TopKSAE, MoESAE
from .training import Trainer, TrainingConfig, WandbConfig, ParallelConfig
from .perf_logger import PerfLogger
from .eval import (
    DeadLatentTracker,
    compute_reconstruction_metrics,
    evaluate_sparsity,
    SparsityMetrics,
    compute_loss_recovered,
    evaluate_loss_recovered,
    LossRecoveredResult,
    evaluate_sae,
    EvalResults,
)
from .utils import get_device, set_seed
from .activation_store import (
    ActivationStore,
    ActivationStoreConfig,
    save_activations,
    load_activations,
)
from .analysis import (
    FeatureStats,
    TopExample,
    FeatureGeometry,
    FeatureLogits,
    ClusterInfo,
    compute_feature_stats,
    compute_feature_umap,
    compute_feature_logits,
    compute_cluster_centroids,
    build_cluster_label_prompt,
    save_cluster_labels,
    save_feature_atlas,
    export_text_features_parquet,
    launch_dashboard,
)
from .collector import (
    TokenActivationCollector,
    TokenExample,
    CollectorResult,
)
from .autointerp import (
    LLMClient,
    LLMResponse,
    AnthropicClient,
    OpenAICompatibleClient,
    OpenAIClient,
    NIMClient,
    NVIDIAInternalClient,
    FeatureSampler,
    FeatureExamples,
    AutoInterpreter,
    FeatureInterpretation,
    DEFAULT_PROMPT_TEMPLATE,
    TOKEN_PROMPT_TEMPLATE,
)
from .steering import SteeredModel, Intervention, InterventionMode
from .process_group_manager import ProcessGroupManager

__version__ = "0.1.0"

__all__ = [
    # Architectures
    'SparseAutoencoder',
    'ReLUSAE',
    'TopKSAE',
    'MoESAE',
    # Training
    'Trainer',
    'TrainingConfig',
    'WandbConfig',
    'ParallelConfig',
    'PerfLogger',
    # Activation Store
    'ActivationStore',
    'ActivationStoreConfig',
    'save_activations',
    'load_activations',
    # Analysis
    'FeatureStats',
    'TopExample',
    'FeatureGeometry',
    'FeatureLogits',
    'ClusterInfo',
    'compute_feature_stats',
    'compute_feature_umap',
    'compute_feature_logits',
    'compute_cluster_centroids',
    'build_cluster_label_prompt',
    'save_cluster_labels',
    'save_feature_atlas',
    'export_text_features_parquet',
    'launch_dashboard',
    # Collector
    'TokenActivationCollector',
    'TokenExample',
    'CollectorResult',
    # Evaluation
    'DeadLatentTracker',
    'compute_reconstruction_metrics',
    'evaluate_sparsity',
    'SparsityMetrics',
    'compute_loss_recovered',
    'evaluate_loss_recovered',
    'LossRecoveredResult',
    'evaluate_sae',
    'EvalResults',
    # Utils
    'get_device',
    'set_seed',
    # Auto-interpretation
    'LLMClient',
    'LLMResponse',
    'AnthropicClient',
    'OpenAICompatibleClient',
    'OpenAIClient',
    'NIMClient',
    'NVIDIAInternalClient',
    'FeatureSampler',
    'FeatureExamples',
    'AutoInterpreter',
    'FeatureInterpretation',
    'DEFAULT_PROMPT_TEMPLATE',
    'TOKEN_PROMPT_TEMPLATE',
    # Steering
    'SteeredModel',
    'Intervention',
    'InterventionMode',
    # Process Group Manager
    'ProcessGroupManager',
]
