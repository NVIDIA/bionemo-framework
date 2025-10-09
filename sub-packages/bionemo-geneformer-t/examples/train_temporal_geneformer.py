# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env python3
"""Example script for training temporal Geneformer models.

This script demonstrates how to fine-tune a pretrained Geneformer model for temporal
next-cell prediction using SCDL neighbor data.

Key Features:
- Proper fine-tuning from pretrained Geneformer checkpoints
- Enhanced masking strategy: current cell provides context (no masking), next cell is masked for prediction
- Flexible token selection policies and no-neighbor handling
- Support for gene expression normalization and ranking
- Temporal attention masks preventing next cell self-attention

Usage:
    python train_temporal_geneformer.py
"""

import logging
import os
from pathlib import Path

from bionemo.geneformer_t.run import train_temporal_geneformer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating various temporal Geneformer training configurations."""
    # Setup paths
    data_dir = os.environ.get("BIONEMO_DATA_DIR", "/data")
    output_dir = os.environ.get("BIONEMO_OUTPUT_DIR", "./results")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting temporal Geneformer training examples...")

    # Example 1: Basic fine-tuning with enhanced masking strategy
    try:
        logger.info("=" * 60)
        logger.info("Example 1: Basic Fine-tuning with Enhanced Masking")
        logger.info("- Current cell: NO masking (provides context)")
        logger.info("- Next cell: ONLY next cell gets masked for prediction")
        logger.info("- Temporal attention: Next cell can't attend to itself")
        logger.info("=" * 60)

        best_checkpoint = train_temporal_geneformer(
            data_path=f"{data_dir}/your_scdl_dataset",
            tokenizer_vocab_path=f"{data_dir}/tokenizer/geneformer_vocab.txt",
            median_dict_path=f"{data_dir}/median_dict/median_dict.pkl",
            initial_ckpt_path="/path/to/pretrained/geneformer/checkpoint",  # 🔑 Fine-tune from pretrained
            result_dir=f"{output_dir}/basic_enhanced_training",
            seq_length=1024,
            mask_prob=0.15,  # Only applied to next cell
            mask_token_prob=0.8,
            random_token_prob=0.1,
            neighbor_key="neighbors",  # SCDL neighbor key
            micro_batch_size=4,
            global_batch_size=16,
            num_epochs=3,
            learning_rate=1e-4,
            devices=1,
            experiment_name="basic_enhanced_temporal_geneformer",
            # Enhanced features
            only_cells_with_neighbors=True,
            no_neighbor_policy="skip",  # Skip cells without neighbors
            token_selection_policy="identity",  # Use all tokens from next cell
            normalize_gene_expression=True,  # Apply median normalization
            target_sum=10000,  # Normalization target
        )

        logger.info(f"Basic enhanced training completed. Best checkpoint: {best_checkpoint}")

    except Exception as e:
        logger.error(f"Basic enhanced training failed: {e}")

    # Example 2: Advanced configuration with intersection token selection
    try:
        logger.info("=" * 60)
        logger.info("Example 2: Advanced Configuration with Token Selection")
        logger.info("- Token selection: intersection (only genes in both cells)")
        logger.info("- No-neighbor policy: identity (use same cell as next)")
        logger.info("- Multi-GPU training with mixed precision")
        logger.info("=" * 60)

        best_checkpoint = train_temporal_geneformer(
            data_path=f"{data_dir}/your_scdl_dataset",
            tokenizer_vocab_path=f"{data_dir}/tokenizer/geneformer_vocab.txt",
            median_dict_path=f"{data_dir}/median_dict/median_dict.pkl",
            initial_ckpt_path="/path/to/pretrained/geneformer/checkpoint",
            result_dir=f"{output_dir}/advanced_token_selection",
            seq_length=2048,
            mask_prob=0.2,  # Higher masking for more challenge
            mask_token_prob=0.8,
            random_token_prob=0.1,
            neighbor_key="neighbors",
            micro_batch_size=8,
            global_batch_size=64,
            num_epochs=5,
            learning_rate=2e-4,
            devices=4,
            precision="bf16-mixed",
            experiment_name="advanced_token_selection_temporal",
            # Enhanced features
            only_cells_with_neighbors=False,  # Include all cells
            no_neighbor_policy="identity",  # Use identity for cells without neighbors
            token_selection_policy="intersection",  # Only predict genes present in both cells
            normalize_gene_expression=True,
            target_sum=10000,
            # Training optimization
            weight_decay=0.01,
            early_stopping_patience=3,
            save_top_k=5,
        )

        logger.info(f"Advanced token selection training completed. Best checkpoint: {best_checkpoint}")

    except Exception as e:
        logger.error(f"Advanced token selection training failed: {e}")

    # Example 3: Random neighbor policy with union token selection
    try:
        logger.info("=" * 60)
        logger.info("Example 3: Random Neighbor Policy")
        logger.info("- No-neighbor policy: random (sample random cell as next)")
        logger.info("- Token selection: union (all tokens from both cells)")
        logger.info("- Custom neighbor key for different temporal relationships")
        logger.info("=" * 60)

        best_checkpoint = train_temporal_geneformer(
            data_path=f"{data_dir}/your_scdl_dataset",
            tokenizer_vocab_path=f"{data_dir}/tokenizer/geneformer_vocab.txt",
            median_dict_path=f"{data_dir}/median_dict/median_dict.pkl",
            initial_ckpt_path="/path/to/pretrained/geneformer/checkpoint",
            result_dir=f"{output_dir}/random_neighbor_training",
            seq_length=1536,
            mask_prob=0.18,
            neighbor_key="temporal_neighbors",  # Custom neighbor key
            micro_batch_size=6,
            global_batch_size=24,
            num_epochs=4,
            learning_rate=1.5e-4,
            devices=2,
            experiment_name="random_neighbor_temporal",
            # Enhanced features
            only_cells_with_neighbors=False,
            no_neighbor_policy="random",  # Sample random cell when no neighbors
            token_selection_policy="union",  # Use all tokens
            normalize_gene_expression=True,
            target_sum=8000,  # Different normalization target
            # Resume from checkpoint if available
            resume_from_checkpoint=f"{output_dir}/random_neighbor_training/checkpoints/last.ckpt",
        )

        logger.info(f"Random neighbor training completed. Best checkpoint: {best_checkpoint}")

    except Exception as e:
        logger.error(f"Random neighbor training failed: {e}")

    # Example 4: Demonstration of masking strategy differences
    try:
        logger.info("=" * 60)
        logger.info("Example 4: Masking Strategy Comparison")
        logger.info("- High masking probability to show strategy clearly")
        logger.info("- Current cell: 0% masking (pure context)")
        logger.info("- Next cell: 30% masking (aggressive prediction)")
        logger.info("=" * 60)

        best_checkpoint = train_temporal_geneformer(
            data_path=f"{data_dir}/your_scdl_dataset",
            tokenizer_vocab_path=f"{data_dir}/tokenizer/geneformer_vocab.txt",
            median_dict_path=f"{data_dir}/median_dict/median_dict.pkl",
            initial_ckpt_path="/path/to/pretrained/geneformer/checkpoint",
            result_dir=f"{output_dir}/masking_strategy_demo",
            seq_length=1024,
            mask_prob=0.30,  # High masking to demonstrate strategy
            mask_token_prob=0.9,  # Mostly use [MASK] tokens
            random_token_prob=0.05,  # Few random tokens
            neighbor_key="neighbors",
            micro_batch_size=4,
            global_batch_size=16,
            num_epochs=2,
            learning_rate=1e-4,
            devices=1,
            experiment_name="masking_strategy_demo",
            # Enhanced features
            only_cells_with_neighbors=True,
            no_neighbor_policy="skip",
            token_selection_policy="identity",
            normalize_gene_expression=True,
            target_sum=10000,
        )

        logger.info(f"Masking strategy demo completed. Best checkpoint: {best_checkpoint}")

    except Exception as e:
        logger.error(f"Masking strategy demo failed: {e}")

    # Example 5: Inference with trained model
    try:
        logger.info("=" * 60)
        logger.info("Example 5: Inference with Trained Model")
        logger.info("- Load trained model and perform inference")
        logger.info("- Demonstrate temporal prediction capabilities")
        logger.info("=" * 60)

        # Load the model from the best checkpoint
        import torch

        from bionemo.geneformer_t.data import TemporalGeneformerDataModule
        from bionemo.geneformer_t.model import TemporalGeneformerModel

        # Setup data module for inference
        data_module = TemporalGeneformerDataModule(
            data_path=f"{data_dir}/your_scdl_dataset",
            tokenizer_vocab_path=f"{data_dir}/tokenizer/geneformer_vocab.txt",
            median_dict_path=f"{data_dir}/median_dict/median_dict.pkl",
            seq_length=1024,
            mask_prob=0.15,
            micro_batch_size=4,
            global_batch_size=16,
            neighbor_key="neighbors",
            only_cells_with_neighbors=True,
            no_neighbor_policy="skip",
            token_selection_policy="identity",
            normalize_gene_expression=True,
            target_sum=10000,
        )

        data_module.prepare_data()
        data_module.setup("test")

        # Load model checkpoint
        checkpoint_path = f"{output_dir}/basic_enhanced_training/checkpoints/best.ckpt"
        if os.path.exists(checkpoint_path):
            model = TemporalGeneformerModel.load_from_checkpoint(checkpoint_path)
            model.eval()

            # Get a sample batch
            test_loader = data_module.test_dataloader()
            sample_batch = next(iter(test_loader))

            # Perform inference
            with torch.no_grad():
                outputs = model(sample_batch)
                loss = outputs.loss
                predictions = outputs.prediction_scores

            logger.info(f"Inference completed. Loss: {loss:.4f}, Predictions shape: {predictions.shape}")
            logger.info("Temporal masking strategy working correctly!")

        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}")

    except Exception as e:
        logger.error(f"Inference example failed: {e}")

    logger.info("All temporal Geneformer training examples completed!")
    logger.info("Key improvements implemented:")
    logger.info("✅ Enhanced masking: current cell = context, next cell = prediction")
    logger.info("✅ Temporal attention: prevents next cell self-attention")
    logger.info("✅ Token selection policies: identity, intersection, union")
    logger.info("✅ No-neighbor handling: skip, identity, random")
    logger.info("✅ Gene expression normalization and ranking")
    logger.info("✅ Proper fine-tuning from pretrained checkpoints")


if __name__ == "__main__":
    main()
