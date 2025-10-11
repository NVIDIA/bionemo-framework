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


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Training script for temporal Geneformer fine-tuning."""

import logging
from pathlib import Path
from typing import Optional, Union

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from nemo import lightning as nl
from nemo.utils import logging as nemo_logging

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.geneformer_t.data.temporal_datamodule import TemporalGeneformerDataModule
from bionemo.geneformer_t.model.temporal_model import create_temporal_geneformer_config
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.utils.logger_utils import WandbConfig
from bionemo.llm.utils.optim_utils import CosineAnnealingScheduler, MegatronOptimizerModule, OptimizerConfig


__all__ = ["train_temporal_geneformer"]

logger = logging.getLogger(__name__)


def train_temporal_geneformer(
    data_path: Union[str, Path],
    tokenizer_vocab_path: Union[str, Path],
    median_dict_path: Union[str, Path],
    initial_ckpt_path: Union[str, Path],
    result_dir: Union[str, Path],
    seq_length: int = 2048,
    mask_prob: float = 0.15,
    mask_token_prob: float = 0.8,
    random_token_prob: float = 0.1,
    neighbor_key: str = "neighbors",
    micro_batch_size: int = 4,
    global_batch_size: int = 16,
    num_epochs: int = 3,
    max_steps: Optional[int] = None,
    learning_rate: float = 1e-4,
    min_lr: Optional[float] = None,
    warmup_steps: int = 100,
    cosine_rampup_frac: float = 0.05,
    cosine_hold_frac: float = 0.05,
    weight_decay: float = 0.01,
    precision: PrecisionTypes = "bf16-mixed",
    devices: int = 1,
    num_nodes: int = 1,
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 1.0,
    val_check_interval: int = 100,
    limit_val_batches: int = 10,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    save_top_k: int = 3,
    save_last: bool = True,
    monitor_metric: str = "val_loss",
    early_stopping_patience: Optional[int] = None,
    experiment_name: str = "temporal_geneformer",
    wandb_config: Optional[WandbConfig] = None,
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec,
    seed: int = 42,
    only_cells_with_neighbors: bool = True,
    resume_from_checkpoint: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Path:
    """Train a temporal Geneformer model for next-cell prediction.

    This function fine-tunes a pretrained Geneformer model for temporal next-cell
    prediction using SCDL neighbor data. The model learns to predict masked tokens
    in the next cell while preserving the current cell's representation.

    Args:
        data_path: Path to SCDL dataset directory
        tokenizer_vocab_path: Path to tokenizer vocabulary file
        median_dict_path: Path to median dictionary file
        initial_ckpt_path: Path to pretrained Geneformer checkpoint
        result_dir: Directory to save results and checkpoints
        seq_length: Maximum sequence length
        mask_prob: Probability of masking tokens in next cell
        mask_token_prob: Probability of using [MASK] token
        random_token_prob: Probability of using random token
        neighbor_key: Key for neighbor data in SCDL
        micro_batch_size: Micro batch size
        global_batch_size: Global batch size
        num_epochs: Number of training epochs
        max_steps: Maximum number of training steps (overrides num_epochs)
        learning_rate: Learning rate
        min_lr: Minimum learning rate for cosine annealing
        warmup_steps: Number of warmup steps
        cosine_rampup_frac: Fraction of steps for cosine rampup
        cosine_hold_frac: Fraction of steps to hold minimum LR
        weight_decay: Weight decay
        precision: Training precision
        devices: Number of devices
        num_nodes: Number of nodes
        accumulate_grad_batches: Gradient accumulation steps
        gradient_clip_val: Gradient clipping value
        val_check_interval: Validation check interval
        limit_val_batches: Limit validation batches
        num_workers: Number of data loader workers
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        save_top_k: Number of top checkpoints to save
        save_last: Whether to save last checkpoint
        monitor_metric: Metric to monitor for checkpointing
        early_stopping_patience: Early stopping patience (None to disable)
        experiment_name: Name of the experiment
        wandb_config: Weights & Biases configuration
        biobert_spec_option: BioBERT architecture specification
        seed: Random seed
        only_cells_with_neighbors: Whether to only use cells with neighbors
        resume_from_checkpoint: Path to checkpoint to resume from
        **kwargs: Additional arguments

    Returns:
        Path to the saved checkpoint directory
    """
    # Set up logging
    nemo_logging.setup_logging()
    logger.info("Starting temporal Geneformer fine-tuning")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Initial checkpoint: {initial_ckpt_path}")
    logger.info(f"Result directory: {result_dir}")

    # Convert paths
    data_path = Path(data_path)
    tokenizer_vocab_path = Path(tokenizer_vocab_path)
    median_dict_path = Path(median_dict_path)
    initial_ckpt_path = Path(initial_ckpt_path)
    result_dir = Path(result_dir)

    # Create result directory
    result_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    if not initial_ckpt_path.exists():
        raise FileNotFoundError(f"Initial checkpoint does not exist: {initial_ckpt_path}")
    if not tokenizer_vocab_path.exists():
        raise FileNotFoundError(f"Tokenizer vocab does not exist: {tokenizer_vocab_path}")
    if not median_dict_path.exists():
        raise FileNotFoundError(f"Median dict does not exist: {median_dict_path}")

    # Create model configuration
    config = create_temporal_geneformer_config(
        initial_ckpt_path=str(initial_ckpt_path),
        seq_length=seq_length,
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),
        biobert_spec_option=biobert_spec_option,
        **kwargs,
    )

    logger.info(f"Created temporal Geneformer config with seq_length={seq_length}")

    # Create data module
    data_module = TemporalGeneformerDataModule(
        data_path=data_path,
        tokenizer_vocab_path=tokenizer_vocab_path,
        median_dict_path=median_dict_path,
        seq_length=seq_length,
        mask_prob=mask_prob,
        mask_token_prob=mask_token_prob,
        random_token_prob=random_token_prob,
        neighbor_key=neighbor_key,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        seed=seed,
        only_cells_with_neighbors=only_cells_with_neighbors,
    )

    # Prepare data to get tokenizer
    data_module.prepare_data()
    logger.info("Data module prepared successfully")

    # Calculate training steps
    if max_steps is None:
        # Estimate steps per epoch (this is approximate)
        data_module.setup("fit")
        steps_per_epoch = len(data_module.train_dataloader())
        max_steps = steps_per_epoch * num_epochs
        logger.info(
            f"Calculated max_steps: {max_steps} (steps_per_epoch: {steps_per_epoch}, num_epochs: {num_epochs})"
        )

    # Set minimum learning rate
    if min_lr is None:
        min_lr = learning_rate / 100

    # Create optimizer configuration
    optimizer_config = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=learning_rate,
            optimizer="adam",
            use_distributed_optimizer=True,
            weight_decay=weight_decay,
            fp16=precision == "16-mixed",
            bf16=precision == "bf16-mixed",
        ),
        lr_scheduler=CosineAnnealingScheduler(
            max_steps=max_steps,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            interval="step",
            monitor=monitor_metric,
            constant_steps=int(max_steps * cosine_hold_frac),
        ),
    )

    # Create lightning module
    model = biobert_lightning_module(
        config=config,
        tokenizer=data_module.tokenizer,
        optimizer=optimizer_config,
    )

    logger.info("Created lightning module")

    # Set up strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
    )

    # Set up callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=result_dir / "checkpoints",
        filename=f"{experiment_name}-{{epoch:02d}}-{{step}}-{{{monitor_metric}:.4f}}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=save_top_k,
        save_last=save_last,
        every_n_train_steps=val_check_interval,
        always_save_context=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if early_stopping_patience is not None:
        early_stopping_callback = EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stopping_callback)

    # Create trainer
    trainer = nl.Trainer(
        devices=devices,
        num_nodes=num_nodes,
        max_steps=max_steps,
        accelerator="gpu",
        strategy=strategy,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=50,
        callbacks=callbacks,
        enable_checkpointing=True,
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(precision=precision),
    )

    # Set up logger
    # nemo_logger = setup_nemo_lightning_logger(
    #     root_dir=result_dir,
    #     name=experiment_name,
    #     initialize_tensorboard_logger=True,
    #     wandb_config=wandb_config,
    #     ckpt_callback=checkpoint_callback,
    # )

    logger.info("Starting training...")

    # Train the model
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=resume_from_checkpoint,
    )

    logger.info("Training completed!")

    # Return path to best checkpoint
    best_checkpoint_path = Path(checkpoint_callback.best_model_path)
    logger.info(f"Best checkpoint saved at: {best_checkpoint_path}")

    return best_checkpoint_path


def main():
    """Main entry point for command-line training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train temporal Geneformer for next-cell prediction")

    # Required arguments
    parser.add_argument("--data-path", type=Path, required=True, help="Path to SCDL dataset")
    parser.add_argument("--tokenizer-vocab-path", type=Path, required=True, help="Path to tokenizer vocabulary")
    parser.add_argument("--median-dict-path", type=Path, required=True, help="Path to median dictionary")
    parser.add_argument(
        "--initial-ckpt-path", type=Path, required=True, help="Path to pretrained Geneformer checkpoint"
    )
    parser.add_argument("--result-dir", type=Path, required=True, help="Directory to save results")

    # Optional arguments
    parser.add_argument("--seq-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Probability of masking tokens")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--global-batch-size", type=int, default=16, help="Global batch size")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--experiment-name", type=str, default="temporal_geneformer", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Train the model
    train_temporal_geneformer(
        data_path=args.data_path,
        tokenizer_vocab_path=args.tokenizer_vocab_path,
        median_dict_path=args.median_dict_path,
        initial_ckpt_path=args.initial_ckpt_path,
        result_dir=args.result_dir,
        seq_length=args.seq_length,
        mask_prob=args.mask_prob,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        devices=args.devices,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
