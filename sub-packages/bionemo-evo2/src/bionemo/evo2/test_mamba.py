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

"""
Test script for the Mamba model integration with BioNeMo Evo2.
This demonstrates how to use the Mamba model with the same infrastructure
as the existing Hyena models in BioNeMo Evo2.
"""

import argparse
import tempfile
from pathlib import Path

from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections.llm.gpt.data import MockDataModule

from bionemo.evo2.models.mamba import MAMBA_MODEL_OPTIONS, MambaModel
from bionemo.testing.testing_callbacks import SignalAfterGivenStepCallback, TimingCallback


def test_mamba(
    model_size: str = "nemotron5_8b",
    micro_batch_size: int = 1,
    sequence_length: int = 512,
    max_steps: int = 5,
    fp8: bool = False,
):
    """
    Test the Mamba model with a mock dataset.

    Args:
        model_size: Size of the model to test
        micro_batch_size: Micro batch size
        sequence_length: Sequence length for training
        max_steps: Number of steps to train
        fp8: Whether to use FP8 precision
    """
    print(f"Testing Mamba model with size: {model_size}")

    # Check if the model size exists
    if model_size not in MAMBA_MODEL_OPTIONS:
        raise ValueError(f"Invalid model size: {model_size}. Available options: {list(MAMBA_MODEL_OPTIONS.keys())}")

    # Set up distributed training
    torch_tp_size = 1
    torch_pp_size = 1
    torch_cp_size = 1
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=torch_tp_size,
        pipeline_model_parallel_size=torch_pp_size,
        context_parallel_size=torch_cp_size,
    )

    # Set up temporary directory for saving checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        tensorboard_dir = experiment_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)

        # Define model configuration
        config_modifiers = {
            "seq_length": sequence_length,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "to_upper": "normalized_weighted",
            "distribute_saved_activations": True,
            "cross_entropy_loss_fusion": False,
            "fp32_residual_connection": True,
            "add_bias_output": False,
        }

        # Create model config
        model_config = MAMBA_MODEL_OPTIONS[model_size](**config_modifiers)

        # Set up data module with mock data
        data = MockDataModule(
            micro_batch_size=micro_batch_size,
            global_batch_size=micro_batch_size,
            seq_length=sequence_length,
            use_lags=False,  # No need for lag tensors
            random_seed=42,
            num_workers=0,
        )

        # Create model
        model = MambaModel(model_config, tokenizer=data.tokenizer)

        # Set up learning rate scheduler
        sched = nl.CosineAnnealing(
            warmup_steps=2,  # Very short warmup for test
            decay_steps=max_steps - 2,
            min_lr=1e-5,
            init_lr=0.0,
            max_lr=3e-4,
        )

        # Set up optimizer
        opt_config = OptimizerConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            eps=1e-8,
            use_cpu_initialization=False,
            overlap_grad_sync=False,
            overlap_param_sync=False,
            dtype=model_config.params_dtype,
            grad_clip_norm=1.0,
        )

        # Connect optimizer to model
        opt = nl.MegatronOptimizerModule(
            opt_config, sched, no_weight_decay_cond=model_config.hyena_no_weight_decay_cond_fn
        )
        opt.connect(model)

        # Set up callbacks
        callbacks = [
            RichModelSummary(max_depth=4),
            LearningRateMonitor(),
            TimingCallback(),
            SignalAfterGivenStepCallback(stop_step=max_steps, stop_before_step=True, use_trainer_should_stop=True),
        ]

        # Set up logger
        logger = TensorBoardLogger(
            save_dir=str(tensorboard_dir),
            name="mamba_test",
        )

        # Set up precision plugin
        precision_plugin = nl.MixedPrecisionPlugin(
            precision="bf16-mixed",
            grad_scaling=False,
            fp8=None if not fp8 else "hybrid",  # "hybrid" for FP8
        )

        # Set up parallel strategy
        ddp_config = DistributedDataParallelConfig(
            bucket_cap_mb=25,
            overlap_grad_sync=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            find_unused_parameters=False,
            verify_sync=False,
        )
        strategy = nl.MegatronStrategy(
            ddp_config=ddp_config,
            num_micro_batches=1,
            sequence_parallel=False,
            tensor_model_parallel_size=torch_tp_size,
            pipeline_model_parallel_size=torch_pp_size,
            model_parallel_backend="nccl",
            grad_sync_timeout=60,
            tp_comm_overlap_strategy=None,
            create_gpu_buffers=False,
            sequence_parallel_communication_early_overlap=False,
            sequence_parallel_communication_data_overlap=False,
            enable_grad_sync_boundary_context=True,
            context_parallel_size=torch_cp_size,
            grad_sync_dtype=None,
            executor_type="data-parallel",
        )

        # Set up trainer
        trainer = nl.Trainer(
            devices=1,
            max_steps=max_steps,
            val_check_interval=max_steps,
            limit_val_batches=1,
            log_every_n_steps=1,
            strategy=strategy,
            precision=precision_plugin,
            logger=logger,
            callbacks=callbacks,
            enable_checkpointing=False,
            use_distributed_sampler=False,
            gradient_clip_val=1.0,
        )

        # Train model
        print("Starting training...")
        trainer.fit(model, data)
        print("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Mamba model integration with BioNeMo Evo2")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=sorted(MAMBA_MODEL_OPTIONS.keys()),
        default="nemotron5_8b",
        help="Model size to test",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Number of steps to train",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Use FP8 precision",
    )

    args = parser.parse_args()
    test_mamba(
        model_size=args.model_size,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.sequence_length,
        max_steps=args.max_steps,
        fp8=args.fp8,
    )
