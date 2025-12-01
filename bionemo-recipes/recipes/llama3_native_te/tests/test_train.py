# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import random

import pytest
import torch
from hydra import compose, initialize_config_dir

from train_ddp import main as main_ddp
from train_fsdp2 import main as main_fsdp2


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def test_sanity_convergence_ddp(tmp_path, recipe_path, mock_genomic_parquet):
    """Test that DDP training converges on mock genomic data.

    This test validates:
    - The train_ddp.py script runs end-to-end without errors
    - Model, optimizer, and dataloader integrate correctly
    - Training converges to reasonable loss on small dataset
    - Uses L0_sanity config with small model and few training steps
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.load_dataset_kwargs.data_files={mock_genomic_parquet}",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_ddp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # For genomic Causal LM, we expect convergence to < 5.0 on the small test dataset
    # The model should learn to predict simple patterns in the mock data
    assert final_loss < 5.0, f"Final loss {final_loss} is too high, expected < 5.0"


def test_sanity_convergence_fsdp2(tmp_path, recipe_path, mock_genomic_parquet):
    """Test that FSDP2 training converges on mock genomic data.

    This test validates:
    - The train_fsdp2.py script runs end-to-end without errors
    - FSDP2 wrapping and sharding work correctly
    - Training converges to reasonable loss on small dataset
    - Uses L0_sanity config with small model and few training steps
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.load_dataset_kwargs.data_files={mock_genomic_parquet}",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_fsdp2(sanity_config)

    # FSDP2 should achieve similar convergence to DDP
    assert final_loss < 5.0, f"Final loss {final_loss} is too high, expected < 5.0"


def test_sanity_convergence_ddp_non_streaming_dataset(tmp_path, recipe_path, mock_genomic_parquet):
    """Test that DDP training works with non-streaming dataset.

    This test validates:
    - The dataloader works correctly with streaming=False
    - Map-style dataset integration works
    - Training converges similarly to streaming mode
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.load_dataset_kwargs.data_files={mock_genomic_parquet}",
                "dataset.load_dataset_kwargs.streaming=False",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_ddp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # Non-streaming mode should converge just as well as streaming
    assert final_loss < 5.0, f"Final loss {final_loss} is too high, expected < 5.0"


def test_sanity_convergence_fsdp2_non_streaming_dataset(tmp_path, recipe_path, mock_genomic_parquet):
    """Test that FSDP2 training works with non-streaming dataset.

    This test validates:
    - FSDP2 works correctly with map-style datasets
    - Non-streaming mode doesn't break FSDP2 sharding
    - Training converges similarly to streaming mode
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.load_dataset_kwargs.data_files={mock_genomic_parquet}",
                "dataset.load_dataset_kwargs.streaming=False",
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # Non-streaming mode should converge just as well as streaming
    assert final_loss < 5.0, f"Final loss {final_loss} is too high, expected < 5.0"


def test_sanity_ddp_with_lazy_tokenization(tmp_path, recipe_path, mock_genomic_parquet):
    """Test that DDP training works with lazy tokenization enabled.

    This test validates:
    - Lazy tokenization (one-to-one mapping) works correctly
    - Training can run with lazy tokenization
    - No errors occur during forward/backward passes
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.load_dataset_kwargs.data_files={mock_genomic_parquet}",
                "dataset.use_lazy_tokenization=True",
                "num_train_steps=10",  # Just verify it runs, don't test convergence
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_ddp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # Just check that training runs without errors
    # We don't check convergence because lazy tokenization produces different windowing
    assert final_loss is not None, "Training should complete and return a loss value"


def test_sanity_fsdp2_with_lazy_tokenization(tmp_path, recipe_path, mock_genomic_parquet):
    """Test that FSDP2 training works with lazy tokenization enabled.

    This test validates:
    - Lazy tokenization works with FSDP2
    - FSDP2 sharding doesn't break with lazy tokenization
    - No errors occur during forward/backward passes
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.load_dataset_kwargs.data_files={mock_genomic_parquet}",
                "dataset.use_lazy_tokenization=True",
                "num_train_steps=10",  # Just verify it runs, don't test convergence
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # Just check that training runs without errors
    assert final_loss is not None, "Training should complete and return a loss value"


def test_sanity_convergence_ddp_with_sequence_packing(tmp_path, recipe_path, mock_genomic_parquet):
    """Test that DDP training works with sequence packing enabled.

    This test validates:
    - Sequence packing works correctly
    - Training can run with sequence packing
    - No errors occur during forward/backward passes
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.load_dataset_kwargs.data_files={mock_genomic_parquet}",
                "use_sequence_packing=true",
                "dataset.max_seq_length=1024",
                "config_kwargs.attn_input_format=thd",
                "num_train_steps=10",  # Just verify it runs, don't test convergence
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_ddp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # Just check that training runs without errors
    assert final_loss < 5.0, f"Final loss {final_loss} is too high, expected < 5.0"


def test_sanity_convergence_fsdp2_with_sequence_packing(tmp_path, recipe_path, mock_genomic_parquet):
    """Test that FSDP2 training works with sequence packing enabled.

    This test validates:
    - Sequence packing works correctly
    - Training can run with sequence packing
    - No errors occur during forward/backward passes
    """
    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.load_dataset_kwargs.data_files={mock_genomic_parquet}",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
                "dataset.max_seq_length=1024",
                "num_train_steps=10",  # Just verify it runs, don't test convergence
                "checkpoint.resume_from_checkpoint=false",  # Don't try to resume - fresh training
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    # Just check that training runs without errors
    assert final_loss < 5.0, f"Final loss {final_loss} is too high, expected < 5.0"
