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

import random

import pytest
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from torch.distributed.device_mesh import _mesh_resources
from transformer_engine.pytorch.fp8 import check_fp8_support

from train_ddp import main as main_ddp
from train_fsdp2 import main as main_fsdp2
from train_mfsdp import main as main_mfsdp


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


requires_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or not check_fp8_support()[0],
    reason="Test requires FP8 support: " + check_fp8_support()[1],
)


@pytest.fixture
def distributed_cleanup():
    yield

    # Try to destroy the process group, but don't fail if it's not available.
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except (AssertionError, RuntimeError):
        pass

    # Clear ALL mesh resources (for both MFSDP and FSDP2) to avoid issues re-running in the same process.
    _mesh_resources.mesh_stack.clear()
    _mesh_resources.child_to_root_mapping.clear()
    _mesh_resources.root_to_flatten_mapping.clear()
    _mesh_resources.flatten_name_to_root_dims.clear()
    _mesh_resources.mesh_dim_group_options.clear()

    # Clear CUDA cache to prevent memory issues between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Reset CUDA device to ensure clean state
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        # Reset all accumulated values in CUDA memory stats
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_accumulated_memory_stats(device)


def test_sanity_convergence_mfsdp(distributed_cleanup, tmp_path, recipe_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
            ],
        )

    final_loss = main_mfsdp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_mfsdp_meta_device(distributed_cleanup, tmp_path, recipe_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_meta_device=true",
            ],
        )

    final_loss = main_mfsdp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


@pytest.mark.xfail(reason="Meta-device init seems to be having some issues with convergence (BIONEMO-2719)")
def test_sanity_convergence_mfsdp_eager_meta_device(distributed_cleanup, tmp_path, recipe_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "model_tag=facebook/esm2_t6_8M_UR50D",
                "use_meta_device=true",
            ],
        )

    final_loss = main_mfsdp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_ddp(distributed_cleanup, tmp_path, recipe_path):
    """Test that the main function can be invoked wrapping the model in DDP."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
            ],
        )

    final_loss = main_ddp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_ddp_non_streaming_dataset(distributed_cleanup, tmp_path, recipe_path):
    """Test that the training script works with a non-streaming dataset."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "dataset.load_dataset_kwargs.streaming=False",
            ],
        )

    final_loss = main_ddp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2(distributed_cleanup, tmp_path, recipe_path):
    """Test that the main function can be invoked wrapping the model in FSDP2."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


@requires_fp8
def test_sanity_mfsdp_fp8(distributed_cleanup, tmp_path, recipe_path):
    # For MFSDP, we only check that the script can run successfully with FP8, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "fp8_config.enabled=true",
                f"checkpoint.ckpt_dir={tmp_path}",
                "num_train_steps=4",
            ],
        )

    main_mfsdp(sanity_config)


@requires_fp8
def test_sanity_ddp_fp8(distributed_cleanup, tmp_path, recipe_path):
    # For DDP, we only check that the script can run successfully with FP8, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
                "num_train_steps=4",
            ],
        )

    main_ddp(sanity_config)


@requires_fp8
def test_sanity_convergence_fsdp2_fp8(distributed_cleanup, tmp_path, recipe_path):
    """For FSDP2, we check that the script can run successfully with FP8 and check convergence."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "fp8_config.enabled=true",
                f"checkpoint.ckpt_dir={tmp_path}",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_thd(distributed_cleanup, tmp_path, monkeypatch, recipe_path):
    """For FSDP2, we check that the script can run successfully with FP8 and check convergence."""
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "dataset.use_sequence_packing=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


@requires_fp8
def test_sanity_convergence_fsdp2_thd_fp8(distributed_cleanup, tmp_path, monkeypatch, recipe_path):
    """For FSDP2, we check that the script can run successfully with THD + FP8 and check convergence."""
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "dataset.use_sequence_packing=true",
                "dataset.sequence_packing_pad_to_multiple_of=16",
                "fp8_config.enabled=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_ddp_thd(distributed_cleanup, tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    # For DDP, we only check that the script can run successfully with THD, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "dataset.use_sequence_packing=true",
                "num_train_steps=4",
            ],
        )

    main_ddp(sanity_config)


def test_sanity_mfsdp_thd(distributed_cleanup, tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    # For MFSDP, we only check that the script can run successfully with THD, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "dataset.use_sequence_packing=true",
                "num_train_steps=4",
            ],
        )

    main_mfsdp(sanity_config)


@requires_fp8
def test_sanity_ddp_thd_fp8(distributed_cleanup, tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    # For DDP, we only check that the script can run successfully with THD, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "dataset.use_sequence_packing=true",
                "num_train_steps=4",
                "fp8_config.enabled=true",
                "dataset.sequence_packing_pad_to_multiple_of=16",
                "checkpoint.resume_from_checkpoint=false",
            ],
        )

    main_ddp(sanity_config)


@requires_fp8
def test_sanity_mfsdp_thd_fp8(distributed_cleanup, tmp_path, monkeypatch, recipe_path):
    if torch.cuda.get_device_capability() == (12, 0):
        # TODO(BIONEMO-2840): On sm120, we need to set NVTE_FUSED_ATTN to 0 since TE will choose fused attn by default,
        # but it's missing this THD implementation.
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    # For MFSDP, we only check that the script can run successfully with THD, not convergence.
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "dataset.use_sequence_packing=true",
                "num_train_steps=4",
                "fp8_config.enabled=true",
                "dataset.sequence_packing_pad_to_multiple_of=16",
            ],
        )

    main_mfsdp(sanity_config)


@pytest.mark.xfail(reason="Meta-device init seems to be having some issues with convergence (BIONEMO-2719)")
def test_sanity_convergence_fsdp2_meta_device(distributed_cleanup, tmp_path, recipe_path):
    """Test that the main function can be invoked wrapping the model in FSDP2."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "use_meta_device=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_mfsdp_eager(distributed_cleanup, tmp_path, recipe_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "model_tag=facebook/esm2_t6_8M_UR50D",
            ],
        )

    final_loss = main_mfsdp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_ddp_eager(distributed_cleanup, tmp_path, recipe_path):
    """Test that the main function can be invoked wrapping the model in DDP."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "model_tag=facebook/esm2_t6_8M_UR50D",
                "checkpoint.resume_from_checkpoint=false",
            ],
        )

    final_loss = main_ddp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_eager(distributed_cleanup, tmp_path, recipe_path):
    """Test that the main function can be invoked wrapping the model in FSDP2."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "model_tag=facebook/esm2_t6_8M_UR50D",
                "checkpoint.resume_from_checkpoint=false",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


@pytest.mark.xfail(reason="Meta-device init seems to be having some issues with convergence (BIONEMO-2719)")
def test_sanity_convergence_fsdp2_eager_meta_device(distributed_cleanup, tmp_path, recipe_path):
    """Test that the main function can be invoked wrapping the model in FSDP2 and using meta-device init."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "model_tag=facebook/esm2_t6_8M_UR50D",
                "use_meta_device=true",
                "checkpoint.resume_from_checkpoint=false",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"
