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
import subprocess
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from torch.distributed.device_mesh import _mesh_resources

from train_ddp import main as main_ddp
from train_fsdp2 import main as main_fsdp2
from train_mfsdp import main as main_mfsdp


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

# Get the recipe directory
recipe_dir = Path(__file__).parent.parent


def run_train_cmd(cmd):
    """Run a training command and check for errors."""

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command:\n{' '.join(cmd)}\nfailed with exit code {result.returncode}")


@pytest.fixture
def distributed_cleanup():
    yield

    # Try to destroy the process group, but don't fail if it's not available.
    try:
        dist.destroy_process_group()
    except AssertionError:
        pass

    # For MFSDP, clear mesh resources to avoid issues re-running in the same process.
    _mesh_resources.mesh_stack.clear()
    _mesh_resources.child_to_root_mapping.clear()
    _mesh_resources.root_to_flatten_mapping.clear()
    _mesh_resources.flatten_name_to_root_dims.clear()
    _mesh_resources.mesh_dim_group_options.clear()


def test_sanity_convergence_mfsdp(distributed_cleanup, tmp_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(config_name="L0_sanity", overrides=[f"+wandb_init_args.dir={tmp_path}"])

    final_loss = main_mfsdp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_mfsdp_meta_device(distributed_cleanup, tmp_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "use_meta_device=true",
            ],
        )

    final_loss = main_mfsdp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_mfsdp_eager_meta_device(distributed_cleanup, tmp_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "model_tag=facebook/esm2_t6_8M_UR50D",
                "use_meta_device=true",
            ],
        )

    final_loss = main_mfsdp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_ddp(distributed_cleanup, tmp_path):
    """Test that the main function can be invoked wrapping the model in DDP."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(config_name="L0_sanity", overrides=[f"+wandb_init_args.dir={tmp_path}"])

    final_loss = main_ddp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_ddp_non_streaming_dataset(distributed_cleanup, tmp_path):
    """Test that the training script works with a non-streaming dataset."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "dataset.load_dataset_kwargs.streaming=False",
            ],
        )

    final_loss = main_ddp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2(distributed_cleanup, tmp_path):
    """Test that the main function can be invoked wrapping the model in FSDP2."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


@pytest.mark.xfail(reason="Meta-device init seems to be having some issues with convergence (BIONEMO-2719)")
def test_sanity_convergence_fsdp2_meta_device(distributed_cleanup, tmp_path):
    """Test that the main function can be invoked wrapping the model in FSDP2."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "use_meta_device=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_mfsdp_eager(distributed_cleanup, tmp_path):
    """Test that the main function can be invoked with the correct arguments."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[f"+wandb_init_args.dir={tmp_path}", "model_tag=facebook/esm2_t6_8M_UR50D"],
        )

    final_loss = main_mfsdp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_ddp_eager(distributed_cleanup, tmp_path):
    """Test that the main function can be invoked wrapping the model in DDP."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[f"+wandb_init_args.dir={tmp_path}", "model_tag=facebook/esm2_t6_8M_UR50D"],
        )

    final_loss = main_ddp(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_eager(distributed_cleanup, tmp_path):
    """Test that the main function can be invoked wrapping the model in FSDP2."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[f"+wandb_init_args.dir={tmp_path}", "model_tag=facebook/esm2_t6_8M_UR50D"],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_eager_meta_device(distributed_cleanup, tmp_path):
    """Test that the main function can be invoked wrapping the model in FSDP2 and using meta-device init."""

    # Run the training script with Hydra configuration overrides
    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "model_tag=facebook/esm2_t6_8M_UR50D",
                "use_meta_device=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 3.0, f"Final loss {final_loss} is too high"


# These tests don't check convergence, they just check that the training script runs successfully on multiple GPUs.


@requires_multi_gpu
def test_multi_gpu_train_te_ddp(tmp_path):
    # Run 'accelerate launch train.py' as a subprocess
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node",
            "2",
            "--master_port",
            f"{random.randint(20000, 40000)}",
            "train_ddp.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",
        ]
    )


@requires_multi_gpu
def test_multi_gpu_train_te_mfsdp(tmp_path):
    # Run 'accelerate launch train.py' as a subprocess
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node",
            "2",
            "--master_port",
            f"{random.randint(20000, 40000)}",
            "train_mfsdp.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",
        ]
    )


@requires_multi_gpu
def test_multi_gpu_train_te_fsdp2(tmp_path):
    # Run 'accelerate launch train.py' as a subprocess
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node",
            "2",
            "--master_port",
            f"{random.randint(20000, 40000)}",
            "train_fsdp2.py",
            "--config-name",
            "L0_sanity",
            "num_train_steps=4",
        ]
    )


@requires_multi_gpu
def test_multi_gpu_train_eager_fsdp2_meta_device(tmp_path):
    # Run 'accelerate launch train.py' as a subprocess
    run_train_cmd(
        [
            "torchrun",
            "--nproc_per_node",
            "2",
            "--master_port",
            f"{random.randint(20000, 40000)}",
            "train_fsdp2.py",
            "--config-name",
            "L0_sanity",
            "model_tag=facebook/esm2_t6_8M_UR50D",
            "use_meta_device=true",
            "num_train_steps=4",
        ]
    )
