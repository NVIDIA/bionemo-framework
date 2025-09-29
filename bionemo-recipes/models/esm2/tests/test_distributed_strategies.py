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

import argparse
import logging
import os
import subprocess

import pytest
import torch
from transformers import DataCollatorForLanguageModeling


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


@pytest.mark.parametrize(
    "strategy",
    [
        "fsdp2",
        "mfsdp",
    ],
)
@pytest.mark.parametrize("backend", ["te", "eager"])
def test_ddp_vs_fsdp_single_gpu(strategy, backend):
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        os.path.relpath(__file__),
        "--strategy",
        strategy,
    ]
    if backend == "te":
        cmd.append("--test_te")

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
        pytest.fail(f"Command failed with exit code {result.returncode}")


@requires_multi_gpu
@pytest.mark.parametrize("strategy", ["fsdp2", pytest.param("mfsdp", marks=pytest.mark.xfail(reason="BIONEMO-2726"))])
@pytest.mark.parametrize("backend", ["te", "eager"])
def test_ddp_vs_fsdp_multi_gpu(strategy, backend):
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
        "--strategy",
        strategy,
    ]
    if backend == "te":
        cmd.append("--test_te")

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
        pytest.fail(f"Command failed with exit code {result.returncode}")


if __name__ == "__main__":
    import argparse
    import enum
    from dataclasses import dataclass, field

    import torch.distributed as dist
    import transformer_engine.pytorch
    import transformers
    from megatron_fsdp.fully_shard import fully_shard as megatron_fsdp_fully_shard
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.optim import AdamW
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    class Strategy(enum.StrEnum):
        DDP = "ddp"
        FSDP2 = "fsdp2"
        MFSDP = "mfsdp"

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_te", action="store_true", default=False)
    parser.add_argument("--strategy", type=Strategy, default=Strategy.FSDP2, choices=[Strategy.FSDP2, Strategy.MFSDP])
    args = parser.parse_args()

    from conftest import get_input_data

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=1024,
        seed=42,
    )

    input_data = get_input_data(tokenizer, data_collator)

    @dataclass
    class DistributedConfig:
        """Class to track distributed ranks."""

        rank: int = field(default_factory=dist.get_rank)
        local_rank: int = field(default_factory=lambda: int(os.environ["LOCAL_RANK"]))
        world_size: int = field(default_factory=dist.get_world_size)

        def is_main_process(self) -> bool:
            """This is the global rank 0 process, to be used for wandb logging, etc."""
            return self.rank == 0

    def run_forward_backward(use_te: bool, strategy: Strategy, input_data: dict, dist_config: DistributedConfig):
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(dist_config.world_size,),
            mesh_dim_names=("dp",),
        )

        device = f"cuda:{dist_config.local_rank}"

        if use_te:
            model = AutoModelForMaskedLM.from_pretrained(
                "nvidia/esm2_t6_8M_UR50D",
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            transformer_layers = model.esm.encoder.layers
        else:
            model = AutoModelForMaskedLM.from_pretrained(
                "facebook/esm2_t6_8M_UR50D",
                dtype=torch.bfloat16,
            )
            transformer_layers = model.esm.encoder.layer
            del model.esm.contact_head  # Unused in backwards pass.

        if strategy is Strategy.FSDP2:
            for layer in transformer_layers:
                fully_shard(layer, mesh=device_mesh["dp"])
            fully_shard(model, mesh=device_mesh["dp"])
            model.to(device)

        elif strategy is Strategy.DDP:
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist_config.local_rank],
                output_device=dist_config.local_rank,
                device_mesh=device_mesh["dp"],
            )

        optimizer = AdamW(model.parameters())

        if strategy is Strategy.MFSDP:
            model, optimizer = megatron_fsdp_fully_shard(
                module=model,
                optimizer=optimizer,
                fsdp_unit_modules=[
                    transformer_engine.pytorch.TransformerLayer,
                    transformer_engine.pytorch.LayerNorm,
                    transformer_engine.pytorch.LayerNormLinear,
                    transformers.models.esm.modeling_esm.EsmLayer,
                ],
                device_mesh=device_mesh,
                dp_shard_dim="dp",
                tp_dim="tp",
                sync_grads_each_step=True,
                preserve_fp32_weights=False,  # TODO: cory, any idea why this is needed?
            )

        model.train()
        input_data = {k: v.to(device) for k, v in input_data.items()}

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**input_data)
        outputs.loss.backward()

        # get gradients
        if strategy is Strategy.FSDP2:
            grads = {name: p.grad.full_tensor() for name, p in model.named_parameters() if p.grad is not None}

        elif strategy is Strategy.DDP:
            grads = {name: p.grad for name, p in model.module.named_parameters() if p.grad is not None}

        elif strategy is Strategy.MFSDP:
            # Because of uneven sharding, we need to manually gather the gradients.
            sharded_grads = [(name, p.grad) for name, p in model.module.named_parameters()]
            grads = {}
            for name, grad in sharded_grads:
                grad_shards = [None] * device_mesh["dp"].size()
                # For FSDP, we are not strided sharding, so gathering across dp_shard_cp is sufficient.
                # For HSDP, we need to first gather across dp_shard_cp, then gather across dp_inter,
                # not the other way around or you'll get wrong zig-zags.
                torch.distributed.all_gather_object(grad_shards, grad, group=device_mesh["dp"].get_group())
                all_valid_shards = [shard for shard in grad_shards if shard is not None]
                # Megatron-FSDP is always sharded across dim=0.
                grads[name] = torch.cat([s.to_local().to(device) for s in all_valid_shards], dim=0)

        del model
        torch.cuda.empty_cache()
        return outputs, grads

    dist.init_process_group(backend="nccl")
    dist_config = DistributedConfig()
    logger.info(f"Distributed config: {dist_config}")
    torch.cuda.set_device(dist_config.local_rank)

    ddp, ddp_grads = run_forward_backward(
        use_te=args.test_te, strategy=Strategy.DDP, input_data=input_data, dist_config=dist_config
    )

    fsdp, fsdp_grads = run_forward_backward(
        use_te=args.test_te, strategy=args.strategy, input_data=input_data, dist_config=dist_config
    )

    torch.testing.assert_close(fsdp.loss, ddp.loss, msg=lambda x: f"Loss mismatch: {x}")
    torch.testing.assert_close(fsdp.logits, ddp.logits, msg=lambda x: f"Logits mismatch: {x}")

    shared_grads = set(ddp_grads) & set(fsdp_grads)
    missing_grads = set(ddp_grads) ^ set(fsdp_grads)

    assert not missing_grads, f"Missing gradients: {missing_grads}"

    for name in shared_grads:
        ddp_grad = ddp_grads[name]
        fsdp_grad = fsdp_grads[name]
        torch.testing.assert_close(ddp_grad, fsdp_grad, msg=lambda x: f"Gradient mismatch for {name}: {x}")

        # Check that the gradients are different when the last dimension is shuffled
        assert not torch.allclose(ddp_grad, torch.roll(fsdp_grad, -1, -1))

    dist.destroy_process_group()
