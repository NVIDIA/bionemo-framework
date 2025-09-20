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

import logging
import time

import hydra
import torch
import transformer_engine.pytorch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.optim import AdamW
from tqdm import tqdm
from transformer_engine.common.recipe import Format
from transformers import AutoConfig, AutoModelForMaskedLM

# This import seems to be needed with meta device init and AutoModel.from_config
from transformers.models.esm.modeling_esm import EsmForMaskedLM  # noqa: F401

from dataset import create_dataloader
from distributed_config import DistributedConfig
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PerfLogger:
    """Logger for performance metrics."""

    def __init__(self):
        """Initialize the PerfLogger."""
        self.total_step_time_buffer = []
        self.model_step_time_buffer = []
        self.optimizer_step_time_buffer = []
        self.samples_per_second_buffer = []
        self.data_batch_load_time_buffer = []
        self.num_tokens_buffer = []
        self.num_sig_tokens_buffer = []
        self.tps_buffer = []
        self.sig_tps_buffer = []
        self.alloc_memory_gb_buffer = []
        self.reserv_memory_gb_buffer = []

    def log_step_metrics(
        self,
        step,
        micro_batch_size,
        seq_length,
        loss_value,
        total_norm,
        step_time,
        model_step_time,
        optimizer_step_time,
        data_batch_load_time,
        num_sig_tokens,
        current_lr,
    ):
        """Log metrics to logger and wandb on main process."""
        # Single-rank training progress and metrics.
        logger.info(
            f"Train: [Step {step:>2d}  "
            f"Loss: {loss_value:#.3g}  "
            f"Time: {step_time:.3f}s ({micro_batch_size / step_time:>5.2f} samples/sec) "
            f"TPS: {seq_length * micro_batch_size / step_time:.3f} tokens/sec  "
            f"Non-Pad TPS: {num_sig_tokens / step_time:.3f} tokens/sec  "
            f"Memory: {torch.cuda.memory.max_memory_reserved() / 1024**3:.3f} GB  "
            f"LR: {current_lr:.3e}  "
            f"Data Load Time: {data_batch_load_time:.3f}s"
        )

        # Performance metrics on a single/local rank.
        self.total_step_time_buffer.append(step_time)
        self.model_step_time_buffer.append(model_step_time)
        self.optimizer_step_time_buffer.append(optimizer_step_time)
        self.data_batch_load_time_buffer.append(data_batch_load_time)
        self.samples_per_second_buffer.append(micro_batch_size / step_time)
        self.num_tokens_buffer.append(seq_length * micro_batch_size)
        self.num_sig_tokens_buffer.append(num_sig_tokens)
        self.tps_buffer.append(seq_length * micro_batch_size / step_time)
        self.sig_tps_buffer.append(num_sig_tokens / step_time)
        self.alloc_memory_gb_buffer.append(torch.cuda.memory.memory_allocated() / 1024**3)
        self.reserv_memory_gb_buffer.append(torch.cuda.memory.memory_reserved() / 1024**3)
        wandb.log(
            {
                "train/loss": loss_value,
                "train/global_step": step,
                "train/learning_rate": current_lr,
                "train/grad_norm": total_norm,
                "train/perf/model_step_time": self.model_step_time_buffer[-1],
                "train/perf/total_step_time": self.total_step_time_buffer[-1],
                "train/perf/optimizer_step_time": self.optimizer_step_time_buffer[-1],
                "train/perf/data_batch_load_time": self.data_batch_load_time_buffer[-1],
                "train/perf/samples_per_second": self.samples_per_second_buffer[-1],
                "train/perf/num_tokens": self.num_tokens_buffer[-1],
                "train/perf/num_sig_tokens": self.num_sig_tokens_buffer[-1],
                "train/perf/tps": self.tps_buffer[-1],
                "train/perf/sig_tps": self.sig_tps_buffer[-1],
                "train/perf/alloc_memory_gb": self.alloc_memory_gb_buffer[-1],
                "train/perf/reserv_memory_gb": self.reserv_memory_gb_buffer[-1],
            }
        )

    def log_average_metrics(self):
        """Log average metrics to logger and wandb on main process."""
        wandb.log(
            {
                "perf/avg_step_time_in_seconds": sum(self.total_step_time_buffer) / len(self.total_step_time_buffer),
                "perf/avg_model_step_time": sum(self.model_step_time_buffer) / len(self.model_step_time_buffer),
                "perf/avg_optimizer_step_time": sum(self.optimizer_step_time_buffer)
                / len(self.optimizer_step_time_buffer),
                "perf/avg_data_batch_load_time": sum(self.data_batch_load_time_buffer)
                / len(self.data_batch_load_time_buffer),
                "perf/avg_samples_per_second": sum(self.samples_per_second_buffer)
                / len(self.samples_per_second_buffer),
                "perf/avg_num_tokens": sum(self.num_tokens_buffer) / len(self.num_tokens_buffer),
                "perf/avg_num_sig_tokens": sum(self.num_sig_tokens_buffer) / len(self.num_sig_tokens_buffer),
                "perf/avg_tps": sum(self.tps_buffer) / len(self.tps_buffer),
                "perf/avg_sig_tps": sum(self.sig_tps_buffer) / len(self.sig_tps_buffer),
                "perf/avg_alloc_memory_gb": sum(self.alloc_memory_gb_buffer) / len(self.alloc_memory_gb_buffer),
                "perf/avg_reserv_memory_gb": sum(self.reserv_memory_gb_buffer) / len(self.reserv_memory_gb_buffer),
            }
        )


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train ESM-2 with TE layers using fsdp2.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(dist_config.local_rank)

    # Create a device mesh for FSDP.
    # We have to create a dummy mesh dimension for context parallel and tensor parallel for things
    # to work correctly with fsdp2.
    device = torch.device(f"cuda:{dist_config.local_rank}")
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size, 1, 1),
        mesh_dim_names=("fsdp", "cp", "tp"),
    )

    if dist_config.is_main_process():
        wandb.init(**args.wandb_init_args, config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True))

    # Create an empty ESM-2 model with a masked language model head.
    config = AutoConfig.from_pretrained(args.model_tag, trust_remote_code=True, dtype=torch.bfloat16)
    # If we're using sequence packing with TE layers, we need to pass the `attn_input_format` argument.
    if args.dataset.use_sequence_packing:
        config.attn_input_format = "thd"
    if "facebook" in args.model_tag and args.attn_backend is not None:
        config._attn_implementation = args.attn_backend
    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    # The huggingface model has a contact head that we don't use in masked language pre-training, so we delete it to
    # avoid errors with unused parameters.
    try:
        del model.esm.contact_head
    except AttributeError:
        pass

    # We call the transformer stack "layers" in our TE models, but it's called "layer" in the original ESM-2 models.
    transformer_stack = model.esm.encoder.layers if hasattr(model.esm.encoder, "layers") else model.esm.encoder.layer
    for layer in transformer_stack:
        fully_shard(layer, mesh=device_mesh["fsdp"])
    fully_shard(model, mesh=device_mesh["fsdp"])

    # Log model and number of parameters on main process.
    if dist_config.is_main_process():
        logger.info("model:\n%s", model)
        logger.info(f"total number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer.
    optimizer = AdamW(model.parameters(), **args.adamw_kwargs)
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_meta_device:
        model.to_empty(device=device)
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    # Create an FP8 recipe
    if args.fp8_config.enabled:
        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )
    else:
        fp8_recipe = None

    # Create a dataloader that just infinitely loops over the dataset.
    train_iterator = create_dataloader(dist_config, **args.dataset)

    # Training loop.
    model.train()
    if dist_config.is_main_process():
        progress_bar = tqdm(range(args.num_train_steps), desc="Training", disable=False)

    # Training loop + performance metrics.
    perf_logger = PerfLogger()
    previous_step_time = time.perf_counter()
    for step in range(args.num_train_steps):
        # Get batch.
        batch = next(train_iterator)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Calculate number of non-padded tokens for THD comparison.
        num_sig_tokens = batch["input_ids"][batch["input_ids"] != 1].shape[0]

        pre_forward_backward_time = time.perf_counter()
        data_batch_load_time = pre_forward_backward_time - previous_step_time

        # Forward pass with mixed precision.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with transformer_engine.pytorch.fp8_autocast(enabled=args.fp8_config.enabled, fp8_recipe=fp8_recipe):
                outputs = model(**batch)

        # Backward pass.
        loss = outputs.loss
        loss.backward()

        post_forward_backward_time = time.perf_counter()
        model_step_time = post_forward_backward_time - pre_forward_backward_time

        # Compute and clip gradient norms.
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

        # Step optimizer.
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        post_optimizer_step_time = time.perf_counter()
        optimizer_step_time = post_optimizer_step_time - post_forward_backward_time

        # Log metrics to logger and wandb on main process.
        loss_value = loss.detach().item()
        if dist_config.is_main_process():
            current_time = time.perf_counter()
            step_time = current_time - previous_step_time
            previous_step_time = current_time
            current_lr = optimizer.param_groups[0]["lr"]

            # Rank 0 training progress and metrics.
            perf_logger.log_step_metrics(
                step,
                args.dataset.micro_batch_size,
                args.dataset.max_seq_length,
                loss_value,
                total_norm,
                step_time,
                model_step_time,
                optimizer_step_time,
                data_batch_load_time,
                num_sig_tokens,
                current_lr,
            )

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss_value})

    # Log average performance metrics.
    if dist_config.is_main_process():
        # Log average performance metrics.
        perf_logger.log_average_metrics()

        # Report Torch memory profile.
        torch_memory_profiler_snapshot = torch.cuda.memory._snapshot()
        from pathlib import Path
        from pickle import dump

        from hydra.core.hydra_config import HydraConfig

        with open(
            # Path will only exist when using @hydra.main()!
            Path(HydraConfig.get().runtime.output_dir)
            / f"{args.wandb_init_args.name.replace('/', '_')}_torch_memory_profiler_snapshot.pickle",
            "wb",
        ) as f:
            dump(torch_memory_profiler_snapshot, f)

    # Clean up distributed training
    if dist_config.is_main_process():
        wandb.finish()

    torch.distributed.destroy_process_group()

    return loss_value


if __name__ == "__main__":
    torch.cuda.memory._record_memory_history(max_entries=250000)
    main()
