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

import gc
import os

import torch
from lightning.pytorch import Callback


class _FirstBatchCudaSync(Callback):
    # TEMPORARY CALLBACK. Remove once bug is fixed.
    # First batch CUDA sync callback: adds barriers for the first training batch to avoid race condition
    # See https://github.com/NVIDIA/bionemo-framework/issues/1301 for more details.
    def __init__(self):
        self._done = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._done and torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_after_backward(self, trainer, pl_module):
        if not self._done and torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._done and torch.cuda.is_available():
            torch.cuda.synchronize()
            # Unset blocking for subsequent batches
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
            self._done = True


class GarbageCollectAtInferenceTime(Callback):
    """Callback to clean up CUDA memory before validation to prevent initialization errors."""

    def on_validation_start(self, trainer, pl_module) -> None:
        """Clean up CUDA memory before validation to prevent initialization errors."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                current_device = torch.cuda.current_device()
                torch.cuda.set_device(current_device)
                torch.cuda.synchronize()
                gc.collect()
            except Exception as e:
                print(f"Warning: CUDA cleanup failed: {e}")


class InitWeightStatsCallback(Callback):
    """Callback to log initialization weight statistics (mean, std) for all model layers.

    Logs weight statistics at the start of training before the first batch.
    Useful for comparing initialization between different frameworks (e.g., TE FSDP2 vs Megatron).

    Args:
        log_all_layers: If True, log all layers. If False, log only key layers.
    """

    def __init__(self, log_all_layers: bool = True):
        """Initialize the callback.

        Args:
            log_all_layers: If True, log all layers. If False, log only key layers.
        """
        super().__init__()
        self.log_all_layers = log_all_layers
        self._logged = False

    def on_train_start(self, trainer, pl_module) -> None:
        """Log weight initialization statistics at the start of training."""
        if self._logged:
            return
        self._logged = True

        rank = trainer.global_rank if hasattr(trainer, "global_rank") else 0

        if rank == 0:
            print("=" * 100)
            print("[INIT_WEIGHT_STATS] Logging weight initialization statistics for all layers")
            print("=" * 100)

        stats = {}
        model = pl_module

        # Handle wrapped models (e.g., Megatron models)
        if hasattr(model, "module"):
            model = model.module

        for name, param in model.named_parameters():
            if not self.log_all_layers:
                # Log only key layers for debugging initialization
                keys_to_log = [
                    "embed",
                    "output_layer",
                    "lm_head",
                    "o_proj",
                    "proj",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "fc1",
                    "fc2",
                    "layernorm",
                    "rmsnorm",
                    "input_layernorm",
                    "norm",
                    "linear_qkv",
                    "linear_proj",
                    "linear_fc1",
                    "linear_fc2",
                ]
                if not any(key in name.lower() for key in keys_to_log):
                    continue

            # Skip meta tensors
            if param.device.type == "meta":
                continue

            # Skip empty tensors
            if param.numel() == 0:
                continue

            # Compute stats
            try:
                with torch.no_grad():
                    data = param.data.float()
                    mean_val = data.mean().item()
                    var_val = torch.mean(torch.pow(data - mean_val, 2)).item()
                    std_val = var_val**0.5
                    min_val = data.min().item()
                    max_val = data.max().item()
            except (RuntimeError, AttributeError):
                continue

            layer_stats = {
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "shape": list(param.shape),
                "numel": param.numel(),
            }
            stats[name] = layer_stats

            if rank == 0:
                print(
                    f"[INIT_WEIGHT_STATS] {name}: mean={layer_stats['mean']:.6f}, "
                    f"std={layer_stats['std']:.6f}, range=[{layer_stats['min']:.6f}, {layer_stats['max']:.6f}], "
                    f"shape={layer_stats['shape']}"
                )

        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Format as human-readable (e.g., "7.8B", "125M")
        def format_params(n: int) -> str:
            if n >= 1e9:
                return f"{n / 1e9:.2f}B"
            elif n >= 1e6:
                return f"{n / 1e6:.2f}M"
            elif n >= 1e3:
                return f"{n / 1e3:.2f}K"
            return str(n)

        if rank == 0:
            print("=" * 100)
            print(f"[INIT_WEIGHT_STATS] Total parameters logged: {len(stats)}")
            print(f"[INIT_WEIGHT_STATS] Total model parameters: {total_params:,} ({format_params(total_params)})")
            print(
                f"[INIT_WEIGHT_STATS] Trainable parameters: {trainable_params:,} ({format_params(trainable_params)})"
            )
            print("=" * 100)
