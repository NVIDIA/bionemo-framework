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

from typing import List, Optional

import torch
from lightning.pytorch.trainer.states import TrainerFn
from nemo.collections.llm.peft.lora import LoRA
from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.lightning.pytorch.utils import is_trainer_attached

from nemo.utils import logging
from torch import nn


class Evo2LoRA(LoRA):
    """LoRA adapter specifically for Evo2/Hyena models."""

    def __init__(
        self,
        peft_ckpt_path: Optional[str] = None,
        skip_freeze_modules: List[str] = ["word_embeddings"], # This modules won't be frozen
        target_modules: List[str] = [
            "linear_qkv", # Belonging to Transformer Layer
            "linear_proj", # Belonging to Transformer Layer
            "linear_fc1", # Belonging to both HyenaLayer and Transformer Layer
            "linear_fc2", # Belonging to both HyenaLayer and Transformer Layer
            "dense_projection", # Belonging to HyenaLayer Layer
        ],
        *args,
        **kwargs,
    ):
        """Initialize the LoRA Adapter for Evo2.

        Args:
            peft_ckpt_path: Path to pre-trained LoRA checkpoint.
            skip_freeze_modules: List of module names to skip freezing (Evo2-specific defaults).
            target_modules: Modules to apply LoRA to (uses Evo2 defaults if None).
            *args: placeholder.
            **kwargs:
                dim: LoRA rank dimension.
                alpha: LoRA scaling parameter.
                dropout: Dropout rate for LoRA layers.
                dropout_position: Where to apply dropout ('pre' or 'post').
                lora_A_init_method: Initialization for A matrix ('xavier', 'uniform', 'normal').
                lora_B_init_method: Initialization for B matrix ('zero', 'normal').
        """
        super().__init__(target_modules=target_modules, *args, **kwargs)
        self.skip_freeze_modules = skip_freeze_modules
        self.peft_ckpt_path = peft_ckpt_path

        # CRITICAL: Set model_transform to self
        # The callback system expects this attribute
        self.model_transform = self

    def setup(self, *args, **kwargs):
        """Setup callback - properly initialize transform."""
        super().setup(*args, **kwargs)

        logging.info(f"Evo2LoRA: Will attempt to apply to model if matches: \n{self.target_modules}")
        # Ensure model_transform is set
        if not hasattr(self, "model_transform") or self.model_transform is None:
            self.model_transform = self

        # Pass checkpoint path to wrapped IO if available
        if hasattr(self, "wrapped_io") and self.peft_ckpt_path:
            self.wrapped_io.adapter_ckpt_path = self.peft_ckpt_path

    def _evo2_selective_freeze(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not any(skip_freeze in name for skip_freeze in self.skip_freeze_modules):
                param.requires_grad = False
            else:
                logging.info(f"Evo2LoRA: Skipping freezing module: {name}.")

        model.eval()

    def freeze_model(self, model: nn.Module):
        """Inspired on PEFT.freeze_model() from NeMo.

        Github: https://github.com/NVIDIA-NeMo/NeMo/blob/6ec88a94af15ba14e38cb3d68f04542e81ced995/nemo/lightning/pytorch/callbacks/peft.py#L114
        """
        # If trainer is not in FITTING mode, all parameters are frozen using super() implementation
        if is_trainer_attached(model) and model.trainer.state.fn != TrainerFn.FITTING:
            super().freeze_model(model)
            return

        if isinstance(model, MegatronParallel) and len(model) > 1:
            for model_chunk in model:
                self._evo2_selective_freeze(model_chunk)
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            self._evo2_selective_freeze(model.module)
        else:
            self._evo2_selective_freeze(model)

        if is_trainer_attached(model) and model.trainer.state.fn == TrainerFn.FITTING:
            model.train(mode=True)
