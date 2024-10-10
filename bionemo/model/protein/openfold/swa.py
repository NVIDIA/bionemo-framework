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


import contextlib
import os
from typing import Any, Dict, Iterable, Tuple

import pytorch_lightning as pl
import torch
from nemo.collections.common.callbacks import EMA
from torch import Tensor  # noqa

from bionemo.model.protein.openfold.triton.fused_adam_swa import FusedAdamSWA
from bionemo.utils.logging_utils import (
    log_with_nemo_at_level,
)


def all_parameters(param_groups: Iterable[Dict]) -> Tuple[Tensor]:
    return (param for group in param_groups for param in group["params"])


class AlphaFoldEMA(EMA):
    """Manages the swapping of model weights with SWA weights, and the loading
    of the SWA checkpoint.

    The computation of averaged model weights occurs in FusedAdamSWA, see
    the function _swa_math(..).

    See parent class at
    https://github.com/NVIDIA/NeMo/blob/1c0bef011eb5b58a6fae76f1ae60cc94bf9b0bbb/nemo/collections/common/callbacks/ema.py#L82

    The child class AlphaFoldEMA does not have a use-case for
    self.validate_original_weights = True, so it is omitted from __init__,
    and hard-code to False, for usage in _should_validate_ema_weights(..).

    The method _should_validate_ema_weights(..) is needed to gate the
    swapping of model weights that occurs in hooks on_validation_start(..)
    and on_validation_end(..).  These methods are not explicity written
    below, and therefore inherit from the parent EMA.
    """

    def __init__(self) -> None:
        """Do not need decay, since there is no EMAOptimizer"""
        # self.swa_params = None        # ToDo.
        self.validate_original_weights = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        log_with_nemo_at_level("""AlphaFoldEMA.setup, begin""")
        # if stage == 'fit' and self.swa_params is None:
        # self.swa_params = [
        #    torch.zeros_like(t) for t in deepcopy(list(pl_module.parameters()))
        # ]  # t.to(dtype=torch.float32)
        log_with_nemo_at_level("""AlphaFoldEMA.setup, end""")

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        log_with_nemo_at_level("""AlphaFoldEMA.on_fit_start, begin""")

        # sanity checks
        assert (
            len(trainer.optimizers) == 1
            and len(trainer.optimizers[0].param_groups) == 1
            and len(trainer.optimizers[0].swa_param_groups) == 1
        )

        log_with_nemo_at_level("""AlphaFoldEMA.on_fit_start, end""")

    def _ema_initialized(self, trainer: "pl.Trainer") -> bool:
        """We consider this instance initialized of there is a FusedAdamSWA instance
        amongst trainer.optimzers.
        """
        return any(isinstance(optimizer, FusedAdamSWA) for optimizer in trainer.optimizers)

    def swap_model_weights(self, trainer: "pl.Trainer") -> None:
        """Swap the values in optimizer.swa_param_groups[0]["params"]
        with optimizer.param_groups[0]["params"].

        Note:
            only one optimizers will always be present, as asserted in on_fit_start
        """
        log_with_nemo_at_level("""AlphaFoldEMA.swap_model_weights, begin""")

        optimizer = trainer.optimizers[0]
        for src_param, dst_param in zip(optimizer.swa_param_groups[0]["params"], optimizer.param_groups[0]["params"]):
            swap_tensor_values(src_param, dst_param)

        log_with_nemo_at_level("""AlphaFoldEMA.swap_model_weights, end""")

    @contextlib.contextmanager
    def save_ema_model(self, trainer: "pl.Trainer"):
        """
        Saves an EMA copy of the model + EMA optimizer states for resume.
        """
        log_with_nemo_at_level("""AlphaFoldEMA.save_ema_model, begin""")
        self.swap_model_weights(trainer)
        try:
            yield
        finally:
            self.swap_model_weights(trainer)
            log_with_nemo_at_level("""AlphaFoldEMA.save_ema_model, after second swap_model_weights""")

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: "pl.Trainer"):
        yield
        # for optimizer in trainer.optimizers:
        #     assert isinstance(optimizer, EMAOptimizer)
        #     optimizer.save_original_optimizer_state = True
        # try:
        #     yield
        # finally:
        #     for optimizer in trainer.optimizers:
        #         optimizer.save_original_optimizer_state = False

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        """
        - use the connector as NeMo calls the connector directly in the exp_manager when restoring.
        - Replace connector._ckpt_path with below to avoid calling into lightning's protected API
        - Updates checkpoint argument.
        Args:
            trainer:
            pl_module:
            checkpoint
        """
        log_with_nemo_at_level("""AlphaFoldEMA.on_load_checkpoint, begin""")
        checkpoint_callback = trainer.checkpoint_callback

        ckpt_path = trainer.ckpt_path
        if ckpt_path and checkpoint_callback is not None and "NeMo" in type(checkpoint_callback).__name__:
            ext = checkpoint_callback.FILE_EXTENSION
            if ckpt_path.endswith(f"-EMA{ext}"):
                print(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = ckpt_path.replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))
                checkpoint["optimizer_states"] = ema_state_dict["optimizer_states"]
                del ema_state_dict
                print("EMA state has been restored.")
            else:
                raise Exception(
                    "Unable to find the associated EMA weights when re-loading, "
                    f"training will start with new EMA weights. Expected them to be at: {ema_path}",
                )
        log_with_nemo_at_level("""AlphaFoldEMA.on_load_checkpoint, end""")


def swap_tensor_values(tensor1: Tensor, tensor2: Tensor) -> None:
    # See https://github.com/NVIDIA/NeMo/blob/95ca2f45034447ecd11bf29a0ab55d9079133db1/nemo/collections/common/callbacks/ema.py#L279C5-L283C27
    # https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/blob/42bd384e91af3e55e170188e414bf11f9a0fc1e9/bionemo/model/protein/openfold/swa.py#L56
    tmp = torch.empty_like(tensor1)
    tmp.copy_(tensor1)
    tensor1.copy_(tensor2)
    tensor2.copy_(tmp)
    # dst_param.detach().copy_(src_param.to(dst_param.device))
