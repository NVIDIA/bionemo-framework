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


from typing import Any, Dict, Literal

import torch
import torch.utils.data
from nemo.utils import logging


class DataloaderWithMode(torch.utils.data.DataLoader):
    """DataLoader wrapper that adds a `mode` attribute."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        mode: Literal["train", "valid", "test"],
        **kwargs: Any,
    ):
        """Initialize DataloaderWithMode.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to be fed into DataLoader.
            mode (str): Mode of the DataLoader to enable correct training resumption with datamodule state dict. Must be one of "train", "valid", or "test".
            **kwargs (Any): Keyword arguments to be passed to torch.utils.data.DataLoader.
        """
        super().__init__(dataset, **kwargs)

        if mode not in ["train", "valid", "test"]:
            raise ValueError(f"Mode must be one of 'train', 'valid', or 'test', but got {mode}")
        self.mode = mode


class DatamoduleMixin:
    """A mixin that adds a `state_dict` and `load_state_dict` method for datamodule training resumption in NeMo."""

    def __init__(self):
        """Set init_global_step to 0 for datamodule resumption."""
        self.init_global_step = 0

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        # TODO @sichu can silently fail and return None in Callbacks
        consumed_samples = self.data_sampler.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        return {"consumed_samples": consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule stat.

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """
        try:
            from megatron.core.num_microbatches_calculator import update_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import update_num_microbatches

        consumed_samples = state_dict["consumed_samples"]
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples

        update_num_microbatches(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        self.data_sampler.if_first_step = 1