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


import torch
from nemo.lightning import io
from nemo.utils import logging
from pytorch_lightning.callbacks.callback import Callback


class MemoryCleanupCallback(Callback, io.IOMixin):
    """Class to print out memory usage at the end of each training batch."""

    def __init__(self):
        """Initialize the memory usage list."""
        self.memory_usage = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:  # noqa: D102
        # gc.collect()
        # torch.cuda.empty_cache()

        self.memory_usage.append((batch_idx, torch.cuda.memory_allocated(), torch.cuda.max_memory_reserved()))

        logging.info(
            f"on_train_batch_end {batch_idx} mem: {torch.cuda.memory_allocated()/1024/1024/1024} /"
            f"{torch.cuda.max_memory_reserved()/1024/1024/1024}"
        )
