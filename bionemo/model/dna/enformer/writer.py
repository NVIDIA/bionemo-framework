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


import itertools
import os
from typing import Literal

import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class FastaRecordsWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, mode: Literal["sep", "all"] = "sep"):
        """_summary_

        Args:
            output_dir (str): where predictions should be stored
            mode (Literal['sep', 'all']): whether each record should be written
            separately ('sep') as a file  with name of the {record_name}.pt or
            ('all') predictions from a process should be written to a single file

            Each prediction must be a dict with keys: name, preds
        """
        super().__init__("epoch")
        self.output_dir = output_dir
        self.mode = mode

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = list(itertools.chain.from_iterable(predictions))
        names = list(itertools.chain(*(rec["name"] for rec in predictions)))
        preds = torch.cat([rec["pred"] for rec in predictions])

        if self.mode == "sep":
            for name, pred in zip(names, preds):
                torch.save(pred, os.path.join(self.output_dir, f"{name}.pt"))
        else:
            torch.save(
                {"names": names, "preds": preds},
                os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
            )
