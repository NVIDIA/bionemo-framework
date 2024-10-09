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


import os

import numpy as np
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import (
    get_samples_mapping,
)
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import CSVMemMapDataset
from nemo.core import Dataset


try:
    from apex.transformer.parallel_state import get_rank_info  # noqa: F401

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ["MoleculeCsvDataset"]


# TODO: use NeMoUpsampling instead of directly calling get_samples_mapping
class MoleculeCsvDataset(Dataset):
    """
    Allow per-line lazy access to multiple text files using numpy memmap.
    """

    def __init__(
        self,
        dataset_paths,
        cfg,
        workers=None,
        num_samples=None,
        index_mapping_dir=None,
    ):
        super().__init__()

        self.num_samples = num_samples
        self.seed = cfg.get("seed")
        # prefix for sample mapping cached indeices
        self.data_prefix = cfg.get("data_prefix")
        if not self.data_prefix:
            self.data_prefix = os.path.commonprefix(dataset_paths)
        self.max_seq_length = cfg.get("max_seq_length")

        self.ds = CSVMemMapDataset(
            dataset_paths=dataset_paths,
            newline_int=cfg.get("newline_int"),
            header_lines=cfg.get("header_lines"),  # skip first N lines
            workers=workers,
            tokenizer=None,
            sort_dataset_paths=cfg.get("sort_dataset_paths"),
            data_col=cfg.get("data_col"),
            data_sep=cfg.get("data_sep"),
            index_mapping_dir=index_mapping_dir,
        )
        # create mapping to a single epoch with num_samples
        self._build_samples_mapping()

    def _build_samples_mapping(self):
        # use Megatron up/down sampling if num_samples is given
        if self.num_samples is None:
            self.num_samples = len(self.ds)

        # map dataset samples to the desired num_samples
        self.samples_mapping = get_samples_mapping(
            indexed_dataset=self.ds,
            data_prefix=self.data_prefix,
            num_epochs=None,
            max_num_samples=self.num_samples,
            # account for <BOS> / <EOS>
            max_seq_length=self.max_seq_length - 2,
            short_seq_prob=0,
            seed=self.seed,
            name=self.data_prefix.split("/")[-1],
            binary_head=False,
        )

        self.samples_mapping = self.samples_mapping[: self.num_samples]

    def __len__(self):
        return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        idx, _, _ = self.samples_mapping[idx]
        if isinstance(idx, np.uint32):
            idx = idx.item()

        return self.ds[idx]
