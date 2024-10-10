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


import glob
import json
import os

import polars as pl
import torch
import webdataset as wds
from more_itertools import peekable

from .genome_interval import GenomeIntervalDataset


ATLAS_NAMES = {"human": "hg38.ml.fa", "mouse": "mm10.ml.fa"}


class WebBasenji(wds.pipeline.DataPipeline, wds.compat.FluidInterface):
    def __init__(
        self,
        dataset_path: str,
        subset: str,
        organism: str,
        shuffle: bool = False,
        context_length: int = 196_608,
        batch_size: int = 1,
        repeat=False,
    ):
        super().__init__()
        self.base_path = os.path.join(dataset_path, organism)
        self.organism = organism
        urls = glob.glob(os.path.join(self.base_path, f"{subset}-*.tar"))
        assert subset in {"train", "valid", "test"}

        if subset == "train":
            return_augs = True
            rc_aug = True
            shift_augs = (-3, 3)
        else:
            return_augs = True
            rc_aug = False
            shift_augs = (0, 0)
        nodesplitter = wds.shardlists.split_by_node
        self.dataset = wds.WebDataset(urls, shardshuffle=shuffle, nodesplitter=nodesplitter)
        self.dataset = self.dataset.decode().rename(target="target.pth")
        if shuffle:
            self.dataset = self.dataset.shuffle(20)
        if repeat:
            self.dataset = self.dataset.repeat()

        self.extended_ds = GenomeIntervalDataset(
            bed_file=os.path.join(self.base_path, "sequences.bed"),
            fasta_file=os.path.join(self.base_path, ATLAS_NAMES[organism]),
            filter_df_fn=lambda df: df.filter(pl.col("column_4") == subset),
            context_length=context_length,
            return_augs=return_augs,
            shift_augs=shift_augs,
            rc_aug=rc_aug,
        )

        def collate_basenji(batch):
            batch_cl = torch.utils.data.default_collate(batch)
            return {"head_name": self.organism, **batch_cl}

        self.dataset = self.dataset.map(self._extend)
        self.dataset = self.dataset.batched(batch_size, collation_fn=collate_basenji)
        self.metadata = self._get_metadata()

    def _get_metadata(self):
        # Keys:
        # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
        # pool_width, crop_bp, target_length
        path = os.path.join(self.base_path, "statistics.json")
        with open(path, "r") as f:
            return json.load(f)

    def _extend(self, sample):
        key = int(sample["__key__"])
        ex_sequence, _, rc_aug = self.extended_ds[key]
        target = torch.tensor(sample["target"])
        if rc_aug:
            target = torch.flipud(target)
        return {"sequence": ex_sequence, "target": target.float()}

    def __iter__(self):
        it = iter(self.dataset)
        p = peekable(it)
        # pytorch lightning performs all_gather before {stage}_epoch_end to pass down results.
        # This means every worker must iterate over something to create result that can be
        # aggregated across processes. Here, we check if a process was assigned any shard
        if p:
            return it
        else:
            return iter([None])


class WebCombinedBasenji(torch.utils.data.IterableDataset):
    def __init__(self, dataset_path: str, context_length: int = 196_608, batch_size: int = 1):
        self.human_dataset = WebBasenji(
            dataset_path, "train", "human", repeat=True, context_length=context_length, batch_size=batch_size
        )
        self.mouse_dataset = WebBasenji(
            dataset_path, "train", "mouse", repeat=True, context_length=context_length, batch_size=batch_size
        )

    def __iter__(self):
        for human_record, mouse_record in zip(self.human_dataset, self.mouse_dataset):
            yield human_record
            yield mouse_record
