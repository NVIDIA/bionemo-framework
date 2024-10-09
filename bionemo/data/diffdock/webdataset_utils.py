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
import pickle
from typing import Any, Callable, Generator, List, Optional

import webdataset as wds
from nemo.utils import logging
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch_geometric.data import HeteroData
from torch_geometric.data.batch import Batch
from torch_geometric.loader.dataloader import Collater

from bionemo.utils.logging import warn_once


@rank_zero_only
def pickles_to_tars(
    dir_input: str,
    input_suffix: str,
    input_prefix_subset: set,
    dir_output: str,
    output_prefix: str,
    func_output_data: Callable = lambda data: {"data": pickle.dumps(data)},
    min_num_shards: Optional[int] = None,
) -> None:
    """Convert a subset of pickle files from a directory to Webdataset tar files
    Input path and name pattern:
    f"{dir_input}/{input_prefix_subset}.{input_suffix}"
    Output path and name pattern:
    f"{dir_output}/{output_prefix}-%06d.tar"

    Args:
        dir_input (str): Input directory
        input_suffix (str): Input pickle file name suffix
        input_prefix_subset (set): Input subset of pickle files' prefix
        dir_output (str): Output directory
        output_prefix (str): Output tar file name prefix
        func_output_data (Callable) : function that maps data to a dictionary
        to be output in the tar files
        min_num_shards (int) : create at least this number of tar files.
        WebDataset has bugs when reading small number of tar files in a
        multi-node lightening + DDP setting so this option can be used to
        guarantee the tar file counts

    Returns: None

    """
    os.makedirs(dir_output, exist_ok=True)
    wd_subset_pattern = os.path.join(dir_output, f"{output_prefix}-%06d.tar")
    maxsize = 1e8
    # Due to a Webdataset bug, number of shards should be >= number of workers
    # (num. of gpus * num. of workers per gpu)
    if min_num_shards is not None and min_num_shards > 1:
        total_size = 0
        for name in input_prefix_subset:
            try:
                total_size += os.stat(os.path.join(dir_input, f"{name}.{input_suffix}")).st_size
            except Exception:
                continue
        maxsize = min(total_size // min_num_shards, maxsize)
    with wds.ShardWriter(wd_subset_pattern, encoder=False, maxsize=maxsize, compress=False, mode=0o777) as sink:
        for name in input_prefix_subset:
            try:
                data = pickle.load(open(os.path.join(dir_input, f"{name}.{input_suffix}"), "rb"))
                sample = func_output_data(data)
            except Exception as e:
                logging.error(f"Failed to write {name} into tar files due to error {e}")
                continue

            sink.write(sample)


class SizeAwareBatching:
    """A WebDataset composable to do batching based on sample size"""

    def __init__(
        self,
        max_total_size: int,
        size_fn: Callable[[HeteroData], int],
        collate_fn: Callable[[List[Any]], Any] = Collater(dataset=None, follow_batch=None, exclude_keys=None),
    ):
        self.max_total_size = max_total_size
        self.size_fn = size_fn
        self.collate_fn = collate_fn
        self.cached_sizes = {}

    def __call__(self, data: Batch) -> Generator[Batch, None, None]:
        batch_size = 0
        batch = []

        for sample in data:
            if sample.name not in self.cached_sizes:
                self.cached_sizes[sample.name] = self.size_fn(sample)
            sample_size = self.cached_sizes[sample.name]
            if sample_size > self.max_total_size:
                warn_once(f"sample {sample.name} has size larger than max size {self.max_total_size}, skipping")
                continue
            if (batch_size + sample_size) <= self.max_total_size:
                batch.append(sample)
                batch_size += sample_size
            else:
                if self.collate_fn is not None:
                    batch = self.collate_fn(batch)
                yield batch

                batch = [sample]
                batch_size = sample_size
