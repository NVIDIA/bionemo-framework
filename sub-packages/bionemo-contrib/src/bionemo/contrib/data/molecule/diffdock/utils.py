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
import random
from typing import Any, Callable, Generator, List, Optional
from copy import deepcopy

from nemo.utils import logging
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.batch import Batch
from torch_geometric.loader.dataloader import Collater
from torch_geometric.transforms import BaseTransform
import numpy as np

import webdataset as wds

from bionemo.contrib.model.molecule.diffdock.utils.diffusion import modify_conformer, set_time
from bionemo.contrib.model.molecule.diffdock.utils import so3, torus


def pickles_to_tars(
    dir_input: str,
    input_suffix: str,
    input_prefix_subset: List[str],
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
        input_prefix_subset (List[str]): Input subset of pickle files' prefix
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
    # TODO: this algorithm is not accurate enough because it doesn't take into
    # account the block structure so I have to multiply the total_size with a
    # small prefactor to purposely underestimate the size so that it ends up
    # creating more tar files than min_num_shards
    if min_num_shards is not None and min_num_shards > 1:
        total_size = 0
        for name in input_prefix_subset:
            try:
                total_size += os.stat(os.path.join(dir_input, f"{name}.{input_suffix}")).st_size
            except Exception:
                continue
        maxsize = min(total_size * 0.6 // min_num_shards, maxsize)
    with wds.ShardWriter(wd_subset_pattern, encoder=False, maxsize=maxsize, compress=False, mode=0o777) as sink:
        for name in input_prefix_subset:
            try:
                data = pickle.load(open(os.path.join(dir_input, f"{name}.{input_suffix}"), "rb"))
                sample = func_output_data(data)
            except ModuleNotFoundError as e:
                logging.error(f"Dependency for parsing input pickle data not "\
                              f"found: {e}")
                raise e
            except Exception as e:
                logging.error(f"Failed to write {name} into tar files due to error {e}")
                continue

            sink.write(sample)


def num_cross_edge_upper_bound_estimate(n1, n2, n3, n4):
    terms = [[4.92, 'ligand_ligand'],
             [0.0118, 'receptor_receptor'],
             [0.0401, 'ligand', 'receptor_receptor']]
    scale = 1.03
    tmpdict = {"ligand": n1, "ligand_ligand": n2, "receptor": n3, "receptor_receptor": n4}
    num_edges = 0.0
    for term in terms:
        tmp = term[0]
        for k in term[1:]:
            tmp *= tmpdict[k]
        num_edges += tmp
    num_edges *= scale
    return num_edges


def estimate_memory_usage(data, num_cross_edges, use_bias=True):
    # bias is from the memory of model, so when estimate the upper bound for size aware batch sampler, we don't need this
    coeff_ligand_num_nodes = 2.9
    coeff_ligand_num_edges = 0.0
    coeff_receptor_num_nodes = 0.0
    coeff_receptor_num_edges = 0.11
    coeff_num_cross_edges = 0.25
    total_memory = (
        coeff_ligand_num_nodes * data["ligand"].num_nodes
        + coeff_ligand_num_edges * data["ligand", "ligand"].num_edges
        + coeff_receptor_num_nodes * data["receptor"].num_nodes
        + coeff_receptor_num_edges * data["receptor", "receptor"].num_edges
        + coeff_num_cross_edges * num_cross_edges
    )
    if use_bias:
        bias = 430.5
        return total_memory + bias
    else:
        return total_memory

def estimate_size(g):
    n1, n2, n3, n4 = (
        g["ligand"].num_nodes,
        g["ligand", "ligand"].num_edges,
        g["receptor"].num_nodes,
        g["receptor", "receptor"].num_edges,
    )
    # estimate the upper bound of the number of cross edges
    # the number of cross edges roughly increases w.r.t. the diffusion step t (sampled from uniform(0,1))
    # the empirical formula here is from the polynomial fitting
    # the scaling constant is to help remove the outliers above the upper bound estimation.
    n5 = num_cross_edge_upper_bound_estimate(n1, n2, n3, n4)
    total_memory = estimate_memory_usage(g, n5,
                                         use_bias=False)
    return total_memory


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
                logging.warning(f"sample {sample.name} has size larger than max size {self.max_total_size}, skipping")
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
