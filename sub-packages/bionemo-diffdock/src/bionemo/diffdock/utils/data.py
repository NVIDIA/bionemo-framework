# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import random
from typing import Any, Callable, Generator, Iterable, List, Union

import numpy as np
import torch
from nemo.utils import logging
from omegaconf.listconfig import ListConfig
from torch_geometric.data import HeteroData
from torch_geometric.data.batch import Batch
from torch_geometric.loader.dataloader import Collater


def num_cross_edge_upper_bound_estimate(n1, n2, n3, n4):
    terms = [[4.92, "ligand_ligand"], [0.0118, "receptor_receptor"], [0.0401, "ligand", "receptor_receptor"]]
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
    total_memory = estimate_memory_usage(g, n5, use_bias=False)
    return total_memory


class SizeAwareBatching:
    """A WebDataset composable to do batching based on sample size"""

    def __init__(
        self,
        max_total_size: int,
        size_fn: Callable[[HeteroData], int],
        collate_fn: Callable[[List[Any]], Any] = Collater(dataset=[], follow_batch=None, exclude_keys=None),
        no_single_sample: bool = True,
    ):
        self.max_total_size = max_total_size
        self.size_fn = size_fn
        self.collate_fn = collate_fn
        self.cached_sizes = {}
        self.no_single_sample = no_single_sample

    def __call__(self, data: Batch) -> Generator[Union[Batch, List[HeteroData]], None, None]:
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
                if self.no_single_sample and len(batch) <= 1:
                    # memory size requirement is met but there is less than 2
                    # samples in the batch so skip
                    batch = [sample]
                    batch_size = sample_size
                    continue
                if self.collate_fn is not None:
                    batch = self.collate_fn(batch)
                yield batch

                batch = [sample]
                batch_size = sample_size


class SelectPoseAndLabelData:
    """A WebDataset composable to select one ligand poses from multiple ones and
    label confidence model training data by RMSD threshold"""

    def __init__(
        self,
        rmsd_classification_cutoff: Union[float, ListConfig],
        samples_per_complex: int,
        balance: bool,
        all_atoms: bool,
        seed: int = 0,
    ):
        """constructor

        Args:
            rmsd_classification_cutoff (Union[float, ListConfig]): RMSD classification cutoff(s)
            samples_per_complex (int): how many inference runs were done per complex
            balance (bool): whether to do balance sampling
            all_atoms (bool): whether the confidence model is all-atom
            seed (int): random number generator seed

        Returns:

        """
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.samples_per_complex = samples_per_complex
        self.balance = balance
        self.all_atoms = all_atoms
        self._seed = seed

    def __call__(self, data: Iterable) -> Generator[HeteroData, None, None]:
        """Map the input data iterator to another one that label the input data

        Args:
            data (Iterable): Input data iterator

        Returns:

        """
        random.seed(self._seed)
        for (complex_graph,) in data:
            positions, rmsds = complex_graph.ligand_data

            if self.balance:
                if isinstance(self.rmsd_classification_cutoff, ListConfig):
                    raise ValueError("a list for rmsd_classification_cutoff can only be used with balance=False")
                # FIXME: should allow random.seed
                label = random.randint(0, 1)
                success = rmsds < self.rmsd_classification_cutoff
                n_success = np.count_nonzero(success)
                if label == 0 and n_success != self.samples_per_complex:
                    # sample negative complex
                    sample = random.randint(0, self.samples_per_complex - n_success - 1)
                    lig_pos = positions[~success][sample]
                    complex_graph["ligand"].pos = torch.from_numpy(lig_pos)
                else:
                    # sample positive complex
                    if n_success > 0:  # if no successful sample returns the matched complex
                        sample = random.randint(0, n_success - 1)
                        lig_pos = positions[success][sample]
                        complex_graph["ligand"].pos = torch.from_numpy(lig_pos)
                complex_graph.y = torch.tensor(label).float()
            else:
                sample = random.randint(0, self.samples_per_complex - 1)
                complex_graph["ligand"].pos = torch.from_numpy(positions[sample])
                ids = (rmsds[sample] < self.rmsd_classification_cutoff).astype(int)
                complex_graph.y = torch.tensor(ids).float().unsqueeze(0)
                if isinstance(self.rmsd_classification_cutoff, ListConfig):
                    complex_graph.y_binned = torch.tensor(
                        np.logical_and(
                            rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],
                            rmsds[sample] >= [0] + self.rmsd_classification_cutoff,
                        ),
                        dtype=torch.float,
                    ).unsqueeze(0)
                    complex_graph.y = (
                        torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff[0]).unsqueeze(0).float()
                    )
                complex_graph.rmsd = torch.tensor(rmsds[sample]).unsqueeze(0).float()

            complex_graph["ligand"].node_t = {
                "tr": 0 * torch.ones(complex_graph["ligand"].num_nodes),
                "rot": 0 * torch.ones(complex_graph["ligand"].num_nodes),
                "tor": 0 * torch.ones(complex_graph["ligand"].num_nodes),
            }
            complex_graph["receptor"].node_t = {
                "tr": 0 * torch.ones(complex_graph["receptor"].num_nodes),
                "rot": 0 * torch.ones(complex_graph["receptor"].num_nodes),
                "tor": 0 * torch.ones(complex_graph["receptor"].num_nodes),
            }
            if self.all_atoms:
                complex_graph["atom"].node_t = {
                    "tr": 0 * torch.ones(complex_graph["atom"].num_nodes),
                    "rot": 0 * torch.ones(complex_graph["atom"].num_nodes),
                    "tor": 0 * torch.ones(complex_graph["atom"].num_nodes),
                }
            complex_graph.complex_t = {
                "tr": 0 * torch.ones(1),
                "rot": 0 * torch.ones(1),
                "tor": 0 * torch.ones(1),
            }
            yield complex_graph
