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


import copy
import glob
import math
import os
import pickle
import random
from functools import lru_cache, partial
from typing import Dict, Generator, Iterable, List, Optional, Union

import numpy as np
import torch
import webdataset as wds
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm

from bionemo.data.diffdock.docking_dataset import (
    DataSplit,
    HeteroGraphDataConfig,
    ProteinLigandDockingDataset,
    get_heterograph_path_from_data_config,
    read_strings_from_txt,
)
from bionemo.data.diffdock.webdataset_utils import pickles_to_tars
from bionemo.model.molecule.diffdock.infer import DiffDockModelInference
from bionemo.model.molecule.diffdock.utils.ddp import get_rank
from bionemo.model.molecule.diffdock.utils.diffusion import get_t_schedule
from bionemo.model.molecule.diffdock.utils.diffusion import (
    t_to_sigma as t_to_sigma_compl,
)
from bionemo.model.molecule.diffdock.utils.sampling import randomize_position, sampling


class SelectPose:
    """A WebDataset composable to select one ligand poses from multiple ones and label confidence model training data by RMSD
    threshold"""

    def __init__(
        self,
        rmsd_classification_cutoff: Union[float, ListConfig],
        samples_per_complex: int,
        balance: bool,
        all_atoms: bool,
    ):
        """constructor

        Args:
            rmsd_classification_cutoff (Union[float, ListConfig]): RMSD classification cutoff(s)
            samples_per_complex (int): how many inference runs were done per complex
            balance (bool): whether to do balance sampling
            all_atoms (bool): whether the confidence model is all-atom

        Returns:

        """
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.samples_per_complex = samples_per_complex
        self.balance = balance
        self.all_atoms = all_atoms

    def __call__(self, data: Iterable) -> Generator[HeteroData, None, None]:
        """Map the input data iterator to another one that label the input data

        Args:
            data (Iterable): Input data iterator

        Returns:

        """
        for (complex_graph,) in data:
            positions, rmsds = complex_graph.ligand_data

            if self.balance:
                if isinstance(self.rmsd_classification_cutoff, ListConfig):
                    raise ValueError("a list for rmsd_classification_cutoff can only be used with balance=False")
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
                complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff).float().unsqueeze(0)
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


class ConfidenceDataset(Dataset):
    # TODO, eventually build this class on top of bionemo.data.diffdock.docking_dataset.ProteinLigandDockingDataset

    def __init__(
        self,
        data_config: HeteroGraphDataConfig,
        rmsd_classification_cutoff: Union[float, ListConfig],
        samples_per_complex: int,
        balance: bool,
        mode: DataSplit,
        score_model_name: str,
        num_workers: int = 1,
    ):
        """Dataset with protein-ligand complex graphs, and also ligand poses/rmsds from reverse diffusion.

        Args:
            data_config (HeteroGraphDataConfig): Protein-Ligand complex hetero graph data config data class, refer to HeteroGraphDataConfig for more details
            rmsd_classification_cutoff (Union[float, ListConfig]): RMSD classification cutoff, can be a float number of a list of float numbers.
            samples_per_complex (int): number of ligand poses to generate for each complex
            balance (bool): if sample good and bad ligand poses in a balanced way.
            mode (DataSplit): mode of the dataset, could be DataSplit("train"), DataSplit("validation") or DataSplit("test").
            score_model_name (str): the name of the score model which will be used to run reverse diffusion to generate ligand poses.
            num_workers (int): number of workers to do data preprocessing. Defaults to 1.
        """
        super(ConfidenceDataset, self).__init__()

        self.data_config = data_config
        self.balance = balance
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.samples_per_complex = samples_per_complex
        self.mode = mode
        self.split_path = data_config.split_path
        self.num_workers = num_workers
        self.score_model_name = score_model_name

        self.complex_graphs_cache = get_heterograph_path_from_data_config(
            self.data_config
        )  # path for complex graph pyd files
        self.ligand_poses_cache = os.path.join(
            self.data_config.cache_path,
            f"model_{self.score_model_name}_limit_{self.data_config.limit_complexes}",
        )  # path for ligand poses and rmsds data pyd files

        self.split_filename = os.path.splitext(os.path.basename(self.split_path))[0]
        self.split_cache_path = f"{self.complex_graphs_cache}_INDEX{self.split_filename}"

        if not (os.path.exists(self.split_cache_path) and os.listdir(self.split_cache_path)):
            logging.info(
                f"Complex graphs split webdataset tar files do not exist yet: {self.split_cache_path}. "
                "Will use load_confidence_dataset() to build"
            )

        self.webdataset_urls: Optional[List] = None
        self.webdataset_fname_suffix_in_tar = "heterodata.pyd"

    @lru_cache(maxsize=None)
    def len(self):
        return len(read_strings_from_txt(self.split_path))

    def build_complex_graphs(self):
        """
        Use ProteinLigandDockingDataset's preprocessing workflow and build the
        HeteroData for the complex graphs, which are later used to combine with
        the reverse diffusion results to create the training data for the
        confidence model
        """
        complex_dataset = ProteinLigandDockingDataset(data_config=self.data_config, num_workers=self.num_workers)
        complex_dataset.build_complex_graphs()
        if complex_dataset.full_cache_path != self.complex_graphs_cache:
            raise RuntimeError(
                f"The directory of *.heterodata.pyd for "
                f"ConfidenceDataset ({self.complex_graphs_cache}) "
                f" doesn't match that of the "
                f"ProteinLigandDockingDataset "
                f"({complex_dataset.full_cache_path}), "
                f"the latter of which is used to construct the "
                f"former"
            )

    @staticmethod
    def _merge_complex_graph_with_ligand_data(complex_graph: HeteroData, positions_rmsds_dict: Dict) -> HeteroData:
        name = complex_graph.name
        if name not in positions_rmsds_dict:
            raise KeyError(f"ligand poses for complex: {name} missing")
        complex_graph.ligand_data = positions_rmsds_dict[name]
        return complex_graph

    def load_confidence_dataset(self):
        # split tar files not prepared or the folder is empty
        if not (os.path.exists(self.split_cache_path) and os.listdir(self.split_cache_path)):
            logging.info(f"Preparing complex graphs split webdataset tar files to: {self.split_cache_path}")

            local_rank = get_rank()
            if local_rank == 0:
                logging.info(f"Loading ligand poses and rmsds from: {self.ligand_poses_cache}")

            # load the *.LigandData.pyd for the name, position and rmsds
            # resulted from the reverse diffusion process (output of
            # generate_ligand_poses())
            positions_rmsds_dict = {}
            if os.path.exists(self.ligand_poses_cache) and os.listdir(self.ligand_poses_cache):
                for file in os.listdir(self.ligand_poses_cache):
                    try:
                        name, lig_poses, rmsds = pickle.load(open(os.path.join(self.ligand_poses_cache, file), "rb"))
                        positions_rmsds_dict[name] = (lig_poses, rmsds)
                    except Exception as e:
                        logging.error(
                            f"Failed to read pickle: {os.path.join(self.ligand_poses_cache, file)} due to error {e}"
                        )
                        continue
            else:
                raise RuntimeError(
                    f"Ligand poses and rmsds pickle files can't "
                    f"be found in directory {self.ligand_poses_cache}, "
                    f"which are required to create WebDataset tar files. "
                    "Use generate_ligand_poses() to build."
                )

            logging.info(
                f"Number of RMSDs and positions for the complex graphs: {len(positions_rmsds_dict)}",
            )

            # Get the list of complex name for the current split (train, val or
            # test)
            split_complex_names = read_strings_from_txt(self.split_path)

            # Filter for those complexes which has the reverse diffusion results
            # (in positions_rmsds_dict)
            dataset_names = [name for name in positions_rmsds_dict.keys() if name in frozenset(split_complex_names)]

            if self.data_config.limit_complexes > 0:
                dataset_names = dataset_names[: self.data_config.limit_complexes]

            logging.info(f"Loading complex graphs from: {self.complex_graphs_cache}")

            # Combine the rmsds and complex graph input to the reverse
            # diffusion into a new complex graph (the lambda function and
            # _merge_complex_graph_with_ligand_data) and tar them up for training
            # confidence model
            if os.path.exists(self.complex_graphs_cache) and os.listdir(self.complex_graphs_cache):
                # output the WebDataset tar files
                pickles_to_tars(
                    self.complex_graphs_cache,
                    "HeteroData.pyd",
                    dataset_names,
                    self.split_cache_path,
                    "heterographs",
                    lambda complex_graph: {
                        "__key__": complex_graph.name.replace(".", "-"),
                        self.webdataset_fname_suffix_in_tar: pickle.dumps(
                            self._merge_complex_graph_with_ligand_data(complex_graph, positions_rmsds_dict)
                        ),
                    },
                    self.data_config.min_num_shards,
                )
            else:
                raise RuntimeError(
                    f"Confidence dataset's complex graph pickle files can't "
                    f"be found in directory {self.complex_graphs_cache}, "
                    f"which are required to create WebDataset tar files. "
                    "Use build_complex_graphs() to build."
                )

        # glob the resulting tar files
        self.webdataset_urls = glob.glob(os.path.join(self.split_cache_path, "heterographs-*.tar"))
        if len(self.webdataset_urls) == 0:
            raise RuntimeError(f"No WebDataset tar file is found in {self.split_cache_path}")

    def get(self, idx):
        raise NotImplementedError("Using webdataset as backend which does not support indexing")

    def generate_ligand_poses(self, score_model: DiffDockModelInference, score_data_config: DictConfig) -> None:
        # TODO: ideally sampling should be parallel
        logging.info(
            f"Run reverse diffusion sampling to generate ligand poses and save to: {self.ligand_poses_cache}."
        )

        if get_rank() == 0:
            os.makedirs(self.ligand_poses_cache, exist_ok=True)

        score_data_config = HeteroGraphDataConfig.init_from_hydra_config(score_data_config)
        score_data_config.split_path = (
            score_model.cfg.model.get(f"{self.mode.name}_ds").split_val
            if "val" in self.mode.name
            else score_model.cfg.model.get(f"{self.mode.name}_ds").get(f"split_{self.mode.name}")
        )
        score_data_config.num_conformers = score_data_config.num_conformers

        score_model_complex_graphs_cache = get_heterograph_path_from_data_config(score_data_config)

        score_model_complex_graphs_split_cache_path = f"{score_model_complex_graphs_cache}_INDEX{self.split_filename}"

        if not (
            os.path.exists(score_model_complex_graphs_split_cache_path)
            and os.listdir(score_model_complex_graphs_split_cache_path)
        ):
            raise RuntimeError(
                f"Complex graphs for score model to run reverse diffusion does not exist here: {score_model_complex_graphs_split_cache_path}. "
                "Try using ProteinLigandDockingDataset.build_complex_graphs() to do score model complex graphs preprocessing first."
            )
        webdataset_urls = glob.glob(os.path.join(score_model_complex_graphs_split_cache_path, "heterographs-*.tar"))

        t_to_sigma = partial(t_to_sigma_compl, cfg=score_model.cfg.model)

        dataset = (
            wds.WebDataset(webdataset_urls, shardshuffle=False, nodesplitter=wds.split_by_node)
            .decode()
            .extract_keys(".heterodata.pyd")
            .batched(1, collation_fn=Collater(dataset=None, follow_batch=None, exclude_keys=None))
        )

        tr_schedule = get_t_schedule(denoising_inference_steps=score_model.cfg.model.denoising_inference_steps)
        rot_schedule = tr_schedule
        tor_schedule = tr_schedule

        logging.info(f"common t (noising time) schedule {tr_schedule}")

        # skip the preprocessed complexes saved in the folder.
        processed_names = {
            filename[: -len(".LigandData.pyd")]
            for filename in os.listdir(self.ligand_poses_cache)
            if filename.endswith(".LigandData.pyd")
            and os.path.getsize(os.path.join(self.ligand_poses_cache, filename)) > 0
        }

        split_complex_names = set(read_strings_from_txt(self.split_path))
        complex_names = split_complex_names - processed_names  # complexes for preprocessing in the split

        if len(complex_names) == 0:
            logging.info(
                f"Ligand poses have been generated for all complexes in split: {self.split_path}, "
                f"and saved in {self.ligand_poses_cache}, skipping."
            )
            return

        for idx, (orig_complex_graph,) in enumerate(tqdm(dataset)):
            # here batch size is 1, only one complex
            if orig_complex_graph.name[0] not in complex_names:
                continue
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
            randomize_position(
                data_list,
                score_model.cfg.model.diffusion.no_torsion,
                False,
                score_model.cfg.model.diffusion.tr_sigma_max,
            )

            predictions_list = None
            failed_convergence_counter = 0
            while predictions_list is None and failed_convergence_counter <= 5:
                try:
                    predictions_list, confidences = sampling(
                        data_list=data_list,
                        model=score_model.model.net,
                        denoising_inference_steps=score_model.cfg.model.denoising_inference_steps,
                        tr_schedule=tr_schedule,
                        rot_schedule=rot_schedule,
                        tor_schedule=tor_schedule,
                        device=score_model.device,
                        t_to_sigma=t_to_sigma,
                        model_cfg=score_model.cfg.model,
                        batch_size=10,
                    )
                except Exception as e:
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        logging.warning(
                            f"| WARNING: Sampling failed 5 times for {orig_complex_graph.name[0]} with error {e}"
                        )
            if failed_convergence_counter > 5:
                predictions_list = data_list
            if score_model.cfg.model.diffusion.no_torsion:
                orig_complex_graph["ligand"].orig_pos = (
                    orig_complex_graph["ligand"].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
                )

            filterHs = torch.not_equal(predictions_list[0]["ligand"].x[:, 0], 0).cpu().numpy()

            if isinstance(orig_complex_graph["ligand"].orig_pos, list):
                orig_complex_graph["ligand"].orig_pos = orig_complex_graph["ligand"].orig_pos[0]

            ligand_pos = np.asarray(
                [complex_graph["ligand"].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list]
            )
            orig_ligand_pos = np.expand_dims(
                orig_complex_graph["ligand"].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(),
                axis=0,
            )
            rmsds = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
            assert len(orig_complex_graph.name) == 1

            name = orig_complex_graph.name[0]
            lig_poses = np.asarray([complex_graph["ligand"].pos.cpu().numpy() for complex_graph in predictions_list])
            with open(os.path.join(self.ligand_poses_cache, f"{name}.LigandData.pyd"), "wb") as f:
                pickle.dump([name, lig_poses, rmsds], f)


def diffdock_build_confidence_dataset(
    data_config: DictConfig,
    split_config: DictConfig,
    mode: DataSplit = DataSplit("train"),
) -> ConfidenceDataset:
    """Build dataset for protein-ligand complexes for DiffDock confidence model training

    Args:
        data_config (DictConfig): hydra config cfg.data section
        split_config (DictConfig): hydra config cfg.model.[train_ds, validation_ds, test_ds] section
        mode (DataSplit, optional): mode of the dataset, could be DataSplit("train"), DataSplit("validation") or DataSplit("test"). Defaults to DataSplit("train").

    Returns:
        ConfidenceDataset: DiffDock Confidence Dataset
    """

    config = HeteroGraphDataConfig.init_from_hydra_config(data_config)
    config.split_path = split_config.split_val if "val" in mode.name else split_config.get(f"split_{mode.name}")
    config.min_num_shards = split_config.get("min_num_shards")

    return ConfidenceDataset(
        data_config=config,
        num_workers=split_config.num_workers,
        rmsd_classification_cutoff=split_config.rmsd_classification_cutoff,
        samples_per_complex=split_config.samples_per_complex,
        balance=split_config.balance,
        mode=mode,
        score_model_name=data_config.score_model_name,
    )
