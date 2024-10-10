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


import pickle
from pathlib import PosixPath
from typing import List, Optional

import numpy as np
import torch
from nemo.utils import logging
from pytorch_lightning.callbacks import Callback

import bionemo.data.protein.openfold.residue_constants as rc
from bionemo.data.protein.openfold.protein import Protein


class PredictionPDBWriter(Callback):
    def __init__(self, result_path: str, force: bool = False):
        """Takes inference output, converts it to Protein instance and writes
        to a PDB file.

        Args:
            result_path (str): directory path to result output.
            force: (bool): whether to overwrite results. Default to false.
        """

        self.result_path = PosixPath(result_path)
        self.result_path.mkdir(exist_ok=True, parents=True)
        self.force = force

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict):
            batch = [batch]
            outputs = [outputs]
        for input_dict, output_dict in zip(batch, outputs):
            try:
                name = input_dict["seq_name"][0]
            except KeyError:
                name = input_dict["seq_index"][0]

            unrelaxed_pdb_filepath = self.result_path / f"{name}.pdb"
            if unrelaxed_pdb_filepath.exists() and not self.force:
                logging.warning(f"Writer target {unrelaxed_pdb_filepath} exists. Skip overwriting.")
                continue

            aatype = input_dict["aatype"][0, :, 0].cpu().numpy()
            final_atom_positions = output_dict["final_atom_positions"][0].cpu().numpy()
            final_atom_mask = output_dict["final_atom_mask"][0].cpu().numpy()
            residue_index = input_dict["residue_index"][0, :, 0].cpu().numpy()
            b_factors = np.repeat(
                output_dict["plddt"][0].cpu().numpy()[:, None],
                repeats=rc.ATOM_TYPE_NUM,
                axis=-1,
            )
            unrelaxed_protein = Protein.from_prediction(
                aatype=aatype,
                final_atom_positions=final_atom_positions,
                final_atom_mask=final_atom_mask,
                residue_index=residue_index,
                b_factors=b_factors,
            )
            unrelaxed_pdb_string = unrelaxed_protein.to_pdb_string()

            with open(unrelaxed_pdb_filepath, "w") as f:
                f.write(unrelaxed_pdb_string)


class PredictionFeatureWriter(Callback):
    """Dump features from inference output"""

    def __init__(self, result_path: str, outputs: Optional[List] = None, force: bool = False):
        """Takes inference output and writes downstream task features to a pickle file.

        Args:
            result_path (str): directory path to result output.
            force: (bool): whether to overwrite results: Default to false.
            outputs: (Optional[List[str]]): list of keys to be written from inference output. Common options are single, msa, pair and sm_single.
        """
        self.result_path = PosixPath(result_path)
        self.result_path.mkdir(exist_ok=True, parents=True)
        self.force = force
        self.outputs = outputs if outputs else []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict):
            batch = [batch]
            outputs = [outputs]

        for input_dict, output_dict in zip(batch, outputs):
            try:
                name = input_dict["seq_name"][0]
            except KeyError:
                name = input_dict["seq_index"][0]

            feature_filepath = self.result_path / f"{name}.pkl"
            if feature_filepath.exists() and not self.force:
                logging.warning(f"Writer target {feature_filepath} exists. Skip overwriting.")
                continue

            features = {}
            for k in self.outputs:
                try:
                    v = output_dict[k]
                    if torch.is_tensor(v):
                        v = v.cpu().numpy()
                    features[k] = v
                except KeyError:
                    raise KeyError(
                        f'{", ".join(output_dict.keys())} are available for downstream features but ' f'{k} is given.'
                    )

            with feature_filepath.open("wb") as f:
                pickle.dump(features, f)
