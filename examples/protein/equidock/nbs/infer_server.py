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


import argparse
import logging
import os
import pathlib
from typing import Dict

import numpy as np
import torch
from biopandas.pdb import PandasPdb
from hydra import compose, initialize
from omegaconf.omegaconf import OmegaConf
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from bionemo.model.protein.equidock.infer import EquiDockInference
from bionemo.model.protein.equidock.utils.remove_clashes import remove_clashes
from bionemo.model.protein.equidock.utils.train_utils import batchify_and_create_hetero_graphs_inference
from bionemo.model.utils import initialize_distributed_parallel_state
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


DATA_NAMES = ["dips", "db5"]
# TODO
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PREPEND_CONFIG_DIR = os.path.join(THIS_FILE_DIR, "../conf")

torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_cfg(prepend_config_path: str, config_name: str, config_path: str = "conf"):
    prepend_config_path = pathlib.Path(prepend_config_path)

    class TestSearchPathConfig(BioNemoSearchPathConfig):
        def __init__(self) -> None:
            super().__init__()
            self.prepend_config_dir = update_relative_config_dir(prepend_config_path, THIS_FILE_DIR)

    register_searchpath_config_plugin(TestSearchPathConfig)
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    return cfg


def parse_args():
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=DATA_NAMES,
        default="dips",
        help=("Dataset the model is trained on",),
    )
    return parser.parse_args()


def main() -> None:
    # Parse command line arguments
    args = parse_args()

    # Setup model in eval mode
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name="infer", config_path="../conf")
    cfg.data.data_name = args.model

    initialize_distributed_parallel_state(
        local_rank=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_model_parallel_split_rank=0,
    )
    model = EquiDockInference(cfg=cfg)
    model.eval()

    logging.info("\n\n************** Inference configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    @batch
    def _infer_fn(
        ligand_filename: np.ndarray, receptor_filename: np.ndarray, out_filename: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Inference function for EquiDock inference server"""
        ligand_filename = np.char.decode(ligand_filename.astype("bytes"), "utf-8")
        ligand_filename = ligand_filename.squeeze(1).tolist()[0]

        receptor_filename = np.char.decode(receptor_filename.astype("bytes"), "utf-8")
        receptor_filename = receptor_filename.squeeze(1).tolist()[0]

        out_filename = np.char.decode(out_filename.astype("bytes"), "utf-8")
        out_filename = out_filename.squeeze(1).tolist()[0]

        # Create ligand and receptor graphs and arrays
        (
            ligand_graph,
            receptor_graph,
            bound_ligand_repres_nodes_loc_clean_array,
            _,
        ) = model.model.create_ligand_receptor_graphs_arrays(ligand_filename, receptor_filename, cfg.data)

        # Create a batch of a single DGL graph
        batch_hetero_graph = batchify_and_create_hetero_graphs_inference(ligand_graph, receptor_graph)

        batch_hetero_graph = batch_hetero_graph.to(model.device)
        (
            model_ligand_coors_deform_list,
            model_keypts_ligand_list,
            model_keypts_receptor_list,
            all_rotation_list,
            all_translation_list,
        ) = model(batch_hetero_graph)

        rotation = all_rotation_list[0].detach().cpu().numpy()  # this is output of model
        translation = all_translation_list[0].detach().cpu().numpy()  # this is output of model

        new_residues = (rotation @ bound_ligand_repres_nodes_loc_clean_array.T).T + translation
        assert (
            np.linalg.norm(new_residues - model_ligand_coors_deform_list[0].detach().cpu().numpy()) < 1e-1
        ), "Norm mismtach"

        # Get initial ligand position
        ppdb_ligand = PandasPdb().read_pdb(ligand_filename)
        unbound_ligand_all_atoms_pre_pos = (
            ppdb_ligand.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy().squeeze().astype(np.float32)
        )
        unbound_ligand_new_pos = (rotation @ unbound_ligand_all_atoms_pre_pos.T).T + translation

        if cfg.get("postprocess", None) is not None and cfg.postprocess.remove_clashes:
            logging.info("Removing clashes!")
            max_iteration = cfg.postprocess.max_iteration
            unbound_ligand_new_pos = remove_clashes(
                unbound_ligand_new_pos,
                receptor_filename,
                max_iteration,
                cfg.postprocess.min_clash_loss,
                cfg.postprocess.lr,
                cfg.postprocess.fast_optimizer,
                cfg.postprocess.half_precision,
            )

            ppdb_ligand.df["ATOM"][["x_coord", "y_coord", "z_coord"]] = (
                unbound_ligand_new_pos  # unbound_ligand_new_pos
            )

        else:
            ppdb_ligand.df["ATOM"][["x_coord", "y_coord", "z_coord"]] = (
                unbound_ligand_new_pos  # unbound_ligand_new_pos
            )

        ppdb_ligand.to_pdb(path=out_filename, records=["ATOM"], gz=False)

        response = {"generated": np.char.encode(["5"], "utf-8").reshape((1, -1))}

        return response

    with Triton() as triton:
        logging.info("Loading model")
        triton.bind(
            model_name="bionemo_model",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="ligand_filename", dtype=bytes, shape=(1,)),
                Tensor(name="receptor_filename", dtype=bytes, shape=(1,)),
                Tensor(name="out_filename", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="generated", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=10),
        )
        logging.info("Serving model")
        triton.serve()


if __name__ == "__main__":
    main()
