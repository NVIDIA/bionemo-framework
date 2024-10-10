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


import logging
import os
import pathlib
import tempfile
from contextlib import ExitStack
from typing import Callable, Dict, Union
from uuid import uuid4

import numpy as np
from hydra import compose, initialize
from pytorch_lightning import Trainer
from pytriton.decorators import batch
from pytriton.model_config import Tensor
from pytriton.triton import Triton
from torch.utils.data import DataLoader

from bionemo.data.protein.openfold.datahub import get_structured_paths
from bionemo.data.protein.openfold.datasets import PredictDataset
from bionemo.data.protein.openfold.helpers import collate
from bionemo.model.protein.openfold.checkpoint_utils import load_pt_checkpoint
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.writer import PredictionPDBWriter
from bionemo.model.utils import setup_trainer
from bionemo.triton.utils import write_tempfiles_from_str_list
from bionemo.utils.tests import (
    BioNemoSearchPathConfig,
    register_searchpath_config_plugin,
    update_relative_config_dir,
)


OpenFoldInferFn = Callable[[np.ndarray, np.ndarray], Dict[str, np.ndarray]]

THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PREPEND_CONFIG_DIR = os.path.join(THIS_FILE_DIR, "../conf")


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


def make_openfold_infer_fn(
    output_dir: Union[str, pathlib.Path], model: AlphaFold, trainer: Trainer, cfg
) -> OpenFoldInferFn:
    @batch
    def _infer_fn(sequences_batch: np.ndarray, msa_a3m_file_contents_batch: np.ndarray) -> Dict[str, np.ndarray]:
        with ExitStack() as stack:  # clean up after every inference call
            # decode and unbatch inputs
            sequences_batch = np.char.decode(sequences_batch.astype("bytes"), "utf-8").tolist()
            msa_a3m_file_contents_batch = np.char.decode(msa_a3m_file_contents_batch.astype("bytes"), "utf-8").tolist()

            sequences = []
            msa_a3m_file_contents = []
            for sequences_, msa_a3m_file_contents_ in zip(sequences_batch, msa_a3m_file_contents_batch):
                sequences.extend(sequences_)
                msa_a3m_file_contents.extend(msa_a3m_file_contents_)

            # clean msa_a3m_file_contents and write to temporary files
            temp_files = []

            for msa_a3m_file_contents_ in msa_a3m_file_contents:
                msa_a3m_file_contents_ = [
                    msa_a3m_file_content_ for msa_a3m_file_content_ in msa_a3m_file_contents_ if msa_a3m_file_content_
                ]  # drop empty entries
                temp_files_ = write_tempfiles_from_str_list(msa_a3m_file_contents_, exit_stack=stack, delete=True)
                temp_files.append(temp_files_)

            msa_a3m_filepaths = [[temp_file.name for temp_file in temp_files_] for temp_files_ in temp_files]

            # create temporary seq_names
            seq_names = [str(uuid4()) for _ in sequences]

            # inference
            ds_paths = get_structured_paths(cfg.model.data)
            ds = PredictDataset(
                sequences=sequences,
                seq_names=seq_names,
                pdb_mmcif_chains_filepath=ds_paths.mmcif_chains,
                pdb_mmcif_dicts_dirpath=ds_paths.mmcif_dicts,
                pdb_obsolete_filepath=ds_paths.obsolete_filepath,
                template_hhr_filepaths=cfg.model.data.template_hhr_filepaths,
                msa_a3m_filepaths=msa_a3m_filepaths,
                generate_templates_if_missing=cfg.model.data.generate_templates_if_missing,
                pdb70_database_path=cfg.model.data.pdb70_database_path,
                cfg=cfg.model,
            )

            dl = DataLoader(
                ds, batch_size=cfg.model.micro_batch_size, num_workers=cfg.model.data.num_workers, collate_fn=collate
            )
            trainer.predict(model, dl, return_predictions=False)

            # collect inference output
            output_pdb_strings = []

            for seq_name in seq_names:
                pdb_filename = os.path.join(output_dir, f"{seq_name}.pdb")
                with open(pdb_filename, "r") as fopen:
                    output_pdb_string = fopen.read()

                os.remove(pdb_filename)
                output_pdb_strings.append(output_pdb_string)

            return {
                "output_pdb_string": np.char.encode([output_pdb_strings], "utf-8"),
            }

    return _infer_fn


def main() -> None:
    # configuration
    cfg = get_cfg(PREPEND_CONFIG_DIR, config_name="infer", config_path="../conf")

    with ExitStack() as stack:  # clean up after server
        # setup model for inference
        logging.info("Loading model")

        tmp_output_folder = stack.enter_context(tempfile.TemporaryDirectory())

        writers = [PredictionPDBWriter(tmp_output_folder, cfg.force)]
        if cfg.model.downstream_task.outputs:
            raise NotImplementedError("Downstream_task outputs are not supported on jupyter notebook yet.")
        trainer = setup_trainer(cfg, callbacks=writers)

        if cfg.get("restore_from_path", None):
            alphafold = AlphaFold.restore_from(
                restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer
            )
        elif cfg.get("torch_restore", None):
            alphafold = AlphaFold(cfg=cfg.model, trainer=trainer)
            load_pt_checkpoint(model=alphafold, checkpoint_path=cfg.torch_restore)
        else:
            raise ValueError(
                "No checkpoint has been provided neither via restore_from_path nor torch_restore. \
                            Inference was not ran."
            )

        # setup for triton
        triton = stack.enter_context(Triton())
        _infer_fn = make_openfold_infer_fn(tmp_output_folder, alphafold, trainer, cfg)

        triton.bind(
            model_name="bionemo_openfold",
            infer_func=_infer_fn,
            inputs=[  # assume inference one-structure-at-a-time
                Tensor(name="sequences_batch", dtype=bytes, shape=(-1,)),
                Tensor(name="msa_a3m_file_contents_batch", dtype=bytes, shape=(-1, -1)),
            ],
            outputs=[
                Tensor(name="output_pdb_string", dtype=bytes, shape=(-1,)),
            ],
        )
        logging.info("Serving model")
        triton.serve()


if __name__ == "__main__":
    main()
