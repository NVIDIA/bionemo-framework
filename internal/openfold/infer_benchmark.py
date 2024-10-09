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
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from hydra.utils import instantiate
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bionemo.data.protein.openfold.datahub import get_structured_paths
from bionemo.data.protein.openfold.datasets import PredictDataset
from bionemo.data.protein.openfold.helpers import collate
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.writer import PredictionPDBWriter
from bionemo.model.utils import setup_trainer


# The following benchmark inference script is used to infer over CAMEO targets
# as structured in BioNeMo Service Benchmarking
# https://gitlab-master.nvidia.com/clara-discovery/bionemo-service-benchmarking


def read_benchmarking_data(benchmark_dir_path: str, allow_missing_msa: bool) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(benchmark_dir_path, "meta.csv"), names=["pdb_id", "length", "seq"])
    msa_path = os.path.join(benchmark_dir_path, "msa")
    meta["msa"] = meta.apply(lambda row: glob.glob(f'{msa_path}/{row["pdb_id"]}*.a3m'), axis=1)
    # benchmarking usually assumes full data will be used, if any of the sequences has no
    # MSA in the msa catalog, raise an error if allow_missing_msa is not turned on
    # otherwise issue a warning
    missing_msa_indices = np.where(~meta["msa"].astype(bool))[0]  # FIXME: where
    if missing_msa_indices.size:
        pdb_id_missing_msa = meta.iloc[missing_msa_indices]["pdb_id"].tolist()
        pdb_id_missing_msa = " ".join(pdb_id_missing_msa)
        msg = f"The following sequences has no MSA provided: {pdb_id_missing_msa}. "
        if allow_missing_msa:
            logging.warning(msg)
        else:
            raise Exception(msg + "If this is expected, use allow_missing_msa flag to ignore the warning ")
    return meta


@hydra_runner(config_path="conf", config_name="infer_benchmark")
def main(cfg) -> None:
    cfg = instantiate(cfg)
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing required keys in config:\n{missing_keys}")

    b_df = read_benchmarking_data(cfg.benchmark_dir_path, cfg.allow_missing_msa)
    sequences = b_df["seq"].to_list()
    seq_names = b_df["pdb_id"].to_list()
    msa_filepaths = b_df["msa"].to_list()
    np.random.seed(cfg.model.seed)
    pl.seed_everything(cfg.model.seed)

    writer = PredictionPDBWriter(cfg.results_path)
    trainer = setup_trainer(cfg, callbacks=[writer])

    if cfg.get("restore_from_path", None):
        # TODO: consider blocking restore if stage is not 'fine-tune'
        alphafold = AlphaFold.restore_from(
            restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer
        )
    else:
        alphafold = AlphaFold(cfg=cfg.model, trainer=trainer)
        if cfg.get("torch_restore", None):
            import torch

            alphafold.load_state_dict(torch.load(cfg.torch_restore))

    ds_paths = get_structured_paths(cfg.model.data)
    ds = PredictDataset(
        sequences=sequences,
        seq_names=seq_names,
        pdb_mmcif_chains_filepath=ds_paths.mmcif_chains,
        pdb_mmcif_dicts_dirpath=ds_paths.mmcif_dicts,
        pdb_obsolete_filepath=ds_paths.obsolete_filepath,
        template_hhr_filepaths=cfg.model.data.template_hhr_filepaths,
        msa_a3m_filepaths=msa_filepaths,
        generate_templates_if_missing=cfg.model.data.generate_templates_if_missing,
        pdb70_database_path=cfg.model.data.pdb70_path,
        cfg=cfg.model,
    )

    dl = DataLoader(
        ds, batch_size=cfg.model.micro_batch_size, num_workers=cfg.model.data.num_workers, collate_fn=collate
    )
    trainer.predict(alphafold, dl, return_predictions=False)


if __name__ == "__main__":
    main()
