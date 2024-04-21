# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Train the AlphaFold model class at bionemo.model.protein.openfold.openfold_model

Typical usage example:
    cd $BIONEMO_HOME
    python examples/protein/openfold/train.py --config=openfold_initial_training

Notes:
    See examples/protein/openfold/conf/openfold_initial_training.yaml
    for parameters and settings for this script.
"""

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from nemo.core.config import hydra_runner
from nemo.core.optim.lr_scheduler import register_scheduler
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.data import OpenProteinSetPreprocess, PDBMMCIFPreprocess
from bionemo.data.preprocess.protein.postprocess import OpenFoldSampleCreator
from bionemo.model.protein.openfold.checkpoint_utils import load_pt_checkpoint
from bionemo.model.protein.openfold.lr_scheduler import AlphaFoldLRScheduler
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.protein.openfold.optim_hub import enable_mlperf_optim
from bionemo.model.protein.openfold.utils.logging_utils import log_with_nemo_at_level
from bionemo.model.protein.openfold.utils.nemo_exp_manager_utils import isolate_last_checkpoint
from bionemo.model.utils import setup_trainer


torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False


@hydra_runner(config_path="conf", config_name="openfold_initial_training")
def main(cfg) -> None:
    cfg = instantiate(cfg)
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing required keys in config:\n{missing_keys}")

    register_scheduler(name='AlphaFoldLRScheduler', scheduler=AlphaFoldLRScheduler, scheduler_params=None)

    np.random.seed(cfg.model.seed)
    pl.seed_everything(cfg.model.seed)

    if cfg.get('do_preprocess', False):
        ops_preprocessor = OpenProteinSetPreprocess(
            dataset_root_path=cfg.model.data.dataset_path, **cfg.model.data.prepare.open_protein_set
        )
        pdb_mmcif_preprocessor = PDBMMCIFPreprocess(
            dataset_root_path=cfg.model.data.dataset_path, **cfg.model.data.prepare.pdb_mmcif
        )

        ops_preprocessor.prepare(**cfg.model.data.prepare.open_protein_set_actions)
        pdb_mmcif_preprocessor.prepare(**cfg.model.data.prepare.pdb_mmcif_actions)

    if cfg.model.data.prepare.create_sample:
        sample_creator = OpenFoldSampleCreator(
            dataset_root_path=cfg.model.data.dataset_path, **cfg.model.data.prepare.sample
        )
        sample_creator.prepare(cfg.model.data.prepare.sample_pdb_chain_ids)

    if cfg.get('do_training', False) or cfg.get('do_validation', False):
        filenames_to_keep, filenames_to_rename = isolate_last_checkpoint(cfg)
        log_with_nemo_at_level(
            f"""
            examples/protein/openfold/train.py:main(),
            checkpoint filenames_to_keep={filenames_to_keep}
            checkpoint filenames_to_rename={filenames_to_rename}
            """
        )
        trainer = setup_trainer(cfg, callbacks=[])
        enable_mlperf_optim(cfg.model)
        if cfg.get('restore_from_path', None):
            # TODO: consider blocking restore if stage is not 'fine-tune'
            alphafold = AlphaFold.restore_from(
                restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer
            )
            alphafold.setup_training_data(cfg.model.train_ds)
            alphafold.setup_validation_data(cfg.model.validation_ds)
        else:
            alphafold = AlphaFold(cfg=cfg.model, trainer=trainer)
            if cfg.get('torch_restore', None):
                load_pt_checkpoint(model=alphafold, checkpoint_path=cfg.torch_restore)

        if cfg.get('do_validation', False):
            trainer.validate(alphafold)
        if cfg.get('do_training', False):
            trainer.fit(alphafold)


if __name__ == '__main__':
    main()
