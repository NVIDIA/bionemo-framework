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


from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bionemo.data.dna.enformer.genome_interval import FastaDataset
from bionemo.model.dna.enformer import Enformer
from bionemo.model.dna.enformer.writer import FastaRecordsWriter
from bionemo.model.utils import setup_trainer


@hydra_runner(config_path="conf", config_name="enformer_infer")
def main(cfg):
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    writer_callback = FastaRecordsWriter(
        output_dir=cfg.model.predictions_output_path, mode=cfg.model.predictions_write_mode
    )
    trainer = setup_trainer(cfg, callbacks=[writer_callback])
    enformer = Enformer.restore_from(restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer)
    ds = FastaDataset(fasta_file=cfg.infer_path, context_length=cfg.model.context_length)
    dl = DataLoader(ds, batch_size=cfg.model.micro_batch_size, num_workers=cfg.model.data.num_workers)
    trainer.predict(enformer, dl, return_predictions=False)


if __name__ == "__main__":
    main()
